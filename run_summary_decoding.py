#!/usr/bin/env python3
"""
Summary decoding: predict summary embeddings from brain data via ridge regression.

For each (ROI, horizon) pair this script:
1. Encodes per-TR summaries with a sentence-transformer embedding model
2. Applies response delays to fMRI data (brain -> embedding regression)
3. Fits a ridge decoder: delayed brain responses -> embeddings
4. Predicts embeddings for held-out test stories
5. Evaluates embedding prediction quality (per-dim correlation, cosine similarity)
6. Optionally inverts predicted embeddings back to text using vec2text

Usage
-----
  # Per-ROI decoding across all horizons (timescale hierarchy analysis)
  python run_summary_decoding.py \\
      --subject S1 \\
      --local-compute-mode \\
      --summary-model gpt-4o-mini \\
      --feature-model embedding \\
      --per-roi \\
      --nboots 5

  # Single full-frontal run for one horizon
  python run_summary_decoding.py \\
      --subject S1 \\
      --local-compute-mode \\
      --summary-model gpt-4o-mini \\
      --summary-horizons 20 \\
      --voxels-from-rois \\
      --nboots 5
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config
from utils_resp import get_resp
from utils_ridge.ridge import bootstrap_ridge

import run_summaries_encoding as rse

TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summary_decoding")
np.random.seed(42)

EMBEDDING_MODELS = {
    "embedding": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "gtr-base": ("sentence-transformers/gtr-t5-base", 768),
}

ROI_FILES = ["BA_10.json", "BA_6.json", "BA_8.json", "BA_9_46.json", "BROCA.json"]


def load_encoder(model_name, device="cpu"):
    """Load a sentence-transformer embedding model."""
    from sentence_transformers import SentenceTransformer

    log.info("Loading encoder: %s", model_name)
    model = SentenceTransformer(model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())
    log.info("  Embedding dim: %d", dim)
    return model, dim


def embed_summaries(model, texts, emb_dim, batch_size=64):
    """Encode a list of summary strings into embeddings."""
    vecs = np.zeros((len(texts), emb_dim), dtype=np.float32)
    nonempty_idx = [i for i, t in enumerate(texts) if t.strip()]
    if not nonempty_idx:
        return vecs
    enc = model.encode(
        [texts[i] for i in nonempty_idx],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    vecs[nonempty_idx] = enc
    return vecs


def build_embedding_targets(stories, texts_by_story, resp_lengths,
                            encoder, emb_dim, batch_size=64):
    """Embed per-TR summaries, trim to response-aligned TRs, and stack."""
    trimmed = []
    for story in stories:
        texts = texts_by_story[story]
        if len(texts) <= TRIM_START + TRIM_END:
            raise ValueError(
                f"Story '{story}' has only {len(texts)} summary TRs; "
                f"need more than {TRIM_START + TRIM_END} to apply trimming."
            )
        vecs = embed_summaries(encoder, texts, emb_dim, batch_size=batch_size)
        t = vecs[TRIM_START:-TRIM_END]
        expected = resp_lengths[story]
        if t.shape[0] != expected:
            raise ValueError(
                f"Story '{story}': {len(texts)} summary TRs trims to "
                f"{t.shape[0]}, but response has {expected} TRs."
            )
        trimmed.append(t)
    return np.vstack(trimmed).astype(np.float32)


def build_brain(subject, stories, vox, response_root):
    """Load brain responses and z-score across time."""
    resp = get_resp(subject, stories, stack=True, vox=vox, response_root=response_root).astype(np.float32)
    mean = resp.mean(0)
    std = resp.std(0)
    std[std == 0] = 1
    resp = np.nan_to_num((resp - mean) / std).astype(np.float32)
    return resp, mean, std


def pca_reduce_brain(train_brain, test_brain, n_components):
    """PCA-reduce delayed brain matrices to make ridge SVD tractable."""
    from sklearn.decomposition import PCA

    n_comp = min(n_components, train_brain.shape[0] - 1, train_brain.shape[1])
    log.info(
        "PCA-reducing brain: %d -> %d components",
        train_brain.shape[1], n_comp,
    )
    pca = PCA(n_components=n_comp, random_state=42)
    train_reduced = pca.fit_transform(train_brain).astype(np.float32)
    var_expl = pca.explained_variance_ratio_.sum()
    log.info("  Variance retained: %.1f%%", 100 * var_expl)

    test_reduced = pca.transform(test_brain).astype(np.float32)

    tr_mean = train_reduced.mean(0)
    tr_std = train_reduced.std(0)
    tr_std[tr_std == 0] = 1
    train_reduced = np.nan_to_num((train_reduced - tr_mean) / tr_std).astype(np.float32)
    test_reduced = np.nan_to_num((test_reduced - tr_mean) / tr_std).astype(np.float32)

    log.info("  Reduced: %s, %s", train_reduced.shape, test_reduced.shape)
    return train_reduced, test_reduced, pca


def zscore_embeddings(emb_train, emb_test=None):
    """Z-score embedding targets using training statistics."""
    mean = emb_train.mean(0)
    std = emb_train.std(0)
    std[std == 0] = 1
    train_z = np.nan_to_num((emb_train - mean) / std).astype(np.float32)
    if emb_test is not None:
        test_z = np.nan_to_num((emb_test - mean) / std).astype(np.float32)
    else:
        test_z = None
    return train_z, test_z, mean, std


def predict_embeddings(brain_test, wt, emb_mean, emb_std):
    """Predict embeddings from brain data and undo z-scoring."""
    pred_z = brain_test.dot(wt)
    return (pred_z * emb_std + emb_mean).astype(np.float32)


def eval_embedding_quality(true_emb, pred_emb):
    """Per-TR cosine similarity and per-dimension Pearson correlation."""
    n = true_emb.shape[0]
    cosines = np.zeros(n, dtype=np.float32)
    pearsons = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t, p = true_emb[i], pred_emb[i]
        tn, pn = np.linalg.norm(t), np.linalg.norm(p)
        if tn > 0 and pn > 0:
            cosines[i] = np.dot(t, p) / (tn * pn)
        tc, pc = t - t.mean(), p - p.mean()
        tcn, pcn = np.linalg.norm(tc), np.linalg.norm(pc)
        if tcn > 0 and pcn > 0:
            pearsons[i] = np.dot(tc, pc) / (tcn * pcn)

    dim_corrs = np.array([
        np.corrcoef(true_emb[:, j], pred_emb[:, j])[0, 1]
        for j in range(true_emb.shape[1])
    ], dtype=np.float32)
    dim_corrs = np.nan_to_num(dim_corrs)

    return cosines, pearsons, dim_corrs


def retrieval_metrics(true_emb, pred_emb):
    """Retrieval-style evaluation on held-out TRs.

    For each test TR i, compute similarity between pred_emb[i] and all true_emb[*],
    and check whether the true match (i) is ranked highest.

    Returns:
      top1: fraction where correct TR is rank-1
      mrr: mean reciprocal rank of the correct TR
      mean_rank: mean 1-indexed rank of the correct TR (lower is better)
    """
    # Normalize embeddings for cosine similarity
    t = true_emb.astype(np.float32, copy=False)
    p = pred_emb.astype(np.float32, copy=False)

    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    t_norm[t_norm == 0] = 1
    p_norm[p_norm == 0] = 1
    t_unit = t / t_norm
    p_unit = p / p_norm

    # Similarity matrix: (T x T)
    sim = p_unit @ t_unit.T
    diag = np.diag(sim)

    # Rank of correct item = 1 + number of items with strictly greater similarity
    # (ties count as correct if argmax hits the diagonal; for rank we use strict >)
    ranks = 1 + (sim > diag[:, None]).sum(axis=1)

    top1 = float((ranks == 1).mean())
    mrr = float((1.0 / ranks).mean())
    mean_rank = float(ranks.mean())
    return top1, mrr, mean_rank


def load_roi_voxels(ba_subject_dir, total_voxels):
    """Load per-ROI voxel indices from BA JSON files."""
    rois = {}
    for fname in ROI_FILES:
        fpath = os.path.join(ba_subject_dir, fname)
        if not os.path.exists(fpath):
            log.warning("ROI file not found: %s", fpath)
            continue
        with open(fpath, encoding="utf-8") as f:
            jdata = json.load(f)
        key = list(jdata.keys())[0]
        indices = np.array(jdata[key], dtype=int)
        indices = indices[indices < total_voxels]
        roi_name = fname.replace(".json", "")
        rois[roi_name] = np.sort(indices)
        log.info("  ROI %s: %d voxels", roi_name, len(indices))
    return rois


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument(
        "--sessions", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    parser.add_argument("--holdout-stories", nargs="+", default=None)
    parser.add_argument("--holdout-count", type=int, default=rse.DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--no-story-holdout", action="store_true")
    parser.add_argument("--local-compute-mode", action="store_true")
    parser.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    parser.add_argument("--local-cache-root", default=str(rse.DEFAULT_LOCAL_CACHE_ROOT))
    parser.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-horizons", nargs="+", type=int, default=None)
    parser.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    parser.add_argument(
        "--feature-model", default="embedding",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Sentence-transformer model for embedding summaries.",
    )
    parser.add_argument(
        "--brain-pca", type=int, default=0,
        help="PCA-reduce brain matrix to this many components. 0 to disable (default).",
    )
    parser.add_argument("--nboots", type=int, default=5)
    parser.add_argument("--single-alpha", type=float, default=None)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--voxels-from-rois", action="store_true",
        help="Use all frontal voxels (BA_full_frontal.json).",
    )
    parser.add_argument(
        "--per-roi", action="store_true",
        help="Run decoding separately for each frontal ROI.",
    )
    parser.add_argument("--all-voxels", action="store_true")
    parser.add_argument(
        "--skip-vec2text", action="store_true", default=True,
        help="Skip vec2text text inversion (default: skip).",
    )
    parser.add_argument("--do-vec2text", action="store_true",
                        help="Enable vec2text text inversion.")
    parser.add_argument("--max-decode-trs", type=int, default=100)
    parser.add_argument("--vec2text-steps", type=int, default=20)
    parser.add_argument("--vec2text-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="summary_decoding_results")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_decoding_for_roi(
    roi_name, train_brain, test_brain,
    train_emb_z, test_emb_z, test_emb,
    emb_mean, emb_std,
    args, horizon, emb_model_tag, summary_model,
    out_dir, split_tag, pca_tag,
):
    """Run ridge decoding for one ROI and one horizon. Returns results dict."""
    alphas = np.array([args.single_alpha]) if args.single_alpha else config.ALPHAS
    nchunks = int(np.ceil(train_brain.shape[0] / 5 / config.CHUNKLEN))

    log.info(
        "  Ridge: brain (%s) -> emb (%s), %d boots",
        train_brain.shape, train_emb_z.shape, args.nboots,
    )

    wt, valphas, bscorrs = bootstrap_ridge(
        train_brain, train_emb_z,
        alphas=alphas, nboots=args.nboots,
        chunklen=config.CHUNKLEN, nchunks=nchunks,
        use_corr=True,
    )
    cv_corrs = bscorrs.mean(2).max(0)
    del valphas, bscorrs

    pred_emb = predict_embeddings(test_brain, wt, emb_mean, emb_std)
    cosines, pearsons, dim_corrs = eval_embedding_quality(test_emb, pred_emb)
    top1, mrr, mean_rank = retrieval_metrics(test_emb, pred_emb)

    log.info(
        "  %s h%d: dim_r=%.4f, top1=%.3f, mrr=%.3f (cos=%.4f)",
        roi_name, horizon,
        dim_corrs.mean(), top1, mrr, cosines.mean(),
    )

    results = {
        "roi": roi_name,
        "horizon": horizon,
        "mean_cosine": float(cosines.mean()),
        "median_cosine": float(np.median(cosines)),
        "mean_pearson": float(pearsons.mean()),
        "mean_dim_corr": float(dim_corrs.mean()),
        "frac_dim_positive": float((dim_corrs > 0).mean()),
        "retrieval_top1": float(top1),
        "retrieval_mrr": float(mrr),
        "retrieval_mean_rank": float(mean_rank),
    }

    safe_sm = rse.sanitize_name(summary_model)
    label = f"decode__{emb_model_tag}__{safe_sm}__h{horizon}__{roi_name}{pca_tag}__{split_tag}"
    out_path = out_dir / f"{label}.npz"

    np.savez(
        out_path,
        cosines=cosines, pearsons=pearsons, dim_corrs=dim_corrs,
        cv_corrs=cv_corrs,
        pred_embeddings=pred_emb, true_embeddings=test_emb,
        summary_horizon=np.array(horizon),
        roi=np.array(roi_name),
        condition_label=np.array(label),
        brain_pca_components=np.array(args.brain_pca if args.brain_pca else 0),
        retrieval_top1=np.array(top1, dtype=np.float32),
        retrieval_mrr=np.array(mrr, dtype=np.float32),
        retrieval_mean_rank=np.array(mean_rank, dtype=np.float32),
    )
    log.info("  Saved %s", out_path.name)

    del wt, pred_emb
    gc.collect()
    return results


def main():
    args = parse_args()
    if args.do_vec2text:
        args.skip_vec2text = False

    mounted_root = rse.configure_local_compute_mode(args) if args.local_compute_mode else None

    stories = rse.load_story_list(args)
    train_stories, test_stories = rse.split_story_list(stories, args)
    if not test_stories:
        raise ValueError("Decoding requires held-out test stories.")
    log.info("Stories: %d train, %d test", len(train_stories), len(test_stories))

    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    if not summaries_dir.is_dir():
        raise FileNotFoundError(f"No summaries directory at {summaries_dir}")

    summary_index = rse.build_summary_index(summaries_dir)
    summary_model = rse.resolve_summary_model(summary_index, stories, args.summary_model)
    summary_horizons = rse.resolve_summary_horizons(
        summary_index, stories, summary_model, args.summary_horizons,
    )
    log.info("Summary model: %s, horizons: %s", summary_model, summary_horizons)

    if args.local_compute_mode:
        local_cache_root = Path(args.local_cache_root).expanduser().resolve()
        if mounted_root is not None and rse.is_relative_to(local_cache_root, mounted_root):
            raise ValueError(
                f"--local-cache-root must not live inside the mounted tree: {local_cache_root}"
            )
        response_data_train_dir = rse.stage_local_response_cache(
            args.subject, stories,
            mounted_data_train_dir=config.DATA_TRAIN_DIR,
            cache_root=local_cache_root,
        )
    else:
        response_data_train_dir = config.DATA_TRAIN_DIR

    train_resp_lengths, total_voxels = rse.load_resp_info(
        args.subject, train_stories, data_train_dir=response_data_train_dir,
    )
    test_resp_lengths, _ = rse.load_resp_info(
        args.subject, test_stories, data_train_dir=response_data_train_dir,
    )
    log.info("Total voxels: %d", total_voxels)

    uts_id = rse.SUBJECT_TO_UTS.get(args.subject)
    ba_subject_dir = os.path.join(args.ba_dir, uts_id) if uts_id else None

    # Determine ROI sets to run
    if args.per_roi:
        if not ba_subject_dir or not os.path.isdir(ba_subject_dir):
            raise FileNotFoundError(f"No BA directory at {ba_subject_dir}")
        roi_dict = load_roi_voxels(ba_subject_dir, total_voxels)
    elif args.voxels_from_rois:
        frontal_path = os.path.join(ba_subject_dir, "BA_full_frontal.json")
        with open(frontal_path, encoding="utf-8") as f:
            frontal = json.load(f)
        frontal_voxels = list(frontal.values())[0]
        vox = np.sort(np.array(frontal_voxels, dtype=int))
        vox = vox[vox < total_voxels]
        roi_dict = {"full_frontal": vox}
    elif args.all_voxels:
        roi_dict = {"all": np.arange(total_voxels)}
    else:
        vox, _ = rse.load_voxel_set(args.subject, total_voxels)
        roi_dict = {"default": vox}

    # Load embedding model
    emb_model_name, emb_dim = EMBEDDING_MODELS[args.feature_model]
    encoder, emb_dim = load_encoder(emb_model_name, device="cpu")
    emb_model_tag = rse.sanitize_name(args.feature_model)

    out_dir = rse.resolve_output_dir(args, mounted_root=mounted_root) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    split_tag = rse.make_story_split_tag(train_stories, test_stories)
    pca_tag = f"__pca{args.brain_pca}" if args.brain_pca and args.brain_pca > 0 else ""

    # Pre-embed all horizons (shared across ROIs)
    emb_cache = {}
    for horizon in summary_horizons:
        texts_by_story = {}
        for story in stories:
            path = summary_index[(story, summary_model, horizon)]
            loaded = rse.load_summary_texts(
                path=path, expected_story=story,
                expected_model=summary_model, expected_horizon=horizon,
            )
            texts_by_story[story] = loaded["texts"]

        log.info("Embedding h%d summaries...", horizon)
        train_emb = build_embedding_targets(
            train_stories, texts_by_story, train_resp_lengths,
            encoder, emb_dim, batch_size=args.embed_batch_size,
        )
        test_emb = build_embedding_targets(
            test_stories, texts_by_story, test_resp_lengths,
            encoder, emb_dim, batch_size=args.embed_batch_size,
        )
        train_emb_z, test_emb_z, emb_mean, emb_std = zscore_embeddings(train_emb, test_emb)
        emb_cache[horizon] = (train_emb_z, test_emb_z, test_emb, emb_mean, emb_std)
        log.info("  h%d: train %s, test %s", horizon, train_emb.shape, test_emb.shape)
        del train_emb

    all_results = []

    for roi_name, roi_vox in roi_dict.items():
        log.info("=" * 60)
        log.info("ROI: %s (%d voxels)", roi_name, len(roi_vox))
        log.info("=" * 60)

        log.info("Loading brain data for %s...", roi_name)
        train_brain, train_brain_mean, train_brain_std = build_brain(
            args.subject, train_stories, roi_vox, response_data_train_dir,
        )
        test_brain_raw = get_resp(
            args.subject, test_stories, stack=True, vox=roi_vox,
            response_root=response_data_train_dir,
        ).astype(np.float32)
        test_brain = np.nan_to_num(
            (test_brain_raw - train_brain_mean) / train_brain_std
        ).astype(np.float32)
        del test_brain_raw
        log.info("  Brain: train %s, test %s", train_brain.shape, test_brain.shape)

        if args.brain_pca and args.brain_pca > 0:
            train_brain_pca, test_brain_pca, _ = pca_reduce_brain(
                train_brain, test_brain, args.brain_pca,
            )
            del train_brain, test_brain
            gc.collect()
        else:
            train_brain_pca = train_brain
            test_brain_pca = test_brain

        for horizon in summary_horizons:
            train_emb_z, test_emb_z, test_emb, emb_mean, emb_std = emb_cache[horizon]

            res = run_decoding_for_roi(
                roi_name, train_brain_pca, test_brain_pca,
                train_emb_z, test_emb_z, test_emb,
                emb_mean, emb_std,
                args, horizon, emb_model_tag, summary_model,
                out_dir, split_tag, pca_tag,
            )
            all_results.append(res)

        del train_brain_pca, test_brain_pca
        gc.collect()

    # Save summary CSV
    if all_results:
        csv_path = out_dir / f"decoding_summary__{emb_model_tag}{pca_tag}__{split_tag}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        log.info("Summary CSV: %s", csv_path)

        log.info("\n%s", "=" * 70)
        log.info("DECODING SUMMARY (%s) — test-set metrics", args.feature_model)
        log.info("%-15s %6s %10s %10s %10s %10s", "ROI", "h", "dim_r", "top1", "mrr", "cosine")
        log.info("-" * 55)
        for r in all_results:
            log.info(
                "%-15s %6d %10.4f %10.3f %10.3f %10.4f",
                r["roi"], r["horizon"],
                r["mean_dim_corr"], r["retrieval_top1"], r["retrieval_mrr"], r["mean_cosine"],
            )

    log.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
