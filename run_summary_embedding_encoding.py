#!/usr/bin/env python3
"""
Summary-embedding context encoding analysis.

Like run_context_encoding.py but uses LLM-generated summaries (from
generate_summaries/) as features instead of raw word-level LLM hidden states.
Only the sentence-transformer embedding model (all-MiniLM-L6-v2) is used.

For each context length (20, 50, 200, 500 words seen), a fixed set of
*anchor* TR indices is derived from the ctx20 file (every 20th entry starting
from the first TR with n_words_seen >= 20).  All four context-length files are
then sampled at those same TR indices, giving aligned feature matrices of
identical length per story.  Each summary text is embedded with
all-MiniLM-L6-v2, interpolated to TR times, and a ridge encoding model is
trained (features -> voxels).

Usage
-----
  python run_summary_embedding_encoding.py --subject S1 --voxels-from-rois

  # Custom summaries directory
  python run_summary_embedding_encoding.py --subject S1 --voxels-from-rois \\
      --summaries-dir /path/to/outputs

  # Subset of context lengths
  python run_summary_embedding_encoding.py --subject S1 \\
      --context-lengths 20 200 --voxels-from-rois
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoding"))
import config
from utils_resp import get_resp
from utils_stim import get_story_wordseqs
from utils_ridge.ridge import bootstrap_ridge
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed

CONTEXT_LENGTHS = [20, 50, 200, 500]
SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
LLM_MODEL_TAG = "gpt-4o-mini"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summary_embedding_encoding")
np.random.seed(42)


# ---------------------------------------------------------------------------
# Summary loading & anchor-index selection
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load all lines of a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_summary_path(summaries_dir, story, ctx_len, model_tag=LLM_MODEL_TAG):
    """Return the path to a single story/context-length JSONL file."""
    fname = f"{story}.{model_tag}.ctx{ctx_len}.jsonl"
    return os.path.join(summaries_dir, fname)


def select_anchor_indices(rows_ctx20):
    """Determine the anchor row indices from the ctx20 JSONL rows.

    Rules
    -----
    1. The first anchor is the row with the smallest row index where
       n_words_seen >= 20.
    2. From that first anchor, take every 20th subsequent row
       (i.e. anchor, anchor+20, anchor+40, ...).

    Returns a list of integer row indices (0-based into ``rows_ctx20``).
    """
    first = None
    for i, row in enumerate(rows_ctx20):
        if row["n_words_seen"] >= 20:
            first = i
            break

    if first is None:
        raise ValueError("No row with n_words_seen >= 20 found in ctx20 file.")

    n = len(rows_ctx20)
    anchors = list(range(first, n, 20))
    return anchors


def extract_summary_features_for_story(
    story, summaries_dir, anchors, context_lengths, st_model
):
    """For one story, embed summaries at anchor indices for each context length.

    Parameters
    ----------
    anchors : list[int]
        Row indices (0-based) selected from the ctx20 file.
    context_lengths : list[int]
        Which context lengths to process (must include 20).

    Returns
    -------
    vecs_by_ctx : dict[int -> (n_anchors, embed_dim) np.ndarray]
    times_by_ctx : dict[int -> list[float]]
        tr_time_s values at anchor positions for each context length.
    """
    vecs_by_ctx = {}
    times_by_ctx = {}

    for ctx_len in context_lengths:
        path = get_summary_path(summaries_dir, story, ctx_len)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Summary file not found: {path}\n"
                "Check --summaries-dir and that summaries have been generated."
            )
        rows = load_jsonl(path)

        if len(rows) < max(anchors) + 1:
            raise ValueError(
                f"{path}: only {len(rows)} rows but anchor index "
                f"{max(anchors)} was requested."
            )

        texts = [rows[i]["summary"] for i in anchors]
        times = [rows[i]["tr_time_s"] for i in anchors]

        # Empty summaries (no words seen yet) become blank strings;
        # sentence-transformer returns a near-zero vector for these.
        texts = [t if t else "" for t in texts]

        embeddings = st_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        vecs_by_ctx[ctx_len] = embeddings
        times_by_ctx[ctx_len] = times

    return vecs_by_ctx, times_by_ctx


# ---------------------------------------------------------------------------
# Feature -> TR conversion  (mirrors features_to_tr in run_context_encoding)
# ---------------------------------------------------------------------------

def features_to_tr(vecs_by_story, times_by_story, word_seqs, stories):
    """Lanczos-interpolate summary embeddings to TR times, z-score, add delays.

    Parameters
    ----------
    vecs_by_story : dict[story -> (n_anchors, D) array]
    times_by_story : dict[story -> list[float]]
        Anchor tr_time_s values (used as the input sample times for Lanczos).
    word_seqs : dict[story -> WordSeq]
        Provides .tr_times for each story (target interpolation grid).
    """
    ds_vecs = {}
    for story in stories:
        anchor_times = np.array(times_by_story[story], dtype=np.float64)
        tr_times = word_seqs[story].tr_times
        ds_vecs[story] = lanczosinterp2D(
            vecs_by_story[story],
            anchor_times,
            tr_times,
        )

    ds_mat = np.vstack([
        ds_vecs[story][5 + config.TRIM:-config.TRIM] for story in stories
    ])
    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    ds_mat = np.nan_to_num((ds_mat - r_mean) / r_std)
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    return del_mat


# ---------------------------------------------------------------------------
# ROI summary  (unchanged from run_context_encoding)
# ---------------------------------------------------------------------------

def load_ba_rois(ba_subject_dir):
    import glob as globmod
    rois = {}
    for path in sorted(globmod.glob(os.path.join(ba_subject_dir, "*.json"))):
        fname = os.path.basename(path)
        if fname == "BA_full_frontal.json":
            continue
        with open(path) as f:
            d = json.load(f)
        for key, indices in d.items():
            rois[key] = indices
    return rois


def print_roi_summary(all_corrs, ba_subject_dir, vox, context_lengths):
    rois = load_ba_rois(ba_subject_dir)
    region_names = sorted(rois.keys())
    global_to_local = {int(g): i for i, g in enumerate(vox)}

    local_rois = {}
    for rn in region_names:
        local_rois[rn] = np.array(
            [global_to_local[v] for v in rois[rn] if v in global_to_local],
            dtype=int,
        )

    hdr = f"  {'condition':<25s}"
    for rn in region_names:
        hdr += f"  {rn + f' ({len(local_rois[rn])})':>25s}"
    hdr += f"  {'all':>10s}"

    print("\n" + "=" * len(hdr))
    print("  Per-ROI mean encoding correlation (summary embeddings)")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for ctx_len in context_lengths:
        label = f"summary_ctx{ctx_len}"
        if label not in all_corrs:
            continue
        corrs = all_corrs[label]
        row = f"  {label:<25s}"
        for rn in region_names:
            idx = local_rois[rn]
            mean_r = corrs[idx].mean() if len(idx) > 0 else float("nan")
            row += f"  {mean_r:25.4f}"
        row += f"  {corrs.mean():10.4f}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Voxel helpers  (unchanged from run_context_encoding)
# ---------------------------------------------------------------------------

def load_voxel_set(subject, all_voxels):
    em_path = os.path.join(config.MODEL_DIR, subject, "encoding_model_perceived.npz")
    if os.path.exists(em_path):
        em = np.load(em_path)
        vox = em["voxels"]
        log.info("Loaded %d language-responsive voxels from pretrained model", len(vox))
        return vox, True
    log.warning(
        "No pretrained model found at %s — using all %d voxels "
        "(may require a lot of RAM)", em_path, all_voxels
    )
    return np.arange(all_voxels), False


def chunked_bootstrap_ridge(rstim, rresp, chunk_size=10000, **kwargs):
    n_voxels = rresp.shape[1]
    if n_voxels <= chunk_size:
        wt, valphas, bscorrs = bootstrap_ridge(rstim, rresp, **kwargs)
        corrs = bscorrs.mean(2).max(0)
        del bscorrs
        return corrs, wt

    all_corrs = np.zeros(n_voxels)
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        log.info("  Voxel chunk [%d:%d] / %d", start, end, n_voxels)
        wt, valphas, bscorrs = bootstrap_ridge(
            rstim, rresp[:, start:end], **kwargs)
        all_corrs[start:end] = bscorrs.mean(2).max(0)
        del wt, bscorrs
    return all_corrs, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sessions", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--context-lengths", nargs="+", type=int,
                        default=CONTEXT_LENGTHS)
    parser.add_argument(
        "--summaries-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "generate_summaries", "outputs", "outputs",
        ),
        help="Directory containing per-story JSONL summary files "
             "(default: generate_summaries/outputs/outputs/)",
    )
    parser.add_argument(
        "--ba-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ba_indices"),
        help="Directory containing per-subject Brodmann area indices",
    )
    parser.add_argument("--nboots", type=int, default=config.NBOOTS)
    parser.add_argument("--single-alpha", type=float, default=None,
                        help="Use a fixed ridge alpha instead of cross-validated search")
    parser.add_argument("--output-dir", default="summary_embedding_results")
    parser.add_argument("--voxels-from-rois", action="store_true",
                        help="Restrict to frontal voxels from ba_indices/")
    parser.add_argument("--all-voxels", action="store_true",
                        help="Use all voxels (needs more RAM, uses chunking)")
    parser.add_argument("--voxel-chunk-size", type=int, default=10000)
    parser.add_argument("--save-weights", action="store_true",
                        help="Also save full regression weights (large files)")
    args = parser.parse_args()

    device = config.GPT_DEVICE
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load stories
    # ------------------------------------------------------------------
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json")) as f:
        sess_to_story = json.load(f)
    stories = []
    for s in args.sessions:
        stories.extend(sess_to_story[str(s)])
    log.info("Stories (%d): %s", len(stories), stories)

    # ------------------------------------------------------------------
    # Word sequences (for TR times) & fMRI responses
    # ------------------------------------------------------------------
    word_seqs = get_story_wordseqs(stories)
    rresp_full = get_resp(args.subject, stories, stack=True)
    log.info("Full response matrix: %s (TRs x voxels)", rresp_full.shape)

    # ------------------------------------------------------------------
    # Voxel selection
    # ------------------------------------------------------------------
    uts_id = SUBJECT_TO_UTS.get(args.subject)
    ba_subject_dir = os.path.join(args.ba_dir, uts_id) if uts_id else None

    if args.voxels_from_rois:
        if not ba_subject_dir or not os.path.isdir(ba_subject_dir):
            log.error("--voxels-from-rois: no BA directory at %s", ba_subject_dir)
            sys.exit(1)
        frontal_path = os.path.join(ba_subject_dir, "BA_full_frontal.json")
        with open(frontal_path) as f:
            frontal = json.load(f)
        frontal_voxels = list(frontal.values())[0]
        vox = np.sort(np.array(frontal_voxels, dtype=int))
        vox = vox[vox < rresp_full.shape[1]]
        log.info("Using %d frontal voxels from %s", len(vox), frontal_path)
    elif args.all_voxels:
        vox = np.arange(rresp_full.shape[1])
        log.info("Using ALL %d voxels (chunked processing)", len(vox))
    else:
        vox, _ = load_voxel_set(args.subject, rresp_full.shape[1])

    rresp = rresp_full[:, vox]
    del rresp_full
    log.info("Working response matrix: %s (TRs x voxels)", rresp.shape)

    # ------------------------------------------------------------------
    # Validate context lengths — 20 must be present (used for anchors)
    # ------------------------------------------------------------------
    context_lengths = list(args.context_lengths)
    if 20 not in context_lengths:
        log.warning("ctx20 not in --context-lengths; adding it to derive anchors "
                    "(its results will still be saved).")
        context_lengths = [20] + context_lengths

    # ------------------------------------------------------------------
    # Load sentence-transformer once
    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    log.info("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # ------------------------------------------------------------------
    # Per-story: derive anchors from ctx20, embed all context lengths
    # ------------------------------------------------------------------
    log.info("Deriving anchor indices and extracting summary embeddings...")

    # vecs_all[ctx_len][story] -> (n_anchors, D) array
    # times_all[ctx_len][story] -> list of float tr_time_s
    vecs_all = {c: {} for c in context_lengths}
    times_all = {c: {} for c in context_lengths}

    for story in stories:
        ctx20_path = get_summary_path(args.summaries_dir, story, 20)
        rows_ctx20 = load_jsonl(ctx20_path)
        anchors = select_anchor_indices(rows_ctx20)
        log.info("  %s: %d anchors (first at row %d, n_words_seen=%d)",
                 story, len(anchors), anchors[0],
                 rows_ctx20[anchors[0]]["n_words_seen"])

        vecs_story, times_story = extract_summary_features_for_story(
            story, args.summaries_dir, anchors, context_lengths, st_model
        )
        for ctx_len in context_lengths:
            vecs_all[ctx_len][story] = vecs_story[ctx_len]
            times_all[ctx_len][story] = times_story[ctx_len]

    del st_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = os.path.join(config.REPO_DIR, args.output_dir, args.subject)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Run ridge regression for each context length
    # ------------------------------------------------------------------
    all_corrs = {}

    for ctx_len in context_lengths:
        label = f"summary_ctx{ctx_len}"
        log.info("=" * 60)
        log.info("Condition: %s", label)
        log.info("=" * 60)

        log.info("Interpolating to TRs and adding FIR delays...")
        rstim = features_to_tr(vecs_all[ctx_len], times_all[ctx_len],
                               word_seqs, stories)
        log.info("Design matrix: %s (TRs x delayed features)", rstim.shape)

        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
        alphas = np.array([args.single_alpha]) if args.single_alpha else config.ALPHAS
        log.info("Bootstrap ridge (%d boots, chunklen=%d, nchunks=%d, alphas=%s)...",
                 args.nboots, config.CHUNKLEN, nchunks, alphas)

        corrs, wt = chunked_bootstrap_ridge(
            rstim, rresp,
            chunk_size=args.voxel_chunk_size,
            alphas=alphas,
            nboots=args.nboots,
            chunklen=config.CHUNKLEN,
            nchunks=nchunks,
            use_corr=True,
        )
        del rstim

        all_corrs[label] = corrs
        log.info("  mean r=%.4f | max r=%.4f | n(r>0.1)=%d",
                 corrs.mean(), corrs.max(), (corrs > 0.1).sum())

        save_dict = dict(
            corrs=corrs,
            voxels=vox,
            context_length=np.array(ctx_len),
            stories=np.array(stories),
        )
        if args.save_weights and wt is not None:
            save_dict["weights"] = wt

        np.savez(os.path.join(out_dir, f"{label}.npz"), **save_dict)
        del wt
        log.info("  -> saved %s/%s.npz", out_dir, label)

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = dict(all_corrs)
    summary["context_lengths"] = np.array(context_lengths)
    summary["voxels"] = vox
    np.savez(os.path.join(out_dir, "summary.npz"), **summary)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY — per-voxel encoding correlation (%d voxels)", len(vox))
    log.info("=" * 60)
    for label, corrs in all_corrs.items():
        log.info("  %-25s  mean=%.4f  max=%.4f  n(r>0.1)=%d",
                 label, corrs.mean(), corrs.max(), (corrs > 0.1).sum())

    if ba_subject_dir and os.path.isdir(ba_subject_dir):
        print_roi_summary(all_corrs, ba_subject_dir, vox, context_lengths)

    log.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
