#!/usr/bin/env python3
"""Per-lag encoding with concatenated MiniLM local-text + summary features.

Feature vector at response-aligned TR/chunk ``i``:

    [ MiniLM(1TR text at i),
      MiniLM(summary h20 at i),
      MiniLM(summary h50 at i),
      MiniLM(summary h200 at i) ]

The target and scoring match ``train_lag_encoding.py``:

    Y = full_frontal brain response at TR (i + lag)

for every lag in ``--lags`` (default 1..10), using story-grouped validation
and per-voxel RidgeCV. Outputs use the same ``lag_corrs.npz`` layout so
``analyze_lag_preference.py`` and ``plot_lag_flatmaps.py`` work unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from _shared import load_or_build_chunk_embeddings  # noqa: E402
from train_lag_encoding import (  # noqa: E402
    configure_data_root,
    fit_ridge_predict,
    load_full_frontal_voxels,
    load_stories,
    per_voxel_corr,
    split_stories,
    stack_lag,
    write_summary_csv,
)
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summary_combo_encoding")

SUBJECT_TO_UTS = rse.SUBJECT_TO_UTS
LOCAL_DEFAULT_BA_DIR = rse.LOCAL_DEFAULT_BA_DIR
LOCAL_DEFAULT_SUMMARIES_DIR = rse.LOCAL_DEFAULT_SUMMARIES_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, choices=sorted(SUBJECT_TO_UTS.keys()))
    p.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--chunk-trs", type=int, default=1)

    p.add_argument("--summary-model", default=None, help="Summary generator model (default: infer).")
    p.add_argument("--summary-horizons", nargs="+", type=int, default=[20, 50, 200])
    p.add_argument("--summaries-dir", default=str(LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer used for both 1TR text and summary text.",
    )
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument(
        "--embedding-cache-dir",
        default=str(THIS_DIR / "cache"),
        help="Cache directory for combo-feature embeddings.",
    )
    p.add_argument(
        "--one-tr-cache-dir",
        default=str(REPO_DIR / "27-04-expts" / "cache"),
        help="Existing cache root used for 1TR text embeddings.",
    )

    p.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0],
    )
    p.add_argument("--voxel-chunk-size", type=int, default=5_000)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ba-dir", default=str(LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--output-dir", default=str(THIS_DIR / "results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def cache_key(args: argparse.Namespace, stories: Sequence[str], resp_lengths: Dict[str, int]) -> str:
    payload = {
        "subject": args.subject,
        "stories": list(stories),
        "resp_lengths": {s: int(resp_lengths[s]) for s in stories},
        "summary_model": args.summary_model,
        "summary_horizons": list(args.summary_horizons),
        "embedding_model": args.embedding_model,
        "chunk_trs": int(args.chunk_trs),
        "max_lag": int(max(args.lags)),
        "feature_blocks": ["1tr_text", "summary_h20", "summary_h50", "summary_h200"],
        "version": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def make_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    htag = "-".join(str(h) for h in args.summary_horizons)
    return (
        f"{args.subject}__embedding-summary-combo-h{htag}"
        f"__lags{min(args.lags)}-{max(args.lags)}__chunk{args.chunk_trs}tr__seed{args.seed}"
    )


def load_or_build_summary_embeddings(
    args: argparse.Namespace,
    stories: Sequence[str],
    resp_lengths: Dict[str, int],
) -> Tuple[Dict[int, Dict[str, np.ndarray]], str, str]:
    """Return {horizon: {story: response-aligned embedding matrix}}."""
    cache_dir = Path(args.embedding_cache_dir).expanduser().resolve() / args.subject
    cache_dir.mkdir(parents=True, exist_ok=True)

    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    index = rse.build_summary_index(summaries_dir)
    summary_model = rse.resolve_summary_model(index, list(stories), args.summary_model)
    horizons = rse.resolve_summary_horizons(index, list(stories), summary_model, args.summary_horizons)
    if sorted(horizons) != sorted(args.summary_horizons):
        raise ValueError(f"Resolved summary horizons {horizons} != requested {args.summary_horizons}")
    args.summary_model = summary_model

    key = cache_key(args, stories, resp_lengths)
    cache_path = cache_dir / f"summary_combo_minilm__{summary_model}__h{'-'.join(map(str, horizons))}__{key}.pkl"
    if cache_path.is_file():
        log.info("Loading cached summary embeddings: %s", cache_path)
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload["summary_embeddings"], summary_model, str(cache_path)

    encoder = rse.SummaryEmbeddingEncoder(
        model_name=args.embedding_model,
        device="cpu",
        batch_size=int(args.embed_batch_size),
    )

    out: Dict[int, Dict[str, np.ndarray]] = {int(h): {} for h in horizons}
    try:
        for horizon in horizons:
            log.info("Embedding summaries: model=%s horizon=%s", summary_model, horizon)
            for story in stories:
                summary_path = index[(story, summary_model, int(horizon))]
                payload = rse.load_summary_texts(summary_path, story, summary_model, int(horizon))
                vecs = encoder.encode(payload["texts"]).astype(np.float32)
                trimmed = vecs[rse.TRIM_START : -rse.TRIM_END]
                expected = int(resp_lengths[story])
                if trimmed.shape[0] != expected:
                    raise ValueError(
                        f"{story} h{horizon}: summary embeddings trim to {trimmed.shape[0]}, "
                        f"but response has {expected} TRs."
                    )
                out[int(horizon)][story] = trimmed
    finally:
        encoder.close()

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "summary_embeddings": out,
                "summary_model": summary_model,
                "horizons": list(map(int, horizons)),
                "embedding_model": args.embedding_model,
                "stories": list(stories),
            },
            f,
        )
    log.info("Wrote summary embedding cache: %s", cache_path)
    return out, summary_model, str(cache_path)


def build_combo_embeddings(
    one_tr: Dict[str, np.ndarray],
    summary_by_horizon: Dict[int, Dict[str, np.ndarray]],
    stories: Sequence[str],
    horizons: Sequence[int],
) -> Dict[str, np.ndarray]:
    combo: Dict[str, np.ndarray] = {}
    for story in stories:
        n = int(one_tr[story].shape[0])
        blocks = [one_tr[story]]
        for horizon in horizons:
            summ = summary_by_horizon[int(horizon)][story]
            if summ.shape[0] < n:
                raise ValueError(f"{story} h{horizon}: {summ.shape[0]} summary TRs < {n} 1TR chunks")
            blocks.append(summ[:n])
        combo[story] = np.concatenate(blocks, axis=1).astype(np.float32)
    return combo


def main() -> None:
    args = parse_args()
    args.lags = sorted({int(l) for l in args.lags})
    args.summary_horizons = sorted({int(h) for h in args.summary_horizons})
    if args.chunk_trs != 1:
        raise ValueError("This quick combo experiment currently expects --chunk-trs 1.")

    mounted_root = configure_data_root(args)
    stories = load_stories(args)
    train_stories, val_stories = split_stories(stories, args)
    log.info("Stories: %d total | %d train | %d val", len(stories), len(train_stories), len(val_stories))
    log.info("Validation stories: %s", ", ".join(val_stories))

    response_root = config.DATA_TRAIN_DIR
    if args.local_compute_mode and mounted_root is not None:
        response_root = str(
            rse.stage_local_response_cache(
                args.subject,
                stories,
                Path(config.DATA_TRAIN_DIR),
                Path(args.local_cache_root).expanduser().resolve(),
            )
        )
        log.info("Using staged local response root: %s", response_root)

    sample_resp = get_resp(args.subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    total_voxels = int(sample_resp.shape[1])
    voxels = load_full_frontal_voxels(args.subject, total_voxels, args.ba_dir)
    log.info("Full-frontal voxels: %d / %d", len(voxels), total_voxels)

    responses_by_story = get_resp(args.subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(r.shape[0]) for s, r in responses_by_story.items()}

    max_lag = max(args.lags)
    one_tr_args = argparse.Namespace(
        subject=args.subject,
        embedding_cache_dir=args.one_tr_cache_dir,
        feature_model="embedding",
        chunk_trs=1,
        lag_trs=int(max_lag),
        embed_batch_size=int(args.embed_batch_size),
    )
    one_tr, one_dim, one_cache = load_or_build_chunk_embeddings(
        one_tr_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )
    summary_embs, summary_model, summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
    combo = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)
    combo_dim = int(next(iter(combo.values())).shape[1])
    log.info(
        "Feature dim: 1TR=%d, summaries=%d x %d, combo=%d",
        one_dim, len(args.summary_horizons), one_dim, combo_dim,
    )

    out_dir = Path(args.output_dir).expanduser().resolve() / make_tag(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_lag_dir = out_dir / "per_lag"
    per_lag_dir.mkdir(parents=True, exist_ok=True)

    x_train, _ = stack_lag(combo, responses_by_story, train_stories, args.lags[0])
    x_val, _ = stack_lag(combo, responses_by_story, val_stories, args.lags[0])
    log.info("Inputs: X_train=%s X_val=%s", x_train.shape, x_val.shape)

    n_lags = len(args.lags)
    n_voxels = int(voxels.size)
    corrs_by_lag = np.zeros((n_lags, n_voxels), dtype=np.float32)
    best_alphas_by_lag = np.zeros((n_lags, n_voxels), dtype=np.float32)

    for li, lag in enumerate(args.lags):
        log.info("==== lag=%d ====", lag)
        t0 = time.time()
        _, y_train = stack_lag(combo, responses_by_story, train_stories, lag)
        _, y_val = stack_lag(combo, responses_by_story, val_stories, lag)
        pred_val, best_alphas = fit_ridge_predict(
            x_train,
            y_train,
            x_val,
            alphas=args.ridge_alphas,
            voxel_chunk_size=args.voxel_chunk_size,
        )
        corrs = per_voxel_corr(pred_val, y_val)
        elapsed = time.time() - t0
        log.info(
            "lag=%d mean_r=%.4f median_r=%.4f p95_r=%.4f max_r=%.4f n(r>0.1)=%d (%.1fs)",
            lag,
            float(corrs.mean()),
            float(np.median(corrs)),
            float(np.quantile(corrs, 0.95)),
            float(corrs.max()),
            int((corrs > 0.10).sum()),
            elapsed,
        )
        np.savez(
            per_lag_dir / f"lag{lag:02d}.npz",
            corrs=corrs,
            best_alphas=best_alphas,
            voxels=voxels,
            lag=int(lag),
            elapsed_sec=float(elapsed),
            train_stories=np.array(train_stories),
            val_stories=np.array(val_stories),
            feature_model="embedding-summary-combo",
            summary_model=summary_model,
            summary_horizons=np.array(args.summary_horizons, dtype=int),
            chunk_trs=int(args.chunk_trs),
        )
        corrs_by_lag[li] = corrs
        best_alphas_by_lag[li] = best_alphas

    np.savez(
        out_dir / "lag_corrs.npz",
        corrs=corrs_by_lag,
        best_alphas=best_alphas_by_lag,
        voxels=voxels,
        lags=np.array(args.lags, dtype=int),
        feature_model="embedding-summary-combo",
        embedding_model=args.embedding_model,
        summary_model=summary_model,
        summary_horizons=np.array(args.summary_horizons, dtype=int),
        chunk_trs=int(args.chunk_trs),
        train_stories=np.array(train_stories),
        val_stories=np.array(val_stories),
        ridge_alphas=np.array(args.ridge_alphas, dtype=float),
        subject=args.subject,
        combo_dim=int(combo_dim),
    )
    write_summary_csv(out_dir / "lag_summary.csv", args.lags, corrs_by_lag)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject": args.subject,
                "feature_model": "embedding-summary-combo",
                "embedding_model": args.embedding_model,
                "summary_model": summary_model,
                "summary_horizons": args.summary_horizons,
                "combo_dim": combo_dim,
                "lags": args.lags,
                "train_stories": train_stories,
                "val_stories": val_stories,
                "n_voxels": n_voxels,
                "one_tr_cache": one_cache,
                "summary_cache": summary_cache,
                "data_train_dir": config.DATA_TRAIN_DIR,
                "response_root": response_root,
                "ba_dir": str(Path(args.ba_dir).expanduser().resolve()),
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
