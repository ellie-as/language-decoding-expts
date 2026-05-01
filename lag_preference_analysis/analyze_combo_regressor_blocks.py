#!/usr/bin/env python3
"""Refit combo ridge encoders and summarize coefficient size by feature block.

The combo model has four equal-size MiniLM blocks:

    1TR text | summary h20 | summary h50 | summary h200

The original training output saves correlations and selected alphas, but not
the coefficient matrix. This script refits the same standardized RidgeCV model
for selected lags and writes ROI-level summaries of per-voxel coefficient block
norms. Because X and Y are z-scored before fitting, block norms are comparable
across blocks and voxels. All blocks are 384 dimensions, so the L2 norm is also
directly comparable across blocks.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Sequence

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
    load_full_frontal_voxels,
    load_stories,
    split_stories,
    stack_lag,
)
from train_summary_combo_encoding import (  # noqa: E402
    build_combo_embeddings,
    load_or_build_summary_embeddings,
)
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("combo_block_weights")

SUBJECT_TO_UTS = rse.SUBJECT_TO_UTS
SUB_ROIS = ["BA_10", "BA_6", "BA_8", "BA_9_46", "BROCA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="Combo run directory containing lag_corrs.npz/config.json.")
    p.add_argument("--subject", default=None, choices=sorted(SUBJECT_TO_UTS))
    p.add_argument("--lags", nargs="+", type=int, default=None, help="Lags to refit (default: all saved lags).")
    p.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--summary-horizons", nargs="+", type=int, default=None)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=None)
    p.add_argument("--voxel-chunk-size", type=int, default=5_000)
    p.add_argument("--reliable-threshold", type=float, default=0.05)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def zscore_train_apply(x_train: np.ndarray, x_val: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train_z = ((x_train - mean) / std).astype(np.float32)
    if x_val is None:
        return x_train_z, None
    return x_train_z, ((x_val - mean) / std).astype(np.float32)


def load_roi_masks(ba_dir: Path, subject: str, voxels: np.ndarray) -> Dict[str, np.ndarray]:
    uts_id = SUBJECT_TO_UTS[subject]
    masks: Dict[str, np.ndarray] = {"full_frontal": np.ones_like(voxels, dtype=bool)}
    for name in SUB_ROIS:
        path = ba_dir / uts_id / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            ids = np.asarray(next(iter(json.load(f).values())), dtype=int)
        masks[name] = np.isin(voxels, ids)
    return masks


def fit_block_norms(
    x_train: np.ndarray,
    y_train: np.ndarray,
    block_slices: Dict[str, slice],
    alphas: Sequence[float],
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    from sklearn.linear_model import RidgeCV

    x_train_z, _ = zscore_train_apply(x_train)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0
    y_train_z = ((y_train - y_mean) / y_std).astype(np.float32)

    n_voxels = y_train_z.shape[1]
    block_norms = {block: np.zeros(n_voxels, dtype=np.float32) for block in block_slices}
    best_alphas = np.zeros(n_voxels, dtype=np.float32)
    chunk_size = max(1, int(chunk_size))

    for ci, start in enumerate(range(0, n_voxels, chunk_size), start=1):
        end = min(start + chunk_size, n_voxels)
        model = RidgeCV(alphas=list(alphas), alpha_per_target=True)
        model.fit(x_train_z, y_train_z[:, start:end])
        coef = np.asarray(model.coef_, dtype=np.float32)
        if coef.ndim == 1:
            coef = coef[None, :]
        for block, sl in block_slices.items():
            block_norms[block][start:end] = np.linalg.norm(coef[:, sl], axis=1)
        alpha = np.asarray(model.alpha_, dtype=np.float32)
        if alpha.ndim == 0:
            alpha = np.full(end - start, float(alpha), dtype=np.float32)
        best_alphas[start:end] = alpha
        log.info("  coefficient chunk %d (%d:%d) done", ci, start, end)

    block_norms["selected_alpha"] = best_alphas
    return block_norms


def summarize_rows(
    *,
    subject: str,
    lag: int,
    block_norms: Dict[str, np.ndarray],
    block_names: Sequence[str],
    masks: Dict[str, np.ndarray],
    reliable: np.ndarray,
) -> list[dict]:
    total = np.zeros_like(next(iter(block_norms.values())), dtype=np.float32)
    for block in block_names:
        total += block_norms[block]
    total[total == 0] = np.nan

    rows: list[dict] = []
    for subset_name, subset_mask in {
        "all": np.ones_like(reliable, dtype=bool),
        "reliable": reliable,
    }.items():
        for roi, roi_mask in masks.items():
            mask = roi_mask & subset_mask
            if not np.any(mask):
                continue
            for block in block_names:
                vals = block_norms[block][mask]
                frac = block_norms[block][mask] / total[mask]
                rows.append(
                    {
                        "subject": subject,
                        "lag": int(lag),
                        "subset": subset_name,
                        "roi": roi,
                        "block": block,
                        "n_voxels": int(mask.sum()),
                        "mean_l2": float(np.nanmean(vals)),
                        "median_l2": float(np.nanmedian(vals)),
                        "mean_fraction": float(np.nanmean(frac)),
                        "median_fraction": float(np.nanmedian(frac)),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    saved = np.load(results_dir / "lag_corrs.npz", allow_pickle=True)
    config_path = results_dir / "config.json"
    run_cfg = json.load(open(config_path, encoding="utf-8")) if config_path.is_file() else {}

    subject = args.subject or str(saved["subject"])
    args.subject = subject
    args.summary_horizons = args.summary_horizons or [int(x) for x in saved["summary_horizons"]]
    args.summary_model = args.summary_model or run_cfg.get("summary_model")
    args.embedding_model = run_cfg.get("embedding_model", args.embedding_model)
    lags = args.lags or [int(x) for x in saved["lags"]]
    alphas = args.ridge_alphas or [float(x) for x in saved["ridge_alphas"]]

    mounted_root = configure_data_root(args)
    stories = load_stories(args)
    if "train_stories" in saved.files and "val_stories" in saved.files:
        train_stories = [str(x) for x in saved["train_stories"]]
        val_stories = [str(x) for x in saved["val_stories"]]
    else:
        train_stories, val_stories = split_stories(stories, args)
    log.info("%s: %d train stories, %d val stories", subject, len(train_stories), len(val_stories))

    response_root = config.DATA_TRAIN_DIR
    if args.local_compute_mode and mounted_root is not None:
        response_root = str(
            rse.stage_local_response_cache(
                subject,
                stories,
                Path(config.DATA_TRAIN_DIR),
                Path(args.local_cache_root).expanduser().resolve(),
            )
        )
        log.info("Using staged local response root: %s", response_root)

    sample_resp = get_resp(subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    voxels = load_full_frontal_voxels(subject, int(sample_resp.shape[1]), args.ba_dir)
    if not np.array_equal(voxels, saved["voxels"]):
        raise ValueError("Voxel list from ROI does not match saved lag_corrs.npz voxels.")
    masks = load_roi_masks(Path(args.ba_dir).expanduser().resolve(), subject, voxels)

    responses_by_story = get_resp(subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(arr.shape[0]) for s, arr in responses_by_story.items()}

    max_lag = max(lags)
    one_tr_args = argparse.Namespace(
        subject=subject,
        embedding_cache_dir=args.one_tr_cache_dir,
        feature_model="embedding",
        chunk_trs=1,
        lag_trs=int(max_lag),
        embed_batch_size=int(args.embed_batch_size),
    )
    one_tr, one_dim, _cache = load_or_build_chunk_embeddings(
        one_tr_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )
    summary_embs, _summary_model, _summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
    combo = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)

    block_names = ["1TR"] + [f"h{h}" for h in args.summary_horizons]
    block_slices = {name: slice(i * one_dim, (i + 1) * one_dim) for i, name in enumerate(block_names)}

    saved_lags = [int(x) for x in saved["lags"]]
    rows: list[dict] = []
    for lag in lags:
        log.info("==== %s lag=%d ====", subject, lag)
        x_train, y_train = stack_lag(combo, responses_by_story, train_stories, int(lag))
        norms = fit_block_norms(x_train, y_train, block_slices, alphas, args.voxel_chunk_size)
        li = saved_lags.index(int(lag))
        best_r = saved["corrs"][np.argmax(saved["corrs"], axis=0), np.arange(saved["corrs"].shape[1])]
        reliable = best_r >= float(args.reliable_threshold)
        rows.extend(
            summarize_rows(
                subject=subject,
                lag=int(lag),
                block_norms=norms,
                block_names=block_names,
                masks=masks,
                reliable=reliable,
            )
        )

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else results_dir / "combo_regressor_block_norms.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "lag",
        "subset",
        "roi",
        "block",
        "n_voxels",
        "mean_l2",
        "median_l2",
        "mean_fraction",
        "median_fraction",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
