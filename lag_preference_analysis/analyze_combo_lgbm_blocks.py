#!/usr/bin/env python3
"""Voxel-sampled LightGBM block importance for combo encoding features.

This is the non-linear analogue of ``analyze_combo_regressor_blocks.py``.
Because LightGBM has no linear coefficients, this script reports block
importance using summed feature gain within each MiniLM block:

    1TR text | summary h20 | summary h50 | summary h200

For each ROI, it samples the top-N reliable voxels by the saved combo model's
best-lag r, fits one LGBMRegressor per sampled voxel at the requested lag, and
averages block gain fractions across voxels.

The output is two CSVs:

    combo_lgbm_block_importance_lag2_summary.csv
    combo_lgbm_block_importance_lag2_voxels.csv

This is designed for quick anatomical comparison, not as the fastest possible
full-brain LGBM encoder.
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
log = logging.getLogger("combo_lgbm_blocks")

SUBJECT_TO_UTS = rse.SUBJECT_TO_UTS
SUB_ROIS = ["BA_10", "BA_6", "BA_8", "BA_9_46", "BROCA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="Combo run directory containing lag_corrs.npz/config.json.")
    p.add_argument("--subject", default=None, choices=sorted(SUBJECT_TO_UTS))
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--top-n-per-roi", type=int, default=200)
    p.add_argument("--reliable-threshold", type=float, default=0.05)
    p.add_argument("--include-full-frontal", action="store_true",
                   help="Also sample top voxels across the whole full_frontal mask.")

    p.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--summary-horizons", nargs="+", type=int, default=None)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))
    p.add_argument("--chunk-trs", type=int, default=None)
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))

    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--out-prefix", default=None,
                   help="Output prefix. Defaults to <results-dir>/combo_lgbm_block_importance_lag<L>.")
    return p.parse_args()


def load_roi_masks(ba_dir: Path, subject: str, voxels: np.ndarray) -> Dict[str, np.ndarray]:
    uts_id = SUBJECT_TO_UTS[subject]
    masks: Dict[str, np.ndarray] = {}
    for name in SUB_ROIS:
        path = ba_dir / uts_id / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            ids = np.asarray(next(iter(json.load(f).values())), dtype=int)
        masks[name] = np.isin(voxels, ids)
    return masks


def per_voxel_corr(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pred = pred - pred.mean()
    true = true - true.mean()
    denom = np.sqrt(np.sum(pred * pred) * np.sum(true * true))
    return float(np.sum(pred * true) / denom) if denom > 0 else 0.0


def select_voxels_for_roi(
    roi_mask: np.ndarray,
    best_r: np.ndarray,
    top_n: int,
    reliable_threshold: float,
) -> np.ndarray:
    candidates = np.where(roi_mask & (best_r >= reliable_threshold))[0]
    if candidates.size == 0:
        candidates = np.where(roi_mask)[0]
    order = np.argsort(-best_r[candidates])
    return candidates[order[: min(int(top_n), candidates.size)]]


def block_gain(importances: np.ndarray, block_slices: Dict[str, slice]) -> Dict[str, float]:
    gains = {name: float(np.sum(importances[sl])) for name, sl in block_slices.items()}
    total = sum(gains.values())
    if total <= 0:
        return {name: 0.0 for name in block_slices}
    return {name: val / total for name, val in gains.items()}


def summarize_rows(voxel_rows: list[dict], block_names: Sequence[str]) -> list[dict]:
    out: list[dict] = []
    keys = sorted({(r["subject"], r["lag"], r["roi"]) for r in voxel_rows})
    for subject, lag, roi in keys:
        rows = [r for r in voxel_rows if r["subject"] == subject and r["lag"] == lag and r["roi"] == roi]
        for block in block_names:
            vals = np.asarray([float(r[f"{block}_gain_fraction"]) for r in rows], dtype=float)
            out.append(
                {
                    "subject": subject,
                    "lag": lag,
                    "roi": roi,
                    "block": block,
                    "n_voxels": len(rows),
                    "mean_gain_fraction": float(vals.mean()),
                    "median_gain_fraction": float(np.median(vals)),
                    "mean_lgbm_val_r": float(np.mean([float(r["lgbm_val_r"]) for r in rows])),
                    "mean_saved_combo_best_r": float(np.mean([float(r["saved_combo_best_r"]) for r in rows])),
                }
            )
    return out


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
    saved_lags = [int(x) for x in saved["lags"]]
    args.lags = saved_lags
    args.chunk_trs = int(args.chunk_trs or run_cfg.get("chunk_trs", saved.get("chunk_trs", 1)))
    if args.lag not in saved_lags:
        raise ValueError(f"--lag {args.lag} not in saved lags {saved_lags}")

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
    if args.include_full_frontal:
        masks["full_frontal"] = np.ones_like(voxels, dtype=bool)

    responses_by_story = get_resp(subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(arr.shape[0]) for s, arr in responses_by_story.items()}

    max_lag = max(saved_lags)
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
    x_train, y_train = stack_lag(combo, responses_by_story, train_stories, args.lag)
    x_val, y_val = stack_lag(combo, responses_by_story, val_stories, args.lag)
    log.info("Lag %d data: X_train=%s X_val=%s Y_train=%s", args.lag, x_train.shape, x_val.shape, y_train.shape)

    block_names = ["1TR"] + [f"h{h}" for h in args.summary_horizons]
    block_slices = {name: slice(i * one_dim, (i + 1) * one_dim) for i, name in enumerate(block_names)}

    saved_corrs = saved["corrs"]
    best_r = saved_corrs[np.argmax(saved_corrs, axis=0), np.arange(saved_corrs.shape[1])]
    lag_idx = saved_lags.index(args.lag)
    lag_r = saved_corrs[lag_idx]

    import lightgbm as lgb

    params = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "num_leaves": int(args.num_leaves),
        "min_child_samples": int(args.min_child_samples),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_alpha": float(args.reg_alpha),
        "reg_lambda": float(args.reg_lambda),
        "n_jobs": int(args.n_jobs),
        "random_state": int(args.random_state),
        "verbosity": -1,
        "force_col_wise": True,
        "importance_type": "gain",
    }
    log.info("LightGBM params: %s", params)

    voxel_rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for roi, roi_mask in masks.items():
        selected = select_voxels_for_roi(roi_mask, best_r, args.top_n_per_roi, args.reliable_threshold)
        log.info("ROI %s: fitting %d voxels", roi, len(selected))
        for rank, local_vox in enumerate(selected, start=1):
            key = (roi, int(local_vox))
            if key in seen:
                continue
            seen.add(key)
            y = y_train[:, local_vox]
            y_mean = float(y.mean())
            y_std = float(y.std()) or 1.0
            y_z = ((y - y_mean) / y_std).astype(np.float32)
            model = lgb.LGBMRegressor(**params)
            model.fit(x_train, y_z)
            pred = model.predict(x_val)
            val_r = per_voxel_corr(pred, (y_val[:, local_vox] - y_mean) / y_std)
            frac = block_gain(model.feature_importances_.astype(float), block_slices)
            row = {
                "subject": subject,
                "lag": int(args.lag),
                "roi": roi,
                "rank_in_roi": int(rank),
                "local_voxel_index": int(local_vox),
                "global_voxel_index": int(voxels[local_vox]),
                "saved_combo_best_r": float(best_r[local_vox]),
                "saved_combo_lag_r": float(lag_r[local_vox]),
                "lgbm_val_r": float(val_r),
            }
            for block in block_names:
                row[f"{block}_gain_fraction"] = float(frac[block])
            voxel_rows.append(row)
            if rank % 25 == 0:
                log.info("  %s %d/%d", roi, rank, len(selected))

    out_prefix = Path(args.out_prefix).expanduser().resolve() if args.out_prefix else results_dir / f"combo_lgbm_block_importance_lag{args.lag}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    vox_csv = out_prefix.with_name(out_prefix.name + "_voxels.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")

    voxel_fieldnames = [
        "subject",
        "lag",
        "roi",
        "rank_in_roi",
        "local_voxel_index",
        "global_voxel_index",
        "saved_combo_best_r",
        "saved_combo_lag_r",
        "lgbm_val_r",
    ] + [f"{b}_gain_fraction" for b in block_names]
    with open(vox_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=voxel_fieldnames)
        writer.writeheader()
        writer.writerows(voxel_rows)

    summary_rows = summarize_rows(voxel_rows, block_names)
    summary_fieldnames = [
        "subject",
        "lag",
        "roi",
        "block",
        "n_voxels",
        "mean_gain_fraction",
        "median_gain_fraction",
        "mean_lgbm_val_r",
        "mean_saved_combo_best_r",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    log.info("Wrote %s", summary_csv)
    log.info("Wrote %s", vox_csv)


if __name__ == "__main__":
    main()
