#!/usr/bin/env python3
"""Retrain combo ridge models with one feature block removed."""
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
    per_voxel_corr,
    split_stories,
    stack_lag,
)
from train_summary_combo_encoding import build_combo_embeddings, load_or_build_summary_embeddings  # noqa: E402
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("combo_block_ablation")

SUB_ROIS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="Combo run directory containing lag_corrs.npz/config.json.")
    p.add_argument("--subject", default=None, choices=sorted(rse.SUBJECT_TO_UTS))
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--top-n", type=int, default=0, help="Use top N voxels by saved full-model r. Default 0 uses all.")
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=None)
    p.add_argument("--voxel-chunk-size", type=int, default=1000)

    p.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--summary-horizons", nargs="+", type=int, default=None)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))
    p.add_argument("--chunk-trs", type=int, default=None)
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def zscore_train_apply(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return ((x_train - mean) / std).astype(np.float32), ((x_val - mean) / std).astype(np.float32)


def fit_ridge_corrs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float],
    voxel_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import RidgeCV

    x_train_z, x_val_z = zscore_train_apply(x_train, x_val)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0
    y_train_z = ((y_train - y_mean) / y_std).astype(np.float32)

    n_voxels = y_train.shape[1]
    corrs = np.zeros(n_voxels, dtype=np.float32)
    best_alphas = np.zeros(n_voxels, dtype=np.float32)
    chunk = max(1, int(voxel_chunk_size))
    for start in range(0, n_voxels, chunk):
        end = min(start + chunk, n_voxels)
        model = RidgeCV(alphas=list(alphas), alpha_per_target=True)
        model.fit(x_train_z, y_train_z[:, start:end])
        pred = model.predict(x_val_z).astype(np.float32) * y_std[start:end] + y_mean[start:end]
        corrs[start:end] = per_voxel_corr(pred, y_val[:, start:end])
        alpha = np.asarray(model.alpha_, dtype=np.float32)
        if alpha.ndim == 0:
            alpha = np.full(end - start, float(alpha), dtype=np.float32)
        best_alphas[start:end] = alpha
    return corrs, best_alphas


def load_roi_masks(ba_dir: Path, subject: str, voxels: np.ndarray) -> Dict[str, np.ndarray]:
    uts_id = rse.SUBJECT_TO_UTS[subject]
    masks: Dict[str, np.ndarray] = {"full_frontal": np.ones_like(voxels, dtype=bool)}
    for name in SUB_ROIS:
        path = ba_dir / uts_id / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            ids = np.asarray(next(iter(json.load(f).values())), dtype=int)
        masks[name] = np.isin(voxels, ids)
    return masks


def select_voxels(saved_corrs: np.ndarray, saved_lags: Sequence[int], lag: int, top_n: int) -> np.ndarray:
    lag_idx = list(saved_lags).index(int(lag))
    lag_r = saved_corrs[lag_idx]
    order = np.argsort(-lag_r)
    if top_n > 0:
        order = order[: min(int(top_n), order.size)]
    return np.sort(order)


def summarize_rows(voxel_rows: list[dict], masks: Dict[str, np.ndarray], blocks: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for roi, mask in masks.items():
        selected_for_roi = np.nonzero(mask)[0].tolist()
        if not selected_for_roi:
            continue
        for block in blocks:
            vals = np.asarray([float(voxel_rows[i][f"delta_drop_{block}"]) for i in selected_for_roi], dtype=float)
            rows.append(
                {
                    "subject": voxel_rows[0]["subject"],
                    "lag": voxel_rows[0]["lag"],
                    "roi": roi,
                    "block": block,
                    "n_voxels": len(selected_for_roi),
                    "mean_delta_r": float(vals.mean()),
                    "median_delta_r": float(np.median(vals)),
                    "fraction_positive_delta": float((vals > 0).mean()),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    block_names = ["1TR"] + [f"h{h}" for h in args.summary_horizons]
    saved_lags = [int(x) for x in saved["lags"]]
    args.lags = saved_lags
    args.chunk_trs = int(args.chunk_trs or run_cfg.get("chunk_trs", saved.get("chunk_trs", 1)))
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

    sample_resp = get_resp(subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    voxels = load_full_frontal_voxels(subject, int(sample_resp.shape[1]), args.ba_dir)
    if not np.array_equal(voxels, saved["voxels"]):
        raise ValueError("Voxel list from ROI does not match saved lag_corrs.npz voxels.")
    selected_local = select_voxels(saved["corrs"], saved_lags, args.lag, args.top_n)
    selected_voxels = voxels[selected_local]
    log.info("%s: selected %d voxels for ablation", subject, len(selected_local))

    log.info("%s: loading responses for selected voxels", subject)
    responses_by_story = get_resp(subject, stories, stack=False, vox=selected_voxels, response_root=response_root)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(arr.shape[0]) for s, arr in responses_by_story.items()}

    one_tr_args = argparse.Namespace(
        subject=subject,
        embedding_cache_dir=args.one_tr_cache_dir,
        feature_model="embedding",
        embedding_model=args.embedding_model,
        chunk_trs=1,
        lag_trs=int(max(saved_lags)),
        embed_batch_size=int(args.embed_batch_size),
    )
    log.info("%s: loading 1TR embeddings", subject)
    one_tr, one_dim, _cache = load_or_build_chunk_embeddings(
        one_tr_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )
    log.info("%s: loading summary embeddings", subject)
    summary_embs, _summary_model, _summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
    log.info("%s: building combo feature matrices", subject)
    combo = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)

    block_slices = {name: slice(i * one_dim, (i + 1) * one_dim) for i, name in enumerate(block_names)}
    variants = {"full": np.arange(one_dim * len(block_names))}
    for block in block_names:
        keep = [np.arange(sl.start, sl.stop) for name, sl in block_slices.items() if name != block]
        variants[f"drop_{block}"] = np.concatenate(keep)

    log.info("%s: stacking lagged train/val matrices", subject)
    x_train, y_train = stack_lag(combo, responses_by_story, train_stories, args.lag)
    x_val, y_val = stack_lag(combo, responses_by_story, val_stories, args.lag)
    log.info("%s: X_train=%s X_val=%s Y_train=%s", subject, x_train.shape, x_val.shape, y_train.shape)

    corrs_by_variant: dict[str, np.ndarray] = {}
    for variant, cols in variants.items():
        log.info("%s: fitting %s with %d features", subject, variant, len(cols))
        corrs, _best_alphas = fit_ridge_corrs(
            x_train[:, cols],
            y_train,
            x_val[:, cols],
            y_val,
            alphas,
            args.voxel_chunk_size,
        )
        corrs_by_variant[variant] = corrs

    voxel_rows: list[dict] = []
    full = corrs_by_variant["full"]
    saved_lag_r = saved["corrs"][saved_lags.index(args.lag), selected_local]
    for i, local_idx in enumerate(selected_local):
        row = {
            "subject": subject,
            "lag": int(args.lag),
            "local_voxel_index": int(local_idx),
            "global_voxel_index": int(selected_voxels[i]),
            "saved_full_r": float(saved_lag_r[i]),
            "refit_full_r": float(full[i]),
        }
        for block in block_names:
            drop_r = float(corrs_by_variant[f"drop_{block}"][i])
            row[f"drop_{block}_r"] = drop_r
            row[f"delta_drop_{block}"] = float(full[i] - drop_r)
        voxel_rows.append(row)

    masks = load_roi_masks(Path(args.ba_dir).expanduser().resolve(), subject, selected_voxels)
    summary_rows = summarize_rows(voxel_rows, masks, block_names)

    out_prefix = Path(args.out_prefix).expanduser().resolve() if args.out_prefix else results_dir / f"combo_block_ablation_lag{args.lag}_top{len(selected_local)}"
    write_csv(out_prefix.with_name(out_prefix.name + "_voxels.csv"), voxel_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary_rows)
    log.info("Wrote %s", out_prefix.with_name(out_prefix.name + "_summary.csv"))


if __name__ == "__main__":
    main()
