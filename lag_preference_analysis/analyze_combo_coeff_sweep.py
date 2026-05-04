#!/usr/bin/env python3
"""Compare coefficient block sizes and single-block performance.

Fits models on:

    1TR | h20 | h50 | h200 | h500

for each requested SentenceTransformer embedding model, then summarizes the
L2 coefficient mass assigned to each block within each ROI. It also fits
single-block models, so standalone feature-block prediction performance can be
compared with combined-model coefficient mass.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
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
from analyze_combo_block_ablation import load_roi_masks, write_csv  # noqa: E402
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
log = logging.getLogger("combo_coeff_sweep")

DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
]
REGRESSORS = ["ridge", "linear", "elasticnet"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, choices=sorted(rse.SUBJECT_TO_UTS))
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--embedding-models", nargs="+", default=DEFAULT_EMBEDDING_MODELS)
    p.add_argument("--summary-horizons", nargs="+", type=int, default=[20, 50, 200, 500])
    p.add_argument("--regressors", nargs="+", choices=REGRESSORS, default=REGRESSORS)
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0])
    p.add_argument("--elasticnet-alpha", type=float, default=0.001)
    p.add_argument("--elasticnet-l1-ratio", type=float, default=0.1)
    p.add_argument("--elasticnet-max-iter", type=int, default=2000)
    p.add_argument("--elasticnet-tol", type=float, default=1e-4)
    p.add_argument("--voxel-chunk-size", type=int, default=1000)
    p.add_argument(
        "--block-pca-dim",
        type=int,
        default=0,
        help="If >0, fit PCA separately within each feature block on training rows and use this many PCs per block.",
    )
    p.add_argument("--skip-single-blocks", action="store_true", help="Only write combined-model coefficient tables.")

    p.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-device", default="auto")
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def resolve_embedding_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def model_tag(model_name: str) -> str:
    tag = model_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tag).strip("-")


def zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def zscore_train_apply(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return ((x_train - mean) / std).astype(np.float32), ((x_val - mean) / std).astype(np.float32)


def apply_block_pca(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    block_slices: Dict[str, slice],
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, slice], dict[str, float]]:
    """Fit PCA separately per block on training rows, then transform validation rows."""
    from sklearn.decomposition import PCA

    if n_components <= 0:
        return x_train, x_val, block_slices, {}

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    pca_slices: Dict[str, slice] = {}
    explained: dict[str, float] = {}
    start = 0

    for block, sl in block_slices.items():
        block_train, block_val = zscore_train_apply(x_train[:, sl], x_val[:, sl])
        n_comp = min(int(n_components), block_train.shape[0] - 1, block_train.shape[1])
        if n_comp <= 0:
            raise ValueError(f"Cannot fit PCA for block {block!r}: requested {n_components}, train shape={block_train.shape}")
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=int(seed))
        train_pcs = pca.fit_transform(block_train).astype(np.float32)
        val_pcs = pca.transform(block_val).astype(np.float32)
        train_parts.append(train_pcs)
        val_parts.append(val_pcs)
        pca_slices[block] = slice(start, start + n_comp)
        explained[block] = float(np.sum(pca.explained_variance_ratio_))
        start += n_comp

    return np.hstack(train_parts).astype(np.float32), np.hstack(val_parts).astype(np.float32), pca_slices, explained


def fit_coefficients(
    *,
    regressor: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return coefficients, selected alphas, and optional validation correlations."""
    if x_val is None:
        x_train_z, _x_mean, _x_std = zscore(x_train)
        x_val_z = None
    else:
        x_train_z, x_val_z = zscore_train_apply(x_train, x_val)
    y_train_z, _y_mean, _y_std = zscore(y_train)
    n_voxels = y_train_z.shape[1]
    n_features = x_train_z.shape[1]
    coef = np.zeros((n_voxels, n_features), dtype=np.float32)
    alphas = np.full(n_voxels, np.nan, dtype=np.float32)
    corrs = np.zeros(n_voxels, dtype=np.float32) if x_val_z is not None and y_val is not None else None
    chunk = max(1, int(args.voxel_chunk_size))

    if regressor == "ridge":
        from sklearn.linear_model import RidgeCV

        for start in range(0, n_voxels, chunk):
            end = min(start + chunk, n_voxels)
            model = RidgeCV(alphas=list(args.ridge_alphas), alpha_per_target=True, fit_intercept=False)
            model.fit(x_train_z, y_train_z[:, start:end])
            block_coef = np.asarray(model.coef_, dtype=np.float32)
            if block_coef.ndim == 1:
                block_coef = block_coef[None, :]
            coef[start:end] = block_coef
            alpha = np.asarray(model.alpha_, dtype=np.float32)
            if alpha.ndim == 0:
                alpha = np.full(end - start, float(alpha), dtype=np.float32)
            alphas[start:end] = alpha
            if corrs is not None and x_val_z is not None and y_val is not None:
                pred = model.predict(x_val_z).astype(np.float32) * _y_std[start:end] + _y_mean[start:end]
                corrs[start:end] = per_voxel_corr(pred, y_val[:, start:end])
            log.info("  ridge chunk %d:%d", start, end)
        return coef, alphas, corrs

    if regressor == "linear":
        from sklearn.linear_model import LinearRegression

        for start in range(0, n_voxels, chunk):
            end = min(start + chunk, n_voxels)
            model = LinearRegression(fit_intercept=False)
            model.fit(x_train_z, y_train_z[:, start:end])
            block_coef = np.asarray(model.coef_, dtype=np.float32)
            if block_coef.ndim == 1:
                block_coef = block_coef[None, :]
            coef[start:end] = block_coef
            if corrs is not None and x_val_z is not None and y_val is not None:
                pred = model.predict(x_val_z).astype(np.float32) * _y_std[start:end] + _y_mean[start:end]
                corrs[start:end] = per_voxel_corr(pred, y_val[:, start:end])
            log.info("  linear chunk %d:%d", start, end)
        return coef, alphas, corrs

    if regressor == "elasticnet":
        from sklearn.linear_model import ElasticNet

        for start in range(0, n_voxels, chunk):
            end = min(start + chunk, n_voxels)
            model = ElasticNet(
                alpha=float(args.elasticnet_alpha),
                l1_ratio=float(args.elasticnet_l1_ratio),
                fit_intercept=False,
                max_iter=int(args.elasticnet_max_iter),
                tol=float(args.elasticnet_tol),
                selection="cyclic",
            )
            model.fit(x_train_z, y_train_z[:, start:end])
            block_coef = np.asarray(model.coef_, dtype=np.float32)
            if block_coef.ndim == 1:
                block_coef = block_coef[None, :]
            coef[start:end] = block_coef
            if corrs is not None and x_val_z is not None and y_val is not None:
                pred = model.predict(x_val_z).astype(np.float32) * _y_std[start:end] + _y_mean[start:end]
                corrs[start:end] = per_voxel_corr(pred, y_val[:, start:end])
            log.info("  elasticnet chunk %d:%d", start, end)
        return coef, alphas, corrs

    raise ValueError(regressor)


def summarize_coefficients(
    *,
    coef: np.ndarray,
    alphas: np.ndarray,
    block_slices: Dict[str, slice],
    masks: Dict[str, np.ndarray],
    subject: str,
    lag: int,
    embedding_model: str,
    regressor: str,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
) -> list[dict]:
    block_names = list(block_slices)
    norms = {block: np.linalg.norm(coef[:, sl], axis=1).astype(np.float32) for block, sl in block_slices.items()}
    total = np.zeros(coef.shape[0], dtype=np.float32)
    for block in block_names:
        total += norms[block]
    total[total == 0] = np.nan

    rows: list[dict] = []
    for roi, mask in masks.items():
        if not np.any(mask):
            continue
        for block in block_names:
            vals = norms[block][mask]
            frac = vals / total[mask]
            row = {
                "subject": subject,
                "lag": int(lag),
                "embedding_model": embedding_model,
                "embedding_tag": model_tag(embedding_model),
                "regressor": regressor,
                "roi": roi,
                "block": block,
                "n_voxels": int(mask.sum()),
                "mean_l2": float(np.nanmean(vals)),
                "median_l2": float(np.nanmedian(vals)),
                "mean_fraction": float(np.nanmean(frac)),
                "median_fraction": float(np.nanmedian(frac)),
            }
            if regressor == "ridge":
                row["mean_selected_alpha"] = float(np.nanmean(alphas[mask]))
                row["median_selected_alpha"] = float(np.nanmedian(alphas[mask]))
            else:
                row["mean_selected_alpha"] = np.nan
                row["median_selected_alpha"] = np.nan
            row["elasticnet_alpha"] = float(elasticnet_alpha) if regressor == "elasticnet" else np.nan
            row["elasticnet_l1_ratio"] = float(elasticnet_l1_ratio) if regressor == "elasticnet" else np.nan
            rows.append(row)
    return rows


def summarize_model_r(
    *,
    corrs: np.ndarray,
    masks: Dict[str, np.ndarray],
    subject: str,
    lag: int,
    embedding_model: str,
    regressor: str,
    model_variant: str,
    block: str,
) -> list[dict]:
    rows: list[dict] = []
    for roi, mask in masks.items():
        if not np.any(mask):
            continue
        vals = corrs[mask]
        rows.append(
            {
                "subject": subject,
                "lag": int(lag),
                "embedding_model": embedding_model,
                "embedding_tag": model_tag(embedding_model),
                "regressor": regressor,
                "model_variant": model_variant,
                "block": block,
                "roi": roi,
                "n_voxels": int(mask.sum()),
                "mean_r": float(np.nanmean(vals)),
                "median_r": float(np.nanmedian(vals)),
                "fraction_r_gt_0": float(np.nanmean(vals > 0)),
                "fraction_r_gt_0p1": float(np.nanmean(vals > 0.1)),
                "fraction_r_gt_0p2": float(np.nanmean(vals > 0.2)),
            }
        )
    return rows


def to_wide_rows(rows: Sequence[dict], block_names: Sequence[str]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    keys = ["subject", "lag", "embedding_model", "embedding_tag", "regressor", "roi", "n_voxels"]
    for row in rows:
        key = tuple(row[k] for k in keys)
        out = grouped.setdefault(key, {k: row[k] for k in keys})
        block = row["block"]
        out[f"{block}_mean_fraction"] = row["mean_fraction"]
        out[f"{block}_median_fraction"] = row["median_fraction"]
        out[f"{block}_mean_l2"] = row["mean_l2"]
    ordered = []
    for out in grouped.values():
        for block in block_names:
            out.setdefault(f"{block}_mean_fraction", np.nan)
            out.setdefault(f"{block}_median_fraction", np.nan)
            out.setdefault(f"{block}_mean_l2", np.nan)
        ordered.append(out)
    return ordered


def main() -> None:
    args = parse_args()
    args.embedding_device = resolve_embedding_device(args.embedding_device)
    args.summary_horizons = sorted({int(h) for h in args.summary_horizons})
    args.lags = [int(args.lag)]
    args.chunk_trs = 1
    args.feature_model = "embedding"

    mounted_root = configure_data_root(args)
    stories = load_stories(args)
    train_stories, val_stories = split_stories(stories, args)
    log.info("%s: %d stories | %d train | %d val", args.subject, len(stories), len(train_stories), len(val_stories))
    log.info("Embedding device: %s", args.embedding_device)

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

    sample_resp = get_resp(args.subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    voxels = load_full_frontal_voxels(args.subject, int(sample_resp.shape[1]), args.ba_dir)
    responses_by_story = get_resp(args.subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(arr.shape[0]) for s, arr in responses_by_story.items()}
    masks = load_roi_masks(Path(args.ba_dir).expanduser().resolve(), args.subject, voxels)

    all_rows: list[dict] = []
    r_rows: list[dict] = []
    block_names = ["1TR"] + [f"h{h}" for h in args.summary_horizons]

    for embedding_model in args.embedding_models:
        log.info("==== embedding_model=%s ====", embedding_model)
        args.embedding_model = embedding_model
        one_tr_args = argparse.Namespace(
            subject=args.subject,
            embedding_cache_dir=args.one_tr_cache_dir,
            feature_model="embedding",
            embedding_model=embedding_model,
            chunk_trs=1,
            lag_trs=int(args.lag),
            embed_batch_size=int(args.embed_batch_size),
            embedding_device=args.embedding_device,
        )
        one_tr, one_dim, _one_cache = load_or_build_chunk_embeddings(
            one_tr_args,
            stories,
            resp_lengths,
            response_root=config.DATA_TRAIN_DIR,
        )
        summary_embs, _summary_model, _summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
        combo = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)
        block_slices = {name: slice(i * one_dim, (i + 1) * one_dim) for i, name in enumerate(block_names)}
        x_train, y_train = stack_lag(combo, responses_by_story, train_stories, args.lag)
        x_val, y_val = stack_lag(combo, responses_by_story, val_stories, args.lag)
        if int(args.block_pca_dim) > 0:
            x_train, x_val, block_slices, pca_explained = apply_block_pca(
                x_train=x_train,
                x_val=x_val,
                block_slices=block_slices,
                n_components=int(args.block_pca_dim),
                seed=int(args.seed),
            )
            log.info(
                "Applied block PCA (%d PCs/block): %s",
                int(args.block_pca_dim),
                ", ".join(f"{block}={frac:.3f}" for block, frac in pca_explained.items()),
            )
        log.info("X_train=%s X_val=%s Y_train=%s", x_train.shape, x_val.shape, y_train.shape)

        for regressor in args.regressors:
            log.info("==== %s / %s ====", model_tag(embedding_model), regressor)
            coef, alphas, full_corrs = fit_coefficients(
                regressor=regressor,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                args=args,
            )
            if full_corrs is not None:
                r_rows.extend(
                    summarize_model_r(
                        corrs=full_corrs,
                        masks=masks,
                        subject=args.subject,
                        lag=args.lag,
                        embedding_model=embedding_model,
                        regressor=regressor,
                        model_variant="full_combined",
                        block="all",
                    )
                )
            all_rows.extend(
                summarize_coefficients(
                    coef=coef,
                    alphas=alphas,
                    block_slices=block_slices,
                    masks=masks,
                    subject=args.subject,
                    lag=args.lag,
                    embedding_model=embedding_model,
                    regressor=regressor,
                    elasticnet_alpha=args.elasticnet_alpha,
                    elasticnet_l1_ratio=args.elasticnet_l1_ratio,
                )
            )
            if args.skip_single_blocks:
                continue
            for block, sl in block_slices.items():
                log.info("==== %s / %s / %s only ====", model_tag(embedding_model), regressor, block)
                _coef_block, _alphas_block, block_corrs = fit_coefficients(
                    regressor=regressor,
                    x_train=x_train[:, sl],
                    y_train=y_train,
                    x_val=x_val[:, sl],
                    y_val=y_val,
                    args=args,
                )
                if block_corrs is None:
                    continue
                r_rows.extend(
                    summarize_model_r(
                        corrs=block_corrs,
                        masks=masks,
                        subject=args.subject,
                        lag=args.lag,
                        embedding_model=embedding_model,
                        regressor=regressor,
                        model_variant="block_only",
                        block=block,
                    )
                )

    out_prefix = (
        Path(args.out_prefix).expanduser().resolve()
        if args.out_prefix
        else THIS_DIR / "results" / "combo_coeff_sweep" / f"{args.subject}_combo_coeff_sweep_lag{args.lag}"
    )
    write_csv(out_prefix.with_name(out_prefix.name + "_long.csv"), all_rows)
    wide_rows = to_wide_rows(all_rows, block_names)
    write_csv(out_prefix.with_name(out_prefix.name + "_wide.csv"), wide_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_model_r_summary.csv"), r_rows)
    log.info("Wrote %s", out_prefix.with_name(out_prefix.name + "_wide.csv"))


if __name__ == "__main__":
    main()
