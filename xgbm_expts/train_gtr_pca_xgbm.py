#!/usr/bin/env python3
"""Decode compact GTR text-window embeddings from Huth language voxels.

This is a direct decoding experiment:

    Huth selected voxels -> PCA(GTR 5-TR text-window embedding)

The target cache is the same 5-TR chunk GTR embedding cache used by
``mindeye_text``. PCA is fit on training-story embeddings only. Brain features
are z-scored using training stories only. Validation is story-heldout by
default.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "gpt1_encoding_comparison"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))

import config  # noqa: E402
from compare_gpt1_encoding import DEFAULT_SESSIONS, configure_data_root  # noqa: E402
from _shared import load_or_build_chunk_embeddings  # noqa: E402
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gtr_pca_xgbm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=["S1"])
    p.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_SESSIONS)
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument(
        "--mounted-project-root",
        default="smb://ceph-gw02.hpc.swc.ucl.ac.uk/behrens/ellie/language-decoding-expts",
    )
    p.add_argument("--feature-model", default="gtr-base", choices=["gtr-base", "embedding"])
    p.add_argument("--chunk-trs", type=int, default=5)
    p.add_argument("--lag-trs", type=int, default=3)
    p.add_argument("--brain-offsets", type=int, nargs="+", default=[0])
    p.add_argument(
        "--brain-pca",
        type=int,
        default=0,
        help="If >0, PCA-reduce z-scored brain features before LightGBM.",
    )
    p.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="If >0 and --brain-pca is 0, randomly keep this many brain features.",
    )
    p.add_argument("--pca-dim", type=int, default=10)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embedding-cache-dir", default="27-04-expts/cache")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--output-dir", default="xgbm_expts/results")
    p.add_argument("--tag", default=None)
    p.add_argument("--backend", default="lightgbm", choices=["lightgbm", "sklearn_hist"])
    p.add_argument("--skip-ridge-baseline", action="store_true")
    p.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0, 10000.0],
        help="Alphas for RidgeCV baseline on the same PCA-GTR target.",
    )
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.5)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--reg-alpha", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=10.0)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--save-models", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_stories(sessions: Sequence[int]) -> List[str]:
    with open(Path(config.DATA_TRAIN_DIR) / "sess_to_story.json", encoding="utf-8") as f:
        sess_to_story = json.load(f)
    stories: List[str] = []
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def split_stories(stories: List[str], args: argparse.Namespace) -> tuple[List[str], List[str]]:
    if args.val_stories:
        val = [s for s in args.val_stories if s in stories]
        missing = sorted(set(args.val_stories) - set(val))
        if missing:
            raise ValueError(f"Validation stories not found: {missing}")
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = list(stories)
        rng.shuffle(shuffled)
        val = sorted(shuffled[:max(1, int(args.val_story_count))])
    val_set = set(val)
    train = [s for s in stories if s not in val_set]
    return train, val


def load_huth_voxels(subject: str) -> np.ndarray:
    path = Path(config.MODEL_DIR) / subject / "encoding_model_perceived.npz"
    if not path.exists():
        raise FileNotFoundError(f"Huth encoding model not found: {path}")
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["voxels"], dtype=np.int64)


def stack_embeddings(embeddings: Dict[str, np.ndarray], stories: List[str]) -> np.ndarray:
    return np.vstack([embeddings[s] for s in stories]).astype(np.float32)


def stack_brain_chunks(
    responses: Dict[str, np.ndarray],
    embeddings: Dict[str, np.ndarray],
    stories: List[str],
    lag_trs: int,
    brain_offsets: Sequence[int],
) -> np.ndarray:
    rows = []
    max_offset = max(int(o) for o in brain_offsets)
    for story in stories:
        resp = responses[story]
        n = int(embeddings[story].shape[0])
        need = n + int(lag_trs) + max(max_offset, 0)
        if resp.shape[0] < need:
            raise ValueError(f"{story}: response rows={resp.shape[0]} need at least {need}")
        for i in range(n):
            parts = [resp[i + lag_trs + int(offset)] for offset in brain_offsets]
            rows.append(np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0])
    return np.asarray(rows, dtype=np.float32)


def zscore_train_apply(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return (
        np.nan_to_num((x_train - mean) / std).astype(np.float32),
        np.nan_to_num((x_val - mean) / std).astype(np.float32),
        mean,
        std,
    )


def pca_targets(y_train_raw: np.ndarray, y_val_raw: np.ndarray, pca_dim: int, seed: int):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=int(pca_dim), random_state=seed)
    y_train = pca.fit_transform(y_train_raw).astype(np.float32)
    y_val = pca.transform(y_val_raw).astype(np.float32)
    mean = y_train.mean(axis=0).astype(np.float32)
    std = y_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return (
        np.nan_to_num((y_train - mean) / std).astype(np.float32),
        np.nan_to_num((y_val - mean) / std).astype(np.float32),
        pca,
        mean,
        std,
    )


def reduce_brain_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if int(args.brain_pca) > 0:
        from sklearn.decomposition import PCA

        n_comp = min(int(args.brain_pca), x_train.shape[0] - 1, x_train.shape[1])
        log.info("Brain PCA: %d features -> %d components", x_train.shape[1], n_comp)
        pca = PCA(n_components=n_comp, random_state=args.seed, svd_solver="randomized")
        x_train_pca = pca.fit_transform(x_train).astype(np.float32)
        x_val_pca = pca.transform(x_val).astype(np.float32)
        return (
            x_train_pca,
            x_val_pca,
            {
                "kind": "pca",
                "pca": pca,
                "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            },
        )
    if int(args.max_features) > 0 and int(args.max_features) < x_train.shape[1]:
        rng = np.random.default_rng(args.seed)
        cols = np.sort(rng.choice(x_train.shape[1], size=int(args.max_features), replace=False))
        log.info("Feature subsampling: %d features -> %d columns", x_train.shape[1], len(cols))
        return (
            x_train[:, cols].astype(np.float32, copy=False),
            x_val[:, cols].astype(np.float32, copy=False),
            {"kind": "subsample", "columns": cols},
        )
    return x_train, x_val, {"kind": "none"}


def fit_models(args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray):
    models = []
    if args.backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as err:
            raise ImportError("Install lightgbm or use --backend sklearn_hist") from err
        for dim in range(y_train.shape[1]):
            log.info(
                "Fitting LightGBM target dim %d/%d on X=%s",
                dim + 1,
                y_train.shape[1],
                x_train.shape,
            )
            model = LGBMRegressor(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                n_jobs=args.n_jobs,
                random_state=args.seed + dim,
                verbose=-1,
            )
            model.fit(x_train, y_train[:, dim])
            models.append(model)
        return models

    from sklearn.ensemble import HistGradientBoostingRegressor

    for dim in range(y_train.shape[1]):
        log.info(
            "Fitting sklearn HistGradientBoosting target dim %d/%d on X=%s",
            dim + 1,
            y_train.shape[1],
            x_train.shape,
        )
        model = HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=max(3, 2 ** int(args.max_depth)),
            l2_regularization=args.reg_lambda,
            random_state=args.seed + dim,
        )
        model.fit(x_train, y_train[:, dim])
        models.append(model)
    return models


def predict_models(models, x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in models]).astype(np.float32)


def fit_predict_ridge_baseline(args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray):
    from sklearn.linear_model import RidgeCV

    log.info("Fitting RidgeCV baseline on X=%s Y=%s", x_train.shape, y_train.shape)
    try:
        model = RidgeCV(alphas=args.ridge_alphas, alpha_per_target=True)
    except TypeError:
        model = RidgeCV(alphas=args.ridge_alphas)
    model.fit(x_train, y_train)
    pred = model.predict(x_val).astype(np.float32)
    return model, pred


def per_dim_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64) - pred.mean(axis=0, keepdims=True)
    true = true.astype(np.float64) - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred * pred).sum(axis=0) * (true * true).sum(axis=0))
    return np.divide((pred * true).sum(axis=0), denom, out=np.zeros(pred.shape[1]), where=denom != 0)


def mean_cosine(pred: np.ndarray, true: np.ndarray) -> float:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cos = np.divide((pred * true).sum(axis=1), denom, out=np.zeros(pred.shape[0]), where=denom != 0)
    return float(np.nan_to_num(cos).mean())


def retrieval_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    true_n = true / np.maximum(np.linalg.norm(true, axis=1, keepdims=True), 1e-8)
    sims = pred_n @ true_n.T
    ranks = []
    for i in range(sims.shape[0]):
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks = np.asarray(ranks)
    return {
        "retrieval_top1": float(np.mean(ranks == 1)),
        "retrieval_top10": float(np.mean(ranks <= 10)),
        "retrieval_mrr": float(np.mean(1.0 / ranks)),
        "retrieval_mean_rank": float(ranks.mean()),
    }


def prediction_metrics(pred: np.ndarray, true: np.ndarray, prefix: str = "") -> dict:
    dim_corrs = per_dim_corr(pred, true)
    out = {
        f"{prefix}mean_dim_r": float(np.nan_to_num(dim_corrs).mean()),
        f"{prefix}median_dim_r": float(np.median(np.nan_to_num(dim_corrs))),
        f"{prefix}min_dim_r": float(np.nan_to_num(dim_corrs).min()),
        f"{prefix}max_dim_r": float(np.nan_to_num(dim_corrs).max()),
        f"{prefix}mean_cosine": mean_cosine(pred, true),
    }
    out.update({f"{prefix}{key}": value for key, value in retrieval_metrics(pred, true).items()})
    return out


def build_tag(args: argparse.Namespace, subject: str) -> str:
    offsets = "-".join(str(int(o)) for o in args.brain_offsets)
    return args.tag or (
        f"{subject}__{args.feature_model}__pca{args.pca_dim}__{args.backend}"
        f"__depth{args.max_depth}-est{args.n_estimators}-lr{args.learning_rate:g}"
        f"__offs{offsets}__seed{args.seed}"
    )


def run_subject(args: argparse.Namespace, subject: str, stories: List[str]) -> dict:
    train_stories, val_stories = split_stories(stories, args)
    voxels = load_huth_voxels(subject)
    log.info("[%s] Huth voxels: %d", subject, len(voxels))
    log.info("[%s] Validation stories: %s", subject, ", ".join(val_stories))

    responses = get_resp(subject, stories, stack=False, vox=voxels)
    responses = {story: arr.astype(np.float32) for story, arr in responses.items()}
    resp_lengths = {story: responses[story].shape[0] for story in stories}

    emb_args = argparse.Namespace(
        subject=subject,
        embedding_cache_dir=args.embedding_cache_dir,
        feature_model=args.feature_model,
        chunk_trs=args.chunk_trs,
        lag_trs=args.lag_trs,
        embed_batch_size=args.embed_batch_size,
    )
    embeddings, emb_dim, cache_path = load_or_build_chunk_embeddings(
        emb_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )

    x_train = stack_brain_chunks(responses, embeddings, train_stories, args.lag_trs, args.brain_offsets)
    x_val = stack_brain_chunks(responses, embeddings, val_stories, args.lag_trs, args.brain_offsets)
    y_train_raw = stack_embeddings(embeddings, train_stories)
    y_val_raw = stack_embeddings(embeddings, val_stories)
    if x_train.shape[0] != y_train_raw.shape[0] or x_val.shape[0] != y_val_raw.shape[0]:
        raise RuntimeError("Brain chunk rows and embedding target rows are misaligned.")

    x_train_z, x_val_z, x_mean, x_std = zscore_train_apply(x_train, x_val)
    x_train_model, x_val_model, reducer = reduce_brain_features(x_train_z, x_val_z, args)
    y_train, y_val, pca, y_mean, y_std = pca_targets(y_train_raw, y_val_raw, args.pca_dim, args.seed)

    log.info(
        "[%s] Train X=%s -> %s Y=%s | Val X=%s -> %s Y=%s",
        subject,
        x_train_z.shape,
        x_train_model.shape,
        y_train.shape,
        x_val_z.shape,
        x_val_model.shape,
        y_val.shape,
    )
    start = time.time()
    models = fit_models(args, x_train_model, y_train)
    pred_val = predict_models(models, x_val_model)
    elapsed = time.time() - start

    metrics = {
        "subject": subject,
        "backend": args.backend,
        "feature_model": args.feature_model,
        "pca_dim": int(args.pca_dim),
        "n_voxels": int(len(voxels)),
        "brain_offsets": " ".join(str(int(o)) for o in args.brain_offsets),
        "n_train": int(x_train_z.shape[0]),
        "n_val": int(x_val_z.shape[0]),
        "brain_feature_dim_raw": int(x_train_z.shape[1]),
        "brain_feature_dim_model": int(x_train_model.shape[1]),
        "brain_reducer": reducer["kind"],
        "brain_pca_explained_variance": reducer.get("explained_variance", ""),
        "val_stories": " ".join(val_stories),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "elapsed_sec": float(elapsed),
        "embedding_cache": str(cache_path),
    }
    metrics.update(prediction_metrics(pred_val, y_val))

    ridge_model = None
    ridge_pred_val = None
    ridge_dim_corrs = np.array([], dtype=np.float32)
    if args.skip_ridge_baseline:
        log.info("[%s] Skipping ridge baseline.", subject)
    else:
        ridge_model, ridge_pred_val = fit_predict_ridge_baseline(args, x_train_model, y_train, x_val_model)
        ridge_dim_corrs = per_dim_corr(ridge_pred_val, y_val).astype(np.float32)
        metrics.update(prediction_metrics(ridge_pred_val, y_val, prefix="ridge_"))
        metrics["delta_mean_dim_r_vs_ridge"] = metrics["mean_dim_r"] - metrics["ridge_mean_dim_r"]
        metrics["delta_mean_cosine_vs_ridge"] = metrics["mean_cosine"] - metrics["ridge_mean_cosine"]
        metrics["delta_retrieval_top1_vs_ridge"] = metrics["retrieval_top1"] - metrics["ridge_retrieval_top1"]
        metrics["delta_retrieval_top10_vs_ridge"] = metrics["retrieval_top10"] - metrics["ridge_retrieval_top10"]

    out_dir = Path(args.output_dir).expanduser().resolve() / build_tag(args, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "predictions.npz",
        pred_val=pred_val,
        true_val=y_val,
        ridge_pred_val=ridge_pred_val if ridge_pred_val is not None else np.array([], dtype=np.float32),
        pred_val_raw_pca=(pred_val * y_std + y_mean),
        true_val_raw_pca=(y_val * y_std + y_mean),
        dim_corrs=per_dim_corr(pred_val, y_val).astype(np.float32),
        ridge_dim_corrs=ridge_dim_corrs,
        voxels=voxels,
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    if args.save_models:
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "models": models,
                    "ridge_model": ridge_model,
                    "pca": pca,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "brain_reducer": reducer,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "args": vars(args),
                    "metrics": metrics,
                },
                f,
            )
    log.info("[%s] mean_dim_r=%.4f top1=%.4f top10=%.4f cos=%.4f -> %s",
             subject, metrics["mean_dim_r"], metrics["retrieval_top1"],
             metrics["retrieval_top10"], metrics["mean_cosine"], out_dir)
    return metrics


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    data_root = args.data_root
    if args.local_compute_mode and data_root is None:
        data_root = args.mounted_project_root
    configure_data_root(data_root)

    stories = load_stories(args.sessions)
    all_rows = [run_subject(args, subject, stories) for subject in args.subjects]

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "summary.csv"
    fields = sorted({key for row in all_rows for key in row})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {csv_path}")
    for row in all_rows:
        print(
            f"{row['subject']} mean_dim_r={row['mean_dim_r']:.4f} "
            f"cos={row['mean_cosine']:.4f} top1={row['retrieval_top1']:.4f} "
            f"top10={row['retrieval_top10']:.4f} pca_var={row['pca_explained_variance']:.3f}"
        )


if __name__ == "__main__":
    main()
