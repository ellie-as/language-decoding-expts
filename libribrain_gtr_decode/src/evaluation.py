from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .utils import l2_normalize


def paired_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = l2_normalize(y_pred.astype(np.float32))
    true = l2_normalize(y_true.astype(np.float32))
    return {
        "mean_cosine_true": float(np.mean(np.sum(pred * true, axis=1))),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
    }


def ranks_from_similarity(sims: np.ndarray, true_cols: np.ndarray) -> np.ndarray:
    true_scores = sims[np.arange(len(sims)), true_cols]
    return (sims > true_scores[:, None]).sum(axis=1) + 1


def retrieval_metrics_from_ranks(ranks: np.ndarray, prefix: str = "") -> dict[str, float]:
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}top1": float(np.mean(ranks <= 1)),
        f"{p}top5": float(np.mean(ranks <= 5)),
        f"{p}top10": float(np.mean(ranks <= 10)),
        f"{p}mrr": float(np.mean(1.0 / ranks)),
        f"{p}median_rank": float(np.median(ranks)),
        f"{p}mean_rank": float(np.mean(ranks)),
    }


def full_retrieval(y_pred: np.ndarray, y_true: np.ndarray, metadata: pd.DataFrame, tolerances: list[float]) -> tuple[dict[str, float], np.ndarray]:
    pred = l2_normalize(y_pred.astype(np.float32))
    true = l2_normalize(y_true.astype(np.float32))
    sims = pred @ true.T
    ranks = ranks_from_similarity(sims, np.arange(len(sims)))
    metrics = retrieval_metrics_from_ranks(ranks, "full")
    metrics.update(relaxed_metrics(sims, metadata, tolerances, "full"))
    return metrics, ranks


def relaxed_metrics(sims: np.ndarray, metadata: pd.DataFrame, tolerances: list[float], prefix: str) -> dict[str, float]:
    out = {}
    runs = metadata["run_group"].to_numpy()
    times = metadata["t"].to_numpy(float)
    order = np.argsort(-sims, axis=1)
    for tol in tolerances:
        relaxed_ranks = []
        for i in range(len(metadata)):
            ok = (runs == runs[i]) & (np.abs(times - times[i]) <= tol)
            ranked_ok = np.flatnonzero(ok[order[i]])
            relaxed_ranks.append(int(ranked_ok[0] + 1) if len(ranked_ok) else len(metadata) + 1)
        ranks = np.asarray(relaxed_ranks)
        label = f"{prefix}_relaxed_{int(tol)}s"
        out[f"{label}_top1"] = float(np.mean(ranks <= 1))
        out[f"{label}_top5"] = float(np.mean(ranks <= 5))
    return out


def sampled_retrieval(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metadata: pd.DataFrame,
    config: dict[str, Any],
    kind: str,
) -> dict[str, float]:
    rng = np.random.default_rng(int(config["project"].get("seed", 0)))
    pred = l2_normalize(y_pred.astype(np.float32))
    true = l2_normalize(y_true.astype(np.float32))
    n = len(metadata)
    ranks = []
    n_skipped = 0
    for i in range(n):
        candidates = candidate_indices(i, metadata, config, kind, rng)
        if len(candidates) == 0:
            n_skipped += 1
            continue
        candidates = np.unique(np.r_[i, candidates])
        sims = pred[i] @ true[candidates].T
        true_col = int(np.where(candidates == i)[0][0])
        ranks.append(int((sims > sims[true_col]).sum() + 1))
    if not ranks:
        metrics = {f"{kind}_{name}": float("nan") for name in ["top1", "top5", "top10", "mrr", "median_rank", "mean_rank"]}
    else:
        metrics = retrieval_metrics_from_ranks(np.asarray(ranks), kind)
    metrics[f"{kind}_n_queries"] = int(len(ranks))
    metrics[f"{kind}_n_skipped_no_distractors"] = int(n_skipped)
    return metrics


def candidate_indices(i: int, metadata: pd.DataFrame, config: dict[str, Any], kind: str, rng: np.random.Generator) -> np.ndarray:
    n = len(metadata)
    runs = metadata["run_group"].to_numpy()
    times = metadata["t"].to_numpy(float)
    if kind == "random_global":
        mask = runs != runs[i]
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            pool = np.setdiff1d(np.arange(n), [i])
        k = min(int(config["evaluation"]["n_random_distractors"]), len(pool))
        return rng.choice(pool, size=k, replace=False) if k else np.array([], dtype=int)
    if kind == "same_run":
        return np.flatnonzero((runs == runs[i]) & (np.arange(n) != i))
    if kind == "nearby":
        dt = np.abs(times - times[i])
        return np.flatnonzero((runs == runs[i]) & (dt >= float(config["evaluation"]["nearby_min_sec"])) & (dt <= float(config["evaluation"]["nearby_max_sec"])))
    raise ValueError(f"Unknown candidate kind: {kind}")


def evaluate_all(y_pred: np.ndarray, y_true: np.ndarray, metadata: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    tolerances = [float(x) for x in config["evaluation"].get("relaxed_tolerances_sec", [config["evaluation"].get("relaxed_tolerance_sec", 10.0)])]
    metrics = paired_regression_metrics(y_true, y_pred)
    full_metrics, ranks = full_retrieval(y_pred, y_true, metadata, tolerances)
    metrics.update(full_metrics)
    rows = [{"distractor_type": "full_test", **full_metrics}]
    for kind, enabled_key in [("random_global", "evaluate_global_random"), ("same_run", "evaluate_same_run"), ("nearby", "evaluate_nearby")]:
        if config["evaluation"].get(enabled_key, True):
            kind_metrics = sampled_retrieval(y_pred, y_true, metadata, config, kind)
            metrics.update(kind_metrics)
            rows.append({"distractor_type": kind, **kind_metrics})
    return metrics, pd.DataFrame(rows), ranks


def make_plots(ranks: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, by_type: pd.DataFrame, plot_dir: str | Path) -> None:
    p = Path(plot_dir)
    p.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(ranks, bins=50)
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.title("Full-test retrieval ranks")
    plt.tight_layout()
    plt.savefig(p / "rank_histogram.png", dpi=150)
    plt.close()

    true_cos = np.sum(l2_normalize(y_pred) * l2_normalize(y_true), axis=1)
    rng = np.random.default_rng(0)
    control = np.sum(l2_normalize(y_pred) * l2_normalize(y_true)[rng.permutation(len(y_true))], axis=1)
    plt.figure()
    plt.hist(true_cos, bins=50, alpha=0.6, label="true")
    plt.hist(control, bins=50, alpha=0.6, label="shuffled")
    plt.xlabel("Cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p / "cosine_true_vs_controls.png", dpi=150)
    plt.close()

    if "distractor_type" in by_type:
        metric_col = next((c for c in ["full_top1", "random_global_top1", "same_run_top1", "nearby_top1"] if c in by_type.columns), None)
        if metric_col:
            plot_df = by_type.set_index("distractor_type").filter(regex="top1$")
            plot_df.plot(kind="bar")
            plt.ylabel("Top-1 accuracy")
            plt.tight_layout()
            plt.savefig(p / "retrieval_by_distractor_type.png", dpi=150)
            plt.close()
