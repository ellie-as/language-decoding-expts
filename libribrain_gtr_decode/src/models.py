from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.random_projection import GaussianRandomProjection

from .evaluation import evaluate_all, make_plots
from .io_utils import ensure_dir, save_json
from .meg_features import load_meg_memmap
from .preprocessing import BatchStandardizer
from .utils import batched_indices, l2_normalize


def train_ridge_baseline(
    meg_path: str | Path,
    meg_info: dict[str, Any],
    meg_metadata: pd.DataFrame,
    embeddings: np.ndarray,
    embeddings_metadata: pd.DataFrame,
    split_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    control: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    ensure_dir(out / "models")
    ensure_dir(out / "predictions")
    ensure_dir(out / "results")

    aligned = meg_metadata[["example_id"]].merge(
        embeddings_metadata.reset_index().rename(columns={"index": "embedding_index"})[["example_id", "embedding_index"]],
        on="example_id",
        how="inner",
    ).merge(split_df[["example_id", "split"]], on="example_id", how="inner")
    if len(aligned) != len(meg_metadata):
        meg_metadata = meg_metadata.merge(aligned[["example_id"]], on="example_id", how="inner")
        aligned = meg_metadata[["example_id"]].merge(
            embeddings_metadata.reset_index().rename(columns={"index": "embedding_index"})[["example_id", "embedding_index"]],
            on="example_id",
        ).merge(split_df[["example_id", "split"]], on="example_id")

    x = load_meg_memmap(meg_path, meg_info)
    y = embeddings[aligned["embedding_index"].to_numpy()]
    y = _apply_control(y, meg_metadata, control, config)

    indices = np.arange(len(aligned))
    split_indices = {s: indices[aligned["split"].to_numpy() == s] for s in ["train", "val", "test"]}
    if len(split_indices["val"]) == 0:
        split_indices["val"] = split_indices["train"]
    if len(split_indices["test"]) == 0:
        split_indices["test"] = split_indices["val"]

    features_cfg = config["features"]
    batch_size = int(features_cfg.get("pca_batch_size", features_cfg.get("projection_batch_size", 2048)))
    scaler = BatchStandardizer().fit(x, split_indices["train"], batch_size=batch_size)
    n_train = len(split_indices["train"])
    n_features = int(np.prod(x.shape[1:]))
    reducer_name = str(features_cfg.get("reducer", "pca")).lower()
    if reducer_name in {"random_projection", "rp", "gaussian_random_projection"}:
        n_components = min(int(features_cfg.get("projection_components", features_cfg.get("pca_components", 4096))), n_features)
        reducer = GaussianRandomProjection(
            n_components=n_components,
            random_state=int(config["project"].get("seed", 0)),
        )
        # Fit establishes the random matrix shape; it does not inspect data values.
        reducer.fit(np.zeros((1, n_features), dtype=np.float32))
        z = {split: _transform_split(x, idx, scaler, reducer, batch_size) for split, idx in split_indices.items()}
    elif reducer_name in {"none", "identity", "raw"} or int(features_cfg.get("pca_components", 0)) == 0:
        n_components = n_features
        reducer = None
        z = {split: _transform_split(x, idx, scaler, reducer, batch_size) for split, idx in split_indices.items()}
    elif reducer_name == "pca":
        n_components = min(int(features_cfg["pca_components"]), max(1, n_train - 1), n_features)
        reducer = IncrementalPCA(n_components=n_components, batch_size=max(batch_size, n_components))
        for batch in scaler.transform_indices(x, split_indices["train"], batch_size=max(batch_size, n_components)):
            if batch.shape[0] >= n_components:
                reducer.partial_fit(batch)
        z = {split: _transform_split(x, idx, scaler, reducer, batch_size) for split, idx in split_indices.items()}
    else:
        raise ValueError(f"Unknown features.reducer={reducer_name!r}")

    best_alpha = None
    best_score = -np.inf
    best_model = None
    val_metrics_by_alpha = {}
    for alpha in config["ridge"]["alphas"]:
        model = Ridge(alpha=float(alpha))
        model.fit(z["train"], y[split_indices["train"]])
        val_pred = model.predict(z["val"]).astype(np.float32)
        if config["ridge"].get("normalize_predictions", True):
            val_pred = l2_normalize(val_pred).astype(np.float32)
        metrics, _, _ = evaluate_all(val_pred, y[split_indices["val"]], meg_metadata.iloc[split_indices["val"]].reset_index(drop=True), config)
        score = float(metrics.get("same_run_mrr", metrics.get("full_mrr", metrics.get("random_global_mrr", 0.0))))
        val_metrics_by_alpha[str(alpha)] = metrics
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_model = model

    if best_model is None:
        raise RuntimeError("No ridge model was fit")

    val_pred = best_model.predict(z["val"]).astype(np.float32)
    test_pred = best_model.predict(z["test"]).astype(np.float32)
    if config["ridge"].get("normalize_predictions", True):
        val_pred = l2_normalize(val_pred).astype(np.float32)
        test_pred = l2_normalize(test_pred).astype(np.float32)

    suffix = f"_{control}" if control else ""
    np.save(out / "predictions" / f"ridge_val_predictions{suffix}.npy", val_pred)
    np.save(out / "predictions" / f"ridge_test_predictions{suffix}.npy", test_pred)
    joblib.dump(best_model, out / "models" / f"ridge_gtr_base_10s{suffix}.joblib")
    joblib.dump(reducer, out / "models" / f"reducer_gtr_base_10s{suffix}.joblib")
    joblib.dump(scaler, out / "models" / f"scaler_gtr_base_10s{suffix}.joblib")

    val_metrics, val_by_type, val_ranks = evaluate_all(val_pred, y[split_indices["val"]], meg_metadata.iloc[split_indices["val"]].reset_index(drop=True), config)
    test_metrics, test_by_type, test_ranks = evaluate_all(test_pred, y[split_indices["test"]], meg_metadata.iloc[split_indices["test"]].reset_index(drop=True), config)
    result = {
        "control": control or "aligned",
        "best_alpha": best_alpha,
        "best_val_selection_score": best_score,
        "reducer": reducer_name,
        "n_components": n_components,
        "n_train": int(len(split_indices["train"])),
        "n_val": int(len(split_indices["val"])),
        "n_test": int(len(split_indices["test"])),
        "val_metrics_by_alpha": val_metrics_by_alpha,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(result, out / "results" / f"ridge_metrics{suffix}.json")
    save_json(val_metrics, out / "results" / f"ridge_val_metrics{suffix}.json")
    save_json(test_metrics, out / "results" / f"ridge_test_metrics{suffix}.json")
    val_by_type.to_csv(out / "results" / f"retrieval_val_by_distractor_type{suffix}.csv", index=False)
    test_by_type.to_csv(out / "results" / f"retrieval_test_by_distractor_type{suffix}.csv", index=False)
    np.save(out / "results" / f"ridge_val_ranks{suffix}.npy", val_ranks)
    np.save(out / "results" / f"ridge_test_ranks{suffix}.npy", test_ranks)
    if control is None:
        make_plots(test_ranks, test_pred, y[split_indices["test"]], test_by_type, out / "plots")
    return result


def _transform_split(x, indices, scaler: BatchStandardizer, reducer, batch_size: int) -> np.ndarray:
    chunks = []
    for batch in scaler.transform_indices(x, indices, batch_size=batch_size):
        transformed = batch if reducer is None else reducer.transform(batch)
        chunks.append(transformed.astype(np.float32))
    if chunks:
        return np.vstack(chunks)
    n_components = x.shape[1] if reducer is None else reducer.n_components
    return np.empty((0, n_components), dtype=np.float32)


def _apply_control(y: np.ndarray, metadata: pd.DataFrame, control: str | None, config: dict[str, Any]) -> np.ndarray:
    if control is None:
        return y
    rng = np.random.default_rng(int(config["project"].get("seed", 0)))
    y2 = y.copy()
    if control == "shuffled_labels":
        return y2[rng.permutation(len(y2))]
    if control == "time_shift":
        shift = float(config["ridge"].get("time_shift_sec", 60.0))
        out = y2.copy()
        for _, idx in metadata.groupby("run_group").indices.items():
            idx = np.asarray(idx)
            times = metadata.iloc[idx]["t"].to_numpy(float)
            shifted = []
            for t in times:
                shifted.append(idx[int(np.argmin(np.abs(times - (t + shift))))])
            out[idx] = y2[np.asarray(shifted)]
        return out
    raise ValueError(f"Unknown control: {control}")
