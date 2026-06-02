"""Shared helpers for word-position and time-based preferred-lag encoding analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_ROOT, component_corr, subject_label


def resolve_target_path(
    prepared_dir: Path,
    target_stem: str | None,
    target_file: Path | None,
    n_components: int,
) -> tuple[Path, str]:
    if target_file is not None:
        path = target_file
        target_name = path.stem
    else:
        target_name = target_stem or f"word_vectors_pca{n_components}"
        path = prepared_dir / f"{target_name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, target_name


def load_roi_lookup(prepared_dir: Path) -> pd.DataFrame:
    path = prepared_dir / "roi_channels.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["subject", "channel", "paper_roi"])
    return pd.read_csv(path)[["subject", "channel", "paper_roi"]].drop_duplicates()


def channel_roi_labels(channels: list[str], roi_lookup: pd.DataFrame, subject: str) -> list[str]:
    sub = subject_label(subject)
    sub_lookup = roi_lookup[roi_lookup["subject"] == sub]
    mapping = dict(zip(sub_lookup["channel"].astype(str), sub_lookup["paper_roi"].astype(str)))
    return [mapping.get(channel, "") for channel in channels]


def select_channels(channels: list[str], roi_labels: list[str], rois: list[str]) -> np.ndarray:
    if "ALL" in {r.upper() for r in rois}:
        return np.arange(len(channels), dtype=int)
    wanted = {r for r in rois}
    return np.asarray([i for i, label in enumerate(roi_labels) if label in wanted], dtype=int)


def load_prepared_scalar_neural(prepared_dir: Path, subject: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sub = subject_label(subject)
    npz_path = prepared_dir / "subjects" / sub / "neural_word_features.npz"
    channel_path = prepared_dir / "subjects" / sub / "channels.csv"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not channel_path.is_file():
        raise FileNotFoundError(channel_path)
    npz = np.load(npz_path)
    channels = pd.read_csv(channel_path)["channel"].astype(str).tolist()
    return npz["x"].astype(np.float32), npz["valid"].astype(bool), channels


def load_prepared_epoch_neural(
    prepared_dir: Path, subject: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sub = subject_label(subject)
    npz_path = prepared_dir / "subjects" / sub / "neural_word_epochs.npz"
    channel_path = prepared_dir / "subjects" / sub / "channels.csv"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{npz_path} not found. Run prepare_data.py with --prepare-epochs on this cache first."
        )
    if not channel_path.is_file():
        raise FileNotFoundError(channel_path)
    npz = np.load(npz_path)
    channels = pd.read_csv(channel_path)["channel"].astype(str).tolist()
    return (
        npz["epochs"].astype(np.float32),
        npz["times"].astype(np.float32),
        npz["valid"].astype(bool),
        channels,
    )


def cv_encode_predict(x_emb: np.ndarray, y_neural: np.ndarray, alpha: float, outer_splits: int) -> np.ndarray:
    pred = np.zeros_like(y_neural, dtype=np.float32)
    cv = KFold(n_splits=int(outer_splits), shuffle=False)
    for train_idx, test_idx in cv.split(x_emb):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_emb[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x_emb[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y_neural[train_idx]).astype(np.float32)
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(x_train, y_train)
        pred[test_idx] = y_scaler.inverse_transform(model.predict(x_test)).astype(np.float32)
    return pred


def corr_across_words(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson r across words (axis 0) for each target column."""
    return component_corr(y_true, y_pred)


def circular_shift_null_pvalues_scalar_lags(
    y: np.ndarray,
    preds: list[np.ndarray | None],
    observed_best: np.ndarray,
    n_iters: int,
    min_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_words, n_channels = y.shape
    ms = int(min(max(1, min_shift), max(1, n_words // 4)))
    valid_preds = [pred for pred in preds if pred is not None]
    ge = np.zeros(n_channels, dtype=np.int64)
    for _ in range(int(n_iters)):
        shift = int(rng.integers(ms, n_words - ms)) if n_words - ms > ms else 1
        y_shift = np.roll(y, shift, axis=0)
        best = np.full(n_channels, -np.inf, dtype=np.float32)
        for pred in valid_preds:
            best = np.maximum(best, np.nan_to_num(corr_across_words(y_shift, pred), nan=-np.inf))
        ge += (best >= observed_best).astype(np.int64)
    return (ge + 1.0) / (int(n_iters) + 1.0)


def circular_shift_null_pvalues_time(
    y_epochs: np.ndarray,
    y_pred: np.ndarray,
    observed_best: np.ndarray,
    n_iters: int,
    min_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null for best-over-times encoding correlation (one model, lag selection over time)."""
    n_words, n_channels, n_times = y_epochs.shape
    ms = int(min(max(1, min_shift), max(1, n_words // 4)))
    ge = np.zeros(n_channels, dtype=np.int64)
    y_flat = y_epochs.reshape(n_words, -1)
    pred_flat = y_pred.reshape(n_words, -1)
    for _ in range(int(n_iters)):
        shift = int(rng.integers(ms, n_words - ms)) if n_words - ms > ms else 1
        y_shift = np.roll(y_flat, shift, axis=0)
        corrs = corr_across_words(y_shift, pred_flat).reshape(n_channels, n_times)
        best = np.nanmax(corrs, axis=1)
        ge += (best >= observed_best).astype(np.int64)
    return (ge + 1.0) / (int(n_iters) + 1.0)


def default_prepared_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / "prepared"
