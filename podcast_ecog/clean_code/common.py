"""Shared utilities for the clean Podcast ECoG decoding pipeline."""

from __future__ import annotations

import json
import string
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PODCAST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BIDS_ROOT = Path("/Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574")
DEFAULT_OUTPUT_ROOT = PODCAST_DIR / "outputs" / "clean_code"
DEFAULT_BOUNDARY_JSON = PODCAST_DIR / "outputs" / "transcript_boundaries.json"
DEFAULT_ROI_METRICS = PODCAST_DIR / "outputs" / "gpt2_paper_roi_context_layer_exact" / "channel_paper_roi_metrics.csv"
ROI_ORDER = ["ALL", "EAC", "STG", "IFG", "PRC", "MFG", "TMP"]


def clean_subject(subject: str) -> str:
    return str(subject).replace("sub-", "")


def subject_label(subject: str) -> str:
    return f"sub-{clean_subject(subject)}"


def highgamma_fif_path(root: Path, subject: str, task: str = "podcast") -> Path:
    sub = subject_label(subject)
    return root / "derivatives" / "ecogprep" / sub / "ieeg" / f"{sub}_task-{task}_desc-highgamma_ieeg.fif"


def transcript_words_from_token_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    tokens = pd.read_csv(path, sep="\t", index_col=0)
    rows = []
    for word_idx, group in tokens.groupby("word_idx", sort=True):
        rows.append(
            {
                "word_idx": int(word_idx),
                "word": str(group["word"].iloc[0]),
                "start": float(group["start"].iloc[0]),
                "end": float(group["end"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values("word_idx").reset_index(drop=True)


def load_dataset_static_word_vectors(root: Path, feature_space: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    stim_dir = root / "stimuli" / feature_space
    transcript = transcript_words_from_token_table(stim_dir / "transcript.tsv")
    token_table = pd.read_csv(stim_dir / "transcript.tsv", sep="\t", index_col=0)
    with h5py.File(stim_dir / "features.hdf5", "r") as handle:
        token_vectors = handle["vectors"][...].astype(np.float32)

    vectors = []
    coverage_rows = []
    for word_idx, group in token_table.groupby("word_idx", sort=True):
        token_rows = group.index.to_numpy()
        token_vec = token_vectors[token_rows]
        finite = np.isfinite(token_vec).all(axis=1) & (np.linalg.norm(token_vec, axis=1) > 0)
        if finite.any():
            vectors.append(token_vec[finite].mean(axis=0))
            matched = str(group.loc[finite, "token"].iloc[0])
            in_vocab = True
        else:
            vectors.append(np.full(token_vectors.shape[1], np.nan, dtype=np.float32))
            matched = None
            in_vocab = False
        coverage_rows.append(
            {
                "word_idx": int(word_idx),
                "word": str(group["word"].iloc[0]),
                "matched_token": matched,
                "in_vocab": bool(in_vocab),
            }
        )
    return transcript, np.vstack(vectors).astype(np.float32), pd.DataFrame(coverage_rows)


def clean_word_candidates(word: str) -> list[str]:
    raw = str(word).strip()
    lower = raw.lower()
    punct = string.punctuation + "“”‘’"
    bare = lower.strip(punct)
    candidates = [raw, lower, bare]
    if bare.endswith("'s"):
        candidates.append(bare[:-2])
    if bare.endswith("s'"):
        candidates.append(bare[:-2])
    if "-" in bare:
        candidates.append(bare.replace("-", "_"))
        candidates.extend(part for part in bare.split("-") if part)
    out = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def compute_word_pca(vectors: np.ndarray, n_components: int) -> tuple[np.ndarray, pd.DataFrame]:
    finite = np.isfinite(vectors).all(axis=1)
    scaler = StandardScaler()
    z = scaler.fit_transform(vectors[finite].astype(np.float64))
    pca = PCA(n_components=int(n_components), random_state=0)
    scores_finite = pca.fit_transform(z).astype(np.float32)
    scores = np.full((len(vectors), int(n_components)), np.nan, dtype=np.float32)
    scores[finite] = scores_finite
    explained = pd.DataFrame(
        {
            "component": np.arange(1, int(n_components) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return scores, explained


def segment_means(data: np.ndarray, centers: np.ndarray, sfreq: float, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    start_offsets = int(round(float(start_s) * float(sfreq)))
    end_offsets = int(round(float(end_s) * float(sfreq)))
    starts = centers.astype(np.int64) + start_offsets
    stops = centers.astype(np.int64) + end_offsets
    valid = (starts >= 0) & (stops <= data.shape[1]) & (stops > starts)
    means = np.full((len(centers), data.shape[0]), np.nan, dtype=np.float32)
    if valid.any():
        csum = np.concatenate(
            [np.zeros((data.shape[0], 1), dtype=np.float64), np.cumsum(data.astype(np.float64), axis=1)],
            axis=1,
        )
        seg_sum = csum[:, stops[valid]] - csum[:, starts[valid]]
        means[valid] = (seg_sum / (stops[valid] - starts[valid])[None, :]).T.astype(np.float32)
    return means, valid


def target_indices(current_indices: np.ndarray, lag: int, direction: str) -> np.ndarray:
    if direction == "past":
        return current_indices.astype(np.int64) - int(lag)
    if direction == "future":
        return current_indices.astype(np.int64) + int(lag)
    raise ValueError(f"Unknown direction: {direction!r}")


def valid_indices_for_lag_pair(valid_neural: np.ndarray, word_scores: np.ndarray, lag: int) -> np.ndarray:
    current = np.flatnonzero(valid_neural)
    keep = np.ones(len(current), dtype=bool)
    for direction in ("past", "future"):
        target = target_indices(current, lag, direction)
        in_range = (target >= 0) & (target < len(word_scores))
        finite = np.zeros(len(current), dtype=bool)
        finite[in_range] = np.isfinite(word_scores[target[in_range]]).all(axis=1)
        keep &= in_range & finite
    return current[keep]


def load_boundary_json(path: Path) -> dict[str, list[int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = ["sentence_end_word_indices", "event_end_word_indices"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")
    return {key: sorted(set(int(v) for v in data[key])) for key in required}


def boundary_after_from_ends(n_words: int, ends: list[int]) -> np.ndarray:
    boundary_after = np.zeros(int(n_words), dtype=bool)
    for end in ends:
        if 0 <= int(end) < int(n_words) - 1:
            boundary_after[int(end)] = True
    return boundary_after


def segment_ids_from_boundary_after(boundary_after: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(boundary_after[:-1], dtype=np.int32)].astype(np.int32)


def boundary_prefix(boundary_after: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(boundary_after.astype(np.int32))]


def crossed_between(prefix: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    left = np.minimum(idx_a, idx_b).astype(np.int32)
    right = np.maximum(idx_a, idx_b).astype(np.int32)
    return (prefix[right] - prefix[left]) > 0


def row_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    y_pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(y_true_centered * y_true_centered, axis=1) * np.sum(y_pred_centered * y_pred_centered, axis=1))
    return np.divide(
        np.sum(y_true_centered * y_pred_centered, axis=1),
        denom,
        out=np.full(len(y_true), np.nan, dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)


def component_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yt * yt, axis=0) * np.sum(yp * yp, axis=0))
    return np.divide(
        np.sum(yt * yp, axis=0),
        denom,
        out=np.full(y_true.shape[1], np.nan, dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pc_r = component_corr(y_true, y_pred)
    y_centered = y_true - y_true.mean(axis=0, keepdims=True)
    ss_total = np.sum(y_centered * y_centered, axis=0)
    weights = ss_total / np.maximum(ss_total.sum(), 1e-12)
    point_r = row_corr(y_true, y_pred)
    return {
        "mean_point_r": float(np.nanmean(point_r)),
        "median_point_r": float(np.nanmedian(point_r)),
        "mean_pc_r": float(np.nanmean(pc_r)),
        "median_pc_r": float(np.nanmedian(pc_r)),
        "pc1_r": float(pc_r[0]),
        "weighted_pc_r": float(np.nansum(pc_r * weights)),
    }


def sem(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def signflip_pvalues(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(values.mean())
    signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * len(values)))).T.reshape(-1, len(values))
    null = (signs * values).mean(axis=1)
    p_one = float(np.mean(null >= observed))
    p_two = float(np.mean(np.abs(null) >= abs(observed)))
    return observed, p_one, p_two


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q
