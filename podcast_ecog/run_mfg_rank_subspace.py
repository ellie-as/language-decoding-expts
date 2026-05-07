#!/usr/bin/env python3
"""Decode linguistic rank content from MFG pseudo-population activity.

This is an exploratory naturalistic analogue of sequence working-memory
subspace analyses. For each current word t, the neural feature is the MFG
high-gamma population vector in a chosen peri-word time window. Targets are
GPT-2 word embeddings at ranks t, t-1, t-2, ... . The script evaluates rank
decoding and compares the decoder subspaces for different ranks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_text_window_encoding import DEFAULT_MOUNT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "mfg_rank_subspace")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--roi", default="MFG")
    parser.add_argument(
        "--roi-metrics",
        type=Path,
        default=THIS_DIR / "outputs" / "gpt2_paper_roi_context_layer_exact" / "channel_paper_roi_metrics.csv",
    )
    parser.add_argument("--feature-space", default="gpt2-xl")
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--neural-windows",
        nargs="+",
        default=["central:-0.5:0.5", "post:0.5:1.5", "late:1.5:4.0", "pre:-1.5:-0.5"],
        help="Window specs as label:start_s:end_s.",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--subspace-dim", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels-per-subject", type=int, default=None)
    return parser.parse_args()


def bids_root(args: argparse.Namespace) -> Path:
    return args.bids_root or args.mounted_root / "podcast_ecog" / "data" / "ds005574"


def fif_path(root: Path, subject: str, task: str) -> Path:
    return (
        root
        / "derivatives"
        / "ecogprep"
        / f"sub-{subject}"
        / "ieeg"
        / f"sub-{subject}_task-{task}_desc-highgamma_ieeg.fif"
    )


def parse_window(spec: str) -> tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected window spec label:start:end, got {spec!r}")
    label, start, end = parts
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        raise ValueError(f"Window end must exceed start in {spec!r}")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
    return safe_label, start_f, end_f


def load_word_embeddings(root: Path, feature_space: str, layer: int, max_words: int | None) -> tuple[np.ndarray, pd.DataFrame]:
    transcript_path = root / "stimuli" / feature_space / "transcript.tsv"
    features_path = root / "stimuli" / feature_space / "features.hdf5"
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    if not features_path.is_file():
        raise FileNotFoundError(features_path)

    transcript = pd.read_csv(transcript_path, sep="\t", index_col=0)
    with h5py.File(features_path, "r") as handle:
        token_embeddings = handle[f"layer-{layer}"][...].astype(np.float32)
    if len(transcript) != len(token_embeddings):
        raise ValueError(f"Transcript rows {len(transcript)} != token embeddings {len(token_embeddings)}")

    word_embeddings = []
    word_rows = []
    for word_idx, group in transcript.groupby("word_idx", sort=True):
        if max_words is not None and len(word_embeddings) >= int(max_words):
            break
        token_idx = group.index.to_numpy()
        word_embeddings.append(token_embeddings[token_idx].mean(axis=0))
        word_rows.append(
            {
                "word_idx": int(word_idx),
                "word": str(group["word"].iloc[0]),
                "start": float(group["start"].iloc[0]),
                "end": float(group["end"].iloc[-1]),
            }
        )
    return np.vstack(word_embeddings).astype(np.float32), pd.DataFrame(word_rows)


def load_roi_channels(path: Path, roi: str, max_channels_per_subject: int | None) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["paper_roi"] == roi].copy()
    out = {}
    for subject, group in df.groupby("subject", sort=True):
        channels = group["channel"].astype(str).tolist()
        if max_channels_per_subject is not None:
            channels = channels[: int(max_channels_per_subject)]
        out[str(subject).replace("sub-", "")] = channels
    return out


def segment_means(data: np.ndarray, centers: np.ndarray, sfreq: float, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    start_offsets = int(round(start_s * sfreq))
    end_offsets = int(round(end_s * sfreq))
    starts = centers + start_offsets
    stops = centers + end_offsets
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


def load_subject_window_features(
    *,
    root: Path,
    subject: str,
    task: str,
    words: pd.DataFrame,
    channels: list[str],
    windows: list[tuple[str, float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    path = fif_path(root, subject, task)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    available = [ch for ch in channels if ch in raw.ch_names]
    if not available:
        raise ValueError(f"No requested {subject} channels found in {path}")
    raw.pick(available)
    print(f"sub-{subject}: reading {len(available)} {raw.info['sfreq']:.1f} Hz channels", flush=True)
    data = raw.get_data().astype(np.float32)
    centers = raw.time_as_index(words["start"].to_numpy(dtype=float), use_rounding=True)

    features = {}
    valids = {}
    for label, start_s, end_s in windows:
        features[label], valids[label] = segment_means(data, centers, float(raw.info["sfreq"]), start_s, end_s)
    return features, valids, available


def standardize_pair(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    return (
        x_scaler.fit_transform(x_train).astype(np.float32),
        x_scaler.transform(x_test).astype(np.float32),
        y_scaler.fit_transform(y_train).astype(np.float32),
        y_scaler.transform(y_test).astype(np.float32),
    )


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(x_train, full_matrices=False)
    scale = (s / (s * s + float(alpha))).astype(np.float32)
    coef = vt.T @ (scale[:, None] * (u.T @ y_train))
    return (x_test @ coef).astype(np.float32), coef.astype(np.float32)


def target_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yt * yt, axis=0) * np.sum(yp * yp, axis=0))
    return np.divide(np.sum(yt * yp, axis=0), denom, out=np.zeros(y_true.shape[1], dtype=np.float32), where=denom > 0)


def sample_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true / np.maximum(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-8)
    yp = y_pred / np.maximum(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-8)
    return np.sum(yt * yp, axis=1)


def retrieval_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    yt = y_true / np.maximum(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-8)
    yp = y_pred / np.maximum(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-8)
    sim = yp @ yt.T
    true_scores = np.diag(sim)
    ranks = (sim > true_scores[:, None]).sum(axis=1) + 1
    return float((ranks == 1).mean()), float((ranks <= 5).mean()), float(np.median(ranks))


def fit_full_decoder(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xz = StandardScaler().fit_transform(x).astype(np.float32)
    yz = StandardScaler().fit_transform(y).astype(np.float32)
    _, coef = ridge_fit_predict(xz, yz, xz[:1], alpha)
    return coef


def evaluate_window(
    *,
    x: np.ndarray,
    embeddings: np.ndarray,
    word_indices: np.ndarray,
    ranks: list[int],
    outer_splits: int,
    alpha: float,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    rows = []
    predictions: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    coefs: dict[int, np.ndarray] = {}
    cv = KFold(n_splits=outer_splits, shuffle=False)
    for rank in ranks:
        y = embeddings[word_indices - int(rank)]
        targets[rank] = y
        pred = np.zeros_like(y, dtype=np.float32)
        fold_corrs = []
        fold_cos = []
        fold_top1 = []
        fold_top5 = []
        fold_med_rank = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
            x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
            y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
            y_pred, _ = ridge_fit_predict(x_train, y_train, x_test, alpha)
            y_pred_raw = y_scaler.inverse_transform(y_pred).astype(np.float32)
            pred[test_idx] = y_pred_raw
            y_test_raw = y[test_idx]
            fold_corrs.append(target_corr(y_test_raw, y_pred_raw))
            fold_cos.append(sample_cosine(y_test_raw, y_pred_raw))
            top1, top5, med_rank = retrieval_scores(y_test_raw, y_pred_raw)
            fold_top1.append(top1)
            fold_top5.append(top5)
            fold_med_rank.append(med_rank)
        corr = np.concatenate(fold_corrs)
        cos = np.concatenate(fold_cos)
        rows.append(
            {
                "rank": int(rank),
                "n_samples": int(len(x)),
                "n_channels": int(x.shape[1]),
                "mean_feature_corr": float(np.nanmean(corr)),
                "median_feature_corr": float(np.nanmedian(corr)),
                "mean_sample_cosine": float(np.nanmean(cos)),
                "median_sample_cosine": float(np.nanmedian(cos)),
                "retrieval_top1": float(np.mean(fold_top1)),
                "retrieval_top5": float(np.mean(fold_top5)),
                "retrieval_median_rank": float(np.median(fold_med_rank)),
            }
        )
        predictions[rank] = pred
        coefs[rank] = fit_full_decoder(x, y, alpha)
    return pd.DataFrame(rows), predictions, targets, coefs


def decoder_subspace(coef: np.ndarray, dim: int) -> np.ndarray:
    u, _s, _vt = np.linalg.svd(coef, full_matrices=False)
    return u[:, : min(dim, u.shape[1])]


def subspace_angle_tables(coefs: dict[int, np.ndarray], dim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = sorted(coefs)
    rows = []
    mean_matrix = pd.DataFrame(index=ranks, columns=ranks, dtype=float)
    min_matrix = pd.DataFrame(index=ranks, columns=ranks, dtype=float)
    spaces = {rank: decoder_subspace(coef, dim) for rank, coef in coefs.items()}
    for a in ranks:
        for b in ranks:
            angles = np.degrees(subspace_angles(spaces[a], spaces[b]))
            rows.append(
                {
                    "rank_a": int(a),
                    "rank_b": int(b),
                    "mean_angle_deg": float(np.mean(angles)),
                    "min_angle_deg": float(np.min(angles)),
                    "max_angle_deg": float(np.max(angles)),
                }
            )
            mean_matrix.loc[a, b] = float(np.mean(angles))
            min_matrix.loc[a, b] = float(np.min(angles))
    return pd.DataFrame(rows), mean_matrix


def cross_rank_generalization(predictions: dict[int, np.ndarray], targets: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for train_rank, pred in predictions.items():
        for target_rank, y in targets.items():
            rows.append(
                {
                    "train_rank": int(train_rank),
                    "target_rank": int(target_rank),
                    "mean_sample_cosine": float(np.nanmean(sample_cosine(y, pred))),
                    "mean_feature_corr": float(np.nanmean(target_corr(y, pred))),
                }
            )
    return pd.DataFrame(rows)


def plot_rank_decoding(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for label, group in summary.groupby("window"):
        group = group.sort_values("rank")
        ax.plot(group["rank"], group["mean_sample_cosine"], marker="o", label=label)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("linguistic rank back from current word")
    ax.set_ylabel("held-out sample cosine")
    ax.legend(title="neural window")
    fig.savefig(output_dir / "rank_decoding_by_window.png", dpi=220)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("rank")
    ax.set_ylabel("rank")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = bids_root(args)
    args.output_dir.joinpath("config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    windows = [parse_window(spec) for spec in args.neural_windows]
    ranks = sorted(set(int(rank) for rank in args.ranks))
    max_rank = max(ranks)
    embeddings, words = load_word_embeddings(root, args.feature_space, args.target_layer, args.max_words)
    roi_channels = load_roi_channels(args.roi_metrics, args.roi, args.max_channels_per_subject)

    subject_features = {}
    subject_valids = {}
    channel_labels = {}
    for subject in args.subjects:
        channels = roi_channels.get(subject, [])
        if not channels:
            print(f"Skipping sub-{subject}: no {args.roi} channels in ROI table", flush=True)
            continue
        features, valids, used_channels = load_subject_window_features(
            root=root,
            subject=subject,
            task=args.task,
            words=words,
            channels=channels,
            windows=windows,
        )
        subject_features[subject] = features
        subject_valids[subject] = valids
        channel_labels[subject] = [f"sub-{subject}:{channel}" for channel in used_channels]

    all_summary = []
    all_angles = []
    all_cross = []
    for label, _start_s, _end_s in windows:
        valid = np.ones(len(words), dtype=bool)
        for subject in subject_features:
            valid &= subject_valids[subject][label]
        valid[:max_rank] = False
        word_indices = np.flatnonzero(valid)
        x_parts = [subject_features[subject][label][word_indices] for subject in subject_features]
        x = np.hstack(x_parts).astype(np.float32)
        channels = sum((channel_labels[subject] for subject in subject_features), [])
        pd.DataFrame({"channel_id": channels}).to_csv(args.output_dir / f"{label}_pseudo_population_channels.csv", index=False)
        np.savez_compressed(args.output_dir / f"{label}_pseudo_population.npz", x=x, word_indices=word_indices)
        print(f"Window {label}: X={x.shape}; ranks={ranks}", flush=True)

        summary, predictions, targets, coefs = evaluate_window(
            x=x,
            embeddings=embeddings,
            word_indices=word_indices,
            ranks=ranks,
            outer_splits=args.outer_splits,
            alpha=args.ridge_alpha,
        )
        summary.insert(0, "window", label)
        all_summary.append(summary)

        angle_rows, mean_angle = subspace_angle_tables(coefs, args.subspace_dim)
        angle_rows.insert(0, "window", label)
        all_angles.append(angle_rows)
        mean_angle.to_csv(args.output_dir / f"{label}_decoder_subspace_mean_angles.csv")
        plot_heatmap(
            mean_angle,
            args.output_dir / f"{label}_decoder_subspace_mean_angles.png",
            f"{args.roi} {label}: decoder subspace angles",
            "mean principal angle (deg)",
        )

        cross = cross_rank_generalization(predictions, targets)
        cross.insert(0, "window", label)
        all_cross.append(cross)
        cross_matrix = cross.pivot(index="train_rank", columns="target_rank", values="mean_sample_cosine")
        cross_matrix.to_csv(args.output_dir / f"{label}_cross_rank_sample_cosine.csv")
        plot_heatmap(
            cross_matrix,
            args.output_dir / f"{label}_cross_rank_sample_cosine.png",
            f"{args.roi} {label}: cross-rank generalization",
            "sample cosine",
        )

    summary_df = pd.concat(all_summary, ignore_index=True)
    angles_df = pd.concat(all_angles, ignore_index=True)
    cross_df = pd.concat(all_cross, ignore_index=True)
    summary_df.to_csv(args.output_dir / "rank_decoding_summary.csv", index=False)
    angles_df.to_csv(args.output_dir / "decoder_subspace_angles.csv", index=False)
    cross_df.to_csv(args.output_dir / "cross_rank_generalization.csv", index=False)
    plot_rank_decoding(summary_df, args.output_dir)

    print("\nRank decoding summary", flush=True)
    print(summary_df.round(4).to_string(index=False), flush=True)
    print(f"\nSaved rank-subspace outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
