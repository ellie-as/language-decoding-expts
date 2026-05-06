#!/usr/bin/env python3
"""Run ridge text-window preference analysis for all Podcast ECoG subjects."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_text_window_encoding import (  # noqa: E402
    DEFAULT_MOUNT_ROOT,
    channel_family,
    channel_hemisphere,
    channel_prefix,
    corr_per_target,
    fit_window_encoding,
    load_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "all_subject_window_preferences")
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=THIS_DIR / "outputs" / "text_window_encoding" / "cache",
    )
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100, 200, 500])
    parser.add_argument("--gpt2-layer", type=int, default=24)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--alpha-tune-targets", type=int, default=256)
    parser.add_argument("--alpha-tune-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bids_root(args: argparse.Namespace) -> Path:
    return args.bids_root or args.mounted_root / "podcast_ecog" / "data" / "ds005574"


def load_words_and_gpt2_features(root: Path, layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    transcript_path = root / "stimuli" / "gpt2-xl" / "transcript.tsv"
    features_path = root / "stimuli" / "gpt2-xl" / "features.hdf5"
    transcript = pd.read_csv(transcript_path, sep="\t", index_col=0)
    with h5py.File(features_path, "r") as handle:
        token_embeddings = handle[f"layer-{layer}"][...].astype(np.float32)

    rows = []
    features = []
    for word_idx, group in transcript.groupby("word_idx", sort=True):
        rows.append(
            {
                "word_idx": int(word_idx),
                "word": str(group["word"].iloc[0]),
                "start": float(group["start"].iloc[0]),
                "end": float(group["end"].iloc[-1]),
            }
        )
        features.append(token_embeddings[group.index.to_numpy()].mean(axis=0))
    words = pd.DataFrame(rows)
    features = np.vstack(features).astype(np.float32)
    order = np.argsort(words["start"].to_numpy())
    return words.iloc[order].reset_index(drop=True), features[order]


def fif_path(root: Path, subject: str, task: str) -> Path:
    return root / "derivatives" / "ecogprep" / f"sub-{subject}" / "ieeg" / (
        f"sub-{subject}_task-{task}_desc-highgamma_ieeg.fif"
    )


def find_embedding_cache(cache_dir: Path, window_size: int) -> Path:
    matches = sorted(cache_dir.glob(f"text_window_w{window_size}__*.npz"))
    if not matches:
        raise FileNotFoundError(f"No cached text-window embeddings for window {window_size} under {cache_dir}")
    return matches[-1]


def load_text_embeddings(cache_dir: Path, window_size: int, selection: np.ndarray) -> np.ndarray:
    path = find_embedding_cache(cache_dir, window_size)
    embeddings = np.load(path)["embeddings"].astype(np.float32)
    return embeddings[selection]


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def ridge_predict_svd(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x_train, full_matrices=False)
    scale = (s / (s * s + float(alpha))).astype(np.float32, copy=False)
    coef = vt.T @ (scale[:, None] * (u.T @ y_train))
    return (x_test @ coef).astype(np.float32)


def compute_gpt2_lag_reference(
    *,
    x: np.ndarray,
    epoch_data: np.ndarray,
    alpha: float,
    outer_splits: int,
) -> np.ndarray:
    y = epoch_data.reshape(len(epoch_data), -1).astype(np.float32)
    fold_corrs = []
    splits = np.array_split(np.arange(len(x)), outer_splits)
    for fold, test_idx in enumerate(splits, start=1):
        train_idx = np.setdiff1d(np.arange(len(x)), test_idx, assume_unique=True)
        print(f"    GPT-2 reference fold {fold}/{outer_splits}", flush=True)
        x_train, x_test = standardize(x[train_idx], x[test_idx])
        y_train, y_test = standardize(y[train_idx], y[test_idx])
        pred = ridge_predict_svd(x_train, y_train, x_test, alpha)
        fold_corrs.append(corr_per_target(y_test, pred).reshape(epoch_data.shape[1:]))
    return np.stack(fold_corrs).astype(np.float32)


def build_channel_rows(
    *,
    subject: str,
    window_sizes: np.ndarray,
    window_corrs: np.ndarray,
    channel_names: np.ndarray,
    preferred_lag_s: np.ndarray,
) -> pd.DataFrame:
    mean_corr = window_corrs.mean(axis=1)
    best_idx = mean_corr.argmax(axis=0)
    best_window = window_sizes[best_idx]
    best_corr = mean_corr[best_idx, np.arange(mean_corr.shape[1])]
    rows = []
    for ch_i, channel in enumerate(channel_names):
        prefix = channel_prefix(str(channel))
        row = {
            "subject": f"sub-{subject}",
            "channel": str(channel),
            "prefix": prefix,
            "family": channel_family(prefix),
            "hemisphere": channel_hemisphere(prefix),
            "preferred_window_words": int(best_window[ch_i]),
            "best_corr": float(best_corr[ch_i]),
            "gpt2_preferred_lag_s": float(preferred_lag_s[ch_i]),
        }
        for w_i, window in enumerate(window_sizes):
            row[f"corr_w{int(window)}"] = float(mean_corr[w_i, ch_i])
        rows.append(row)
    return pd.DataFrame(rows)


def group_summary(channel_df: pd.DataFrame, window_sizes: np.ndarray) -> pd.DataFrame:
    rows = []
    for keys in [["subject"], ["subject", "family"], ["family"], ["subject", "hemisphere"], ["hemisphere"]]:
        for values, group_df in channel_df.groupby(keys, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            row = {key: value for key, value in zip(keys, values)}
            row["group_type"] = "+".join(keys)
            row["n_channels"] = int(len(group_df))
            row["mean_best_corr"] = float(group_df["best_corr"].mean())
            row["median_preferred_window_words"] = float(group_df["preferred_window_words"].median())
            row["short_window_share_le10"] = float((group_df["preferred_window_words"] <= 10).mean())
            row["long_window_share_ge100"] = float((group_df["preferred_window_words"] >= 100).mean())
            for window in window_sizes:
                row[f"mean_corr_w{int(window)}"] = float(group_df[f"corr_w{int(window)}"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_aggregate(channel_df: pd.DataFrame, window_sizes: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for subject, subject_df in channel_df.groupby("subject", sort=True):
        vals = [subject_df[f"corr_w{int(window)}"].mean() for window in window_sizes]
        ax.plot(window_sizes, vals, marker="o", alpha=0.55, lw=1.2, label=subject)
    vals = [channel_df[f"corr_w{int(window)}"].mean() for window in window_sizes]
    ax.plot(window_sizes, vals, marker="o", color="black", lw=2.5, label="all channels")
    ax.set_xscale("log")
    ax.set_xlabel("trailing text window (words)")
    ax.set_ylabel("mean encoding r")
    ax.set_title("Text-window encoding across Podcast ECoG subjects")
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(output_dir / "all_subject_window_curves.png", dpi=220)
    plt.close(fig)

    counts = (
        channel_df["preferred_window_words"]
        .value_counts()
        .reindex(window_sizes, fill_value=0)
        .rename_axis("window_words")
        .reset_index(name="n_channels")
    )
    counts["share_channels"] = counts["n_channels"] / len(channel_df)
    counts.to_csv(output_dir / "all_subject_preferred_window_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.bar(counts["window_words"].astype(str), counts["n_channels"], color="#4C78A8")
    ax.set_xlabel("preferred trailing window (words)")
    ax.set_ylabel("channels")
    ax.set_title("Preferred window across all subjects")
    fig.savefig(output_dir / "all_subject_preferred_window_counts.png", dpi=220)
    plt.close(fig)

    family = channel_df.groupby("family")
    family_rows = []
    for fam, fam_df in family:
        row = {"family": fam, "n_channels": len(fam_df)}
        for window in window_sizes:
            row[str(int(window))] = float(fam_df[f"corr_w{int(window)}"].mean())
        family_rows.append(row)
    fam_table = pd.DataFrame(family_rows).sort_values("n_channels", ascending=False)
    fam_table.to_csv(output_dir / "all_subject_family_window_means.csv", index=False)
    mat = fam_table[[str(int(w)) for w in window_sizes]].to_numpy()
    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.35 * len(fam_table) + 1.2)), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(window_sizes)))
    ax.set_xticklabels([str(int(w)) for w in window_sizes])
    ax.set_yticks(np.arange(len(fam_table)))
    ax.set_yticklabels([f"{r.family} (n={int(r.n_channels)})" for r in fam_table.itertuples()])
    ax.set_xlabel("trailing text window (words)")
    ax.set_title("Mean r by electrode family, all subjects")
    fig.colorbar(im, ax=ax, label="mean r")
    fig.savefig(output_dir / "all_subject_family_window_heatmap.png", dpi=220)
    plt.close(fig)


def write_markdown(channel_df: pd.DataFrame, group_df: pd.DataFrame, window_sizes: np.ndarray, output_dir: Path) -> None:
    overall = []
    for window in window_sizes:
        vals = channel_df[f"corr_w{int(window)}"]
        overall.append((int(window), float(vals.mean()), float(vals.median())))
    best_window, best_mean, best_median = max(overall, key=lambda item: item[1])
    long_share = float((channel_df["preferred_window_words"] >= 100).mean())
    short_share = float((channel_df["preferred_window_words"] <= 10).mean())
    subject_rows = group_df[group_df["group_type"] == "subject"].copy()
    lines = [
        "# All-Subject Text-Window Preference Summary",
        "",
        f"- Subjects: {channel_df['subject'].nunique()}",
        f"- Channels: {len(channel_df)}",
        f"- Best overall mean window: {best_window} words (mean r={best_mean:.4f}, median r={best_median:.4f}).",
        f"- Short-window preference (<=10 words): {short_share:.1%}.",
        f"- Long-window preference (>=100 words): {long_share:.1%}.",
        "",
        "## Mean r by Window",
        "",
    ]
    for window, mean_r, median_r in overall:
        lines.append(f"- {window:>3} words: mean r={mean_r:.4f}, median r={median_r:.4f}")
    lines.extend(["", "## Subject Summary", ""])
    for row in subject_rows.sort_values("subject").itertuples(index=False):
        lines.append(
            f"- {row.subject}: n={int(row.n_channels)}, "
            f"short={row.short_window_share_le10:.1%}, long={row.long_window_share_ge100:.1%}, "
            f"mean best r={row.mean_best_corr:.4f}"
        )
    (output_dir / "all_subject_interpretation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_subject(args: argparse.Namespace, words: pd.DataFrame, gpt2_features: np.ndarray, subject: str) -> pd.DataFrame:
    subject_dir = args.output_dir / f"sub-{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    channel_csv = subject_dir / "channel_text_window_scores.csv"
    if channel_csv.is_file() and not args.force:
        print(f"Skipping sub-{subject}; found {channel_csv}", flush=True)
        return pd.read_csv(channel_csv)

    root = bids_root(args)
    epochs = load_epochs(
        fif_path=fif_path(root, subject, args.task),
        words=words,
        picks_regex=args.picks_regex,
        max_channels=None,
        tmin=-2.0,
        tmax=2.0,
        resample_sfreq=args.resample_sfreq,
    )
    selection = epochs.selection
    epoch_data = epochs.get_data(copy=True).astype(np.float32)
    x_gpt2 = gpt2_features[selection].astype(np.float32)

    ref_path = subject_dir / f"sub-{subject}_gpt2_ridge_lag_reference.npz"
    if ref_path.is_file() and not args.force:
        ref = np.load(ref_path)
        gpt2_corrs = ref["corrs"]
    else:
        gpt2_corrs = compute_gpt2_lag_reference(
            x=x_gpt2,
            epoch_data=epoch_data,
            alpha=args.ridge_alpha,
            outer_splits=args.outer_splits,
        )
        np.savez_compressed(
            ref_path,
            corrs=gpt2_corrs,
            lags=epochs.times.copy(),
            channel_names=np.asarray(epochs.info["ch_names"]).astype(str),
        )
        print(f"    Saved {ref_path}", flush=True)

    mean_gpt2 = gpt2_corrs.mean(axis=0)
    preferred_idx = mean_gpt2.argmax(axis=-1)
    preferred_lag_s = epochs.times[preferred_idx].astype(np.float32)
    targets = epoch_data[:, np.arange(epoch_data.shape[1]), preferred_idx].astype(np.float32)

    window_sizes = np.asarray(args.window_sizes, dtype=int)
    window_corrs = []
    for window in window_sizes:
        print(f"    sub-{subject}: ridge text window {int(window)}", flush=True)
        x_text = load_text_embeddings(args.embedding_cache_dir, int(window), selection)
        corrs, _alphas = fit_window_encoding(
            x_text,
            targets,
            outer_splits=args.outer_splits,
            alphas=[args.ridge_alpha],
            tune_frac=args.alpha_tune_frac,
            tune_targets=args.alpha_tune_targets,
            dtype_name="float32",
            seed=args.seed + int(subject) + int(window),
        )
        window_corrs.append(corrs)
    window_corrs_arr = np.stack(window_corrs)
    channel_names = np.asarray(epochs.info["ch_names"]).astype(str)
    channel_df = build_channel_rows(
        subject=subject,
        window_sizes=window_sizes,
        window_corrs=window_corrs_arr,
        channel_names=channel_names,
        preferred_lag_s=preferred_lag_s,
    )
    channel_df.to_csv(channel_csv, index=False)
    np.savez_compressed(
        subject_dir / "text_window_ridge_results.npz",
        corrs=window_corrs_arr,
        window_sizes=window_sizes,
        channel_names=channel_names,
        preferred_lag_s=preferred_lag_s,
        gpt2_reference=ref_path.as_posix(),
    )
    return channel_df


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = bids_root(args)
    words, gpt2_features = load_words_and_gpt2_features(root, args.gpt2_layer)
    (args.output_dir / "config.json").write_text(
        json.dumps({**vars(args), "mounted_root": str(args.mounted_root), "bids_root": str(root)}, default=str, indent=2),
        encoding="utf-8",
    )

    all_channels = []
    for subject in args.subjects:
        t0 = time.time()
        print(f"=== sub-{subject} ===", flush=True)
        all_channels.append(run_subject(args, words, gpt2_features, subject))
        print(f"=== sub-{subject} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    channel_df = pd.concat(all_channels, ignore_index=True)
    channel_df.to_csv(args.output_dir / "all_subject_channel_text_window_scores.csv", index=False)
    group_df = group_summary(channel_df, np.asarray(args.window_sizes, dtype=int))
    group_df.to_csv(args.output_dir / "all_subject_group_summary.csv", index=False)
    plot_aggregate(channel_df, np.asarray(args.window_sizes, dtype=int), args.output_dir)
    write_markdown(channel_df, group_df, np.asarray(args.window_sizes, dtype=int), args.output_dir)
    print((args.output_dir / "all_subject_interpretation_summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
