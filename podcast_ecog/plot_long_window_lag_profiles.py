#!/usr/bin/env python3
"""Plot long-text-window channels and their GPT-2 lag profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.plotting import plot_markers


DEFAULT_RESULTS = Path(__file__).resolve().parent / "outputs" / "text_window_encoding"
DEFAULT_REFERENCE = (
    Path("/Volumes/ellie/language-decoding-expts")
    / "podcast_ecog"
    / "outputs_all_channels"
    / "sub-03_gpt2-xl_layer-24_encoding_results.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-window-results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference-results", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "long_window_lags")
    parser.add_argument("--long-window-min", type=int, default=100)
    parser.add_argument("--extreme-n", type=int, default=5)
    return parser.parse_args()


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    if values.shape[axis] <= 1:
        return np.zeros(values.shape[1 - axis], dtype=float)
    return np.nanstd(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_csv = args.text_window_results_dir / "channel_text_window_scores.csv"
    if not channel_csv.is_file():
        raise FileNotFoundError(channel_csv)
    if not args.reference_results.is_file():
        raise FileNotFoundError(args.reference_results)

    channels = pd.read_csv(channel_csv)
    with np.load(args.reference_results, allow_pickle=True) as ref:
        corrs = ref["corrs"].mean(axis=0)
        lags = ref["lags"]
        ref_channels = ref["channel_names"].astype(str)
        coords = ref["coords"]
    if np.nanmax(np.abs(coords)) < 1:
        coords = coords * 1000.0
    return channels, corrs, lags, ref_channels, coords


def aligned_long_channels(
    channels: pd.DataFrame,
    ref_channels: np.ndarray,
    long_window_min: int,
) -> pd.DataFrame:
    ref_lookup = {channel: i for i, channel in enumerate(ref_channels)}
    long_df = channels[channels["preferred_window_words"] >= int(long_window_min)].copy()
    long_df["ref_index"] = long_df["channel"].map(ref_lookup)
    missing = long_df[long_df["ref_index"].isna()]
    if not missing.empty:
        raise ValueError(f"Reference results missing channels: {missing['channel'].head().tolist()}")
    long_df["ref_index"] = long_df["ref_index"].astype(int)
    return long_df.sort_values("gpt2_preferred_lag_s").reset_index(drop=True)


def plot_long_window_locations(long_df: pd.DataFrame, coords: np.ndarray, output_dir: Path) -> None:
    idx = long_df["ref_index"].to_numpy(dtype=int)
    values = np.log10(long_df["preferred_window_words"].to_numpy(dtype=float))
    order = np.argsort(values)
    display = plot_markers(
        values[order],
        coords[idx][order],
        node_size=45,
        display_mode="lzr",
        node_vmin=float(values.min()),
        node_vmax=float(values.max()),
        node_cmap="viridis",
        colorbar=True,
    )
    path = output_dir / "long_window_channels_brain_log10_words.png"
    display.savefig(path, dpi=250)
    display.close()


def plot_extreme_lag_profiles(
    long_df: pd.DataFrame,
    corrs: np.ndarray,
    lags: np.ndarray,
    output_dir: Path,
    extreme_n: int,
) -> dict[str, object]:
    min_row = long_df.iloc[0]
    max_row = long_df.iloc[-1]

    n = min(int(extreme_n), len(long_df))
    neg = long_df.head(n)
    pos = long_df.tail(n)
    neg_profiles = corrs[neg["ref_index"].to_numpy(dtype=int)]
    pos_profiles = corrs[pos["ref_index"].to_numpy(dtype=int)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, row, title_color in [
        (axes[0], min_row, "#3569b7"),
        (axes[1], max_row, "#b73535"),
    ]:
        profile = corrs[int(row.ref_index)]
        ax.plot(lags, profile, color=title_color, lw=2.0)
        ax.axvline(float(row.gpt2_preferred_lag_s), color=title_color, ls="--", lw=1.5)
        ax.axvline(0, color="0.75", ls=":", lw=1)
        ax.axhline(0, color="0.85", lw=1)
        ax.set_xlabel("lag relative to word onset (s)")
        ax.set_title(
            f"{row.channel}: preferred GPT-2 lag {row.gpt2_preferred_lag_s:.3g}s\n"
            f"text window {int(row.preferred_window_words)} words, text r={row.best_corr:.3f}"
        )
    axes[0].set_ylabel("GPT-2 encoding correlation r")
    fig.savefig(output_dir / "extreme_long_window_channel_lag_profiles.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    neg_mean = neg_profiles.mean(axis=0)
    pos_mean = pos_profiles.mean(axis=0)
    neg_sem = sem(neg_profiles, axis=0)
    pos_sem = sem(pos_profiles, axis=0)
    ax.plot(lags, neg_mean, color="#3569b7", lw=2, label=f"{n} most negative-lag long-window channels")
    ax.fill_between(lags, neg_mean - neg_sem, neg_mean + neg_sem, color="#3569b7", alpha=0.18)
    ax.plot(lags, pos_mean, color="#b73535", lw=2, label=f"{n} most positive-lag long-window channels")
    ax.fill_between(lags, pos_mean - pos_sem, pos_mean + pos_sem, color="#b73535", alpha=0.18)
    ax.axvline(0, color="0.75", ls=":", lw=1)
    ax.axhline(0, color="0.85", lw=1)
    ax.set_xlabel("lag relative to word onset (s)")
    ax.set_ylabel("GPT-2 encoding correlation r")
    ax.set_title("Temporal correlation profiles for long-window channels")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "extreme_group_mean_lag_profiles.png", dpi=220)
    plt.close(fig)

    return {
        "most_negative_channel": str(min_row.channel),
        "most_negative_lag_s": float(min_row.gpt2_preferred_lag_s),
        "most_positive_channel": str(max_row.channel),
        "most_positive_lag_s": float(max_row.gpt2_preferred_lag_s),
        "extreme_n": int(n),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    channels, corrs, lags, ref_channels, coords = load_inputs(args)
    long_df = aligned_long_channels(channels, ref_channels, args.long_window_min)
    if long_df.empty:
        raise ValueError(f"No channels have preferred_window_words >= {args.long_window_min}.")

    long_df.to_csv(args.output_dir / "long_window_channels.csv", index=False)
    plot_long_window_locations(long_df, coords, args.output_dir)
    summary = plot_extreme_lag_profiles(long_df, corrs, lags, args.output_dir, args.extreme_n)

    (args.output_dir / "long_window_lag_summary.txt").write_text(
        "\n".join(
            [
                f"Long-window threshold: >= {args.long_window_min} words",
                f"Long-window channels: {len(long_df)}",
                f"Most negative-lag channel: {summary['most_negative_channel']} ({summary['most_negative_lag_s']:.5g} s)",
                f"Most positive-lag channel: {summary['most_positive_channel']} ({summary['most_positive_lag_s']:.5g} s)",
                f"Extreme group size: {summary['extreme_n']} per side",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved long-window lag plots to {args.output_dir}", flush=True)
    print((args.output_dir / "long_window_lag_summary.txt").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
