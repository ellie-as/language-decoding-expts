#!/usr/bin/env python3
"""Plot the distribution of per-channel preferred lags within each paper ROI.

Reads CSVs from ``preferred_lag_analysis.py`` (word-position lags) or
``preferred_lag_time_analysis.py`` (peri-word time in seconds) and produces,
per target:

  - a grid of per-ROI histograms (proportion of channels)
  - a single ROI x lag heatmap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_OUTPUT_ROOT, ROI_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "preferred_lag")
    parser.add_argument("--rois", nargs="+", default=[r for r in ROI_ORDER if r != "ALL"])
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--significant-only", action="store_true")
    parser.add_argument("--min-abs-corr", type=float, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--lag-column",
        choices=["auto", "preferred_lag_words", "preferred_lag_sec"],
        default="auto",
        help="Which preferred-lag column to plot. auto detects from the CSV header.",
    )
    return parser.parse_args()


def detect_lag_column(df: pd.DataFrame, requested: str) -> tuple[str, str]:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested lag column {requested!r} not in CSV")
        unit = "words" if requested == "preferred_lag_words" else "seconds"
        return requested, unit
    if "preferred_lag_words" in df.columns:
        return "preferred_lag_words", "words"
    if "preferred_lag_sec" in df.columns:
        return "preferred_lag_sec", "seconds"
    raise ValueError("CSV has neither preferred_lag_words nor preferred_lag_sec")


def infer_time_bins(df: pd.DataFrame, lag_column: str, csv_path: Path) -> np.ndarray:
    summary_candidates = [
        csv_path.parent / csv_path.name.replace("preferred_lag_time__", "preferred_lag_time_summary__").replace(".csv", ".json"),
        csv_path.parent / csv_path.name.replace("preferred_lag__", "preferred_lag_time_summary__").replace(".csv", ".json"),
    ]
    for summary_path in summary_candidates:
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                times = json.load(handle).get("times_sec")
            if times:
                return np.asarray(times, dtype=float)
    step = 1.0 / 32.0
    lo = float(np.floor(df[lag_column].min() / step) * step)
    hi = float(np.ceil(df[lag_column].max() / step) * step)
    return np.arange(lo, hi + step * 0.5, step)


def ordered_rois(df: pd.DataFrame, rois: list[str], include_other: bool) -> list[str]:
    present = set(df["paper_roi"].astype(str).unique())
    out = [roi for roi in rois if roi in present]
    if include_other and "Other" in present and "Other" not in out:
        out.append("Other")
    return out


def lag_grid(df: pd.DataFrame, lag_column: str, unit: str, csv_path: Path) -> np.ndarray:
    if unit == "words":
        lags = sorted(int(v) for v in df[lag_column].dropna().unique())
        return np.arange(min(lags), max(lags) + 1, dtype=int)
    return infer_time_bins(df, lag_column, csv_path)


def assign_bins(values: np.ndarray, bins: np.ndarray, unit: str) -> np.ndarray:
    if unit == "words":
        return values.astype(int)
    idx = np.argmin(np.abs(values[:, None] - bins[None, :]), axis=1)
    return bins[idx]


def proportions(values: np.ndarray, bins: np.ndarray, unit: str) -> np.ndarray:
    if unit == "words":
        counts = np.array([(values == lag).sum() for lag in bins], dtype=float)
    else:
        assigned = assign_bins(values.astype(float), bins, unit)
        counts = np.array([(assigned == lag).sum() for lag in bins], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def plot_histograms(
    df: pd.DataFrame,
    rois: list[str],
    bins: np.ndarray,
    lag_column: str,
    unit: str,
    out_path: Path,
    title: str,
) -> None:
    n = len(rois)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), sharex=True, sharey=True, squeeze=False)
    xlabel = "preferred lag (words)" if unit == "words" else "preferred lag (s relative to anchor)"
    for ax, roi in zip(axes.flat, rois):
        values = df.loc[df["paper_roi"] == roi, lag_column].to_numpy(dtype=float)
        props = proportions(values, bins, unit)
        mean_lag = float(np.nanmean(values)) if len(values) else float("nan")
        if unit == "words":
            ax.bar(bins, props, width=0.85, color="#3a7ca5", edgecolor="white")
            ax.set_xticks(bins)
        else:
            widths = np.diff(bins, append=bins[-1] + (bins[-1] - bins[-2] if len(bins) > 1 else 0.03125))
            ax.bar(bins, props, width=widths * 0.9, align="edge", color="#3a7ca5", edgecolor="white")
        ax.axvline(0, color=(0.8, 0.8, 0.8), ls="--", zorder=0)
        if np.isfinite(mean_lag):
            ax.axvline(mean_lag, color="#d1495b", lw=2, label=f"mean={mean_lag:.2f}")
            ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title(f"{roi} (n={len(values)})", fontsize=10)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("proportion of channels")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    rois: list[str],
    bins: np.ndarray,
    lag_column: str,
    unit: str,
    out_path: Path,
    title: str,
) -> None:
    matrix = np.vstack([proportions(df.loc[df["paper_roi"] == roi, lag_column].to_numpy(dtype=float), bins, unit) for roi in rois])
    counts = [int((df["paper_roi"] == roi).sum()) for roi in rois]
    fig, ax = plt.subplots(figsize=(0.6 * len(bins) + 2.5, 0.5 * len(rois) + 1.8), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0)
    if unit == "words":
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels([f"{lag:+d}" for lag in bins])
        xlabel = "preferred lag (words)"
    else:
        tick_idx = np.linspace(0, len(bins) - 1, num=min(9, len(bins)), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{bins[i]:+.2f}" for i in tick_idx], rotation=45, ha="right")
        xlabel = "preferred lag (s relative to anchor)"
    ax.set_yticks(np.arange(len(rois)))
    ax.set_yticklabels([f"{roi} (n={c})" for roi, c in zip(rois, counts)])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="proportion of channels")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def stem_from_csv(csv_path: Path) -> str:
    name = csv_path.stem
    for prefix in ("preferred_lag_time__", "preferred_lag__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        if args.significant_only:
            if "significant_best" not in df.columns:
                print(f"{csv_path}: no significant_best column, skipping", flush=True)
                continue
            df = df[df["significant_best"]]
        if args.min_abs_corr is not None and "max_abs_corr" in df.columns:
            df = df[df["max_abs_corr"] >= float(args.min_abs_corr)]
        if df.empty:
            print(f"{csv_path}: no channels left after filtering, skipping", flush=True)
            continue
        lag_column, unit = detect_lag_column(df, args.lag_column)
        rois = ordered_rois(df, args.rois, args.include_other)
        if not rois:
            print(f"{csv_path}: no requested ROIs present, skipping", flush=True)
            continue
        bins = lag_grid(df, lag_column, unit, csv_path)
        target = stem_from_csv(csv_path)
        hist_path = args.output_dir / f"preferred_lag_dist_by_roi__{target}{args.suffix}.png"
        heat_path = args.output_dir / f"preferred_lag_dist_heatmap__{target}{args.suffix}.png"
        kind = "word-position" if unit == "words" else "time"
        title = f"Preferred-lag distribution by ROI ({kind}) - {target}"
        plot_histograms(df, rois, bins, lag_column, unit, hist_path, title)
        plot_heatmap(df, rois, bins, lag_column, unit, heat_path, title)
        print(f"Saved: {hist_path}", flush=True)
        print(f"Saved: {heat_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
