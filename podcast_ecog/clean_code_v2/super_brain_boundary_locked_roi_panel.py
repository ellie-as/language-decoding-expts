#!/usr/bin/env python3
"""Panel plot of boundary-locked super-brain decoder performance by ROI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_OUTPUT_ROOT, ROI_ORDER  # noqa: E402
from super_brain_boundary_locked_performance import (  # noqa: E402
    boundary_display_name,
    boundary_file_label,
    boundary_locked_prediction,
    load_boundary_end_indices,
    n_label,
    prediction_path,
    score_axis_label,
    sem,
)

ROI_NAMES = {
    "EAC": "Early auditory cortex",
    "STG": "Superior temporal gyrus",
    "IFG": "Inferior frontal gyrus",
    "PRC": "Precentral gyrus",
    "MFG": "Middle frontal gyrus",
    "TMP": "Temporal pole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared_midpoint")
    parser.add_argument(
        "--decoder-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint" / "gpt2_ctx32_layer8_pca20" / "decoders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint" / "gpt2_ctx32_layer8_pca20" / "boundary_locked",
    )
    parser.add_argument("--target-label", default="GPT2 features")
    parser.add_argument("--rois", nargs="+", default=[roi for roi in ROI_ORDER if roi != "ALL"])
    parser.add_argument("--directions", nargs="+", choices=["past", "future"], default=["past", "future"])
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--word-window", type=int, default=10)
    parser.add_argument("--boundary-level", choices=["sentence", "event", "constituent"], default="sentence")
    parser.add_argument("--metric", choices=["point_r", "rank_identification"], default="point_r")
    return parser.parse_args()


def collect_locked(args: argparse.Namespace, rois: list[str], directions: list[str], lags: list[int]) -> pd.DataFrame:
    boundaries = load_boundary_end_indices(args.prepared_dir, args.boundary_level)
    rows = []
    for roi in rois:
        for direction in directions:
            for lag in lags:
                path = prediction_path(args.decoder_dir, roi, direction, lag)
                if not path.is_file():
                    raise FileNotFoundError(path)
                locked = boundary_locked_prediction(path, boundaries, args.word_window, args.metric)
                locked.insert(0, "metric", args.metric)
                locked.insert(0, "direction", direction)
                locked.insert(0, "lag", int(lag))
                locked.insert(0, "roi", roi)
                rows.append(locked)
    return pd.concat(rows, ignore_index=True)


def summarize_locked(locked: pd.DataFrame) -> pd.DataFrame:
    return (
        locked.groupby(["roi", "direction", "lag", "relative_word"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            sem_score=("score", sem),
            n=("score", "count"),
            n_boundaries=("boundary_i", "nunique"),
        )
        .sort_values(["roi", "direction", "lag", "relative_word"])
    )


def plot_panel(summary: pd.DataFrame, args: argparse.Namespace, rois: list[str], directions: list[str], lags: list[int]) -> None:
    palette = sns.color_palette("viridis", n_colors=len(lags))
    n_rows = len(rois)
    n_cols = len(directions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 2.45 * n_rows + 1.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    y_vals = summary["mean_score"].to_numpy(float)
    y_sem = summary["sem_score"].fillna(0.0).to_numpy(float)
    y_min = float(np.nanmin(y_vals - y_sem))
    y_max = float(np.nanmax(y_vals + y_sem))
    pad = max(0.01, (y_max - y_min) * 0.08)
    baseline = 0.5 if args.metric == "rank_identification" else 0.0

    for row_i, roi in enumerate(rois):
        for col_i, direction in enumerate(directions):
            ax = axes[row_i, col_i]
            panel = summary[(summary["roi"] == roi) & (summary["direction"] == direction)]
            for color, lag in zip(palette, lags):
                lag_df = panel[panel["lag"] == lag].sort_values("relative_word")
                ax.plot(
                    lag_df["relative_word"],
                    lag_df["mean_score"],
                    marker="o",
                    markersize=3.2,
                    lw=1.55,
                    color=color,
                )
                ax.fill_between(
                    lag_df["relative_word"],
                    lag_df["mean_score"] - lag_df["sem_score"],
                    lag_df["mean_score"] + lag_df["sem_score"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            ax.axvline(0, color="#c44e52", lw=1.0, alpha=0.9)
            ax.axhline(baseline, color="0.25", lw=0.7)
            ax.set_xlim(-args.word_window, args.word_window)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xticks([-args.word_window, -5, 0, 5, args.word_window])
            ax.grid(axis="y", color="0.88", linewidth=0.6)
            ax.tick_params(labelsize=8)

            if row_i == 0:
                label = "Past lags (t-N)" if direction == "past" else "Future lags (t+N)"
                ax.set_title(label, fontsize=12)
            if col_i == 0:
                ax.set_ylabel(f"{ROI_NAMES.get(roi, roi)}\n({roi})", fontsize=10)
            if row_i == n_rows - 1:
                ax.set_xlabel("word offset from boundary", fontsize=10)

    boundary_name = boundary_display_name(args.boundary_level)
    fig.suptitle(
        f"{args.target_label}: {boundary_name}-locked decoding by ROI\n"
        "0 = final word before boundary, +1 = first word after boundary",
        fontsize=14,
    )
    fig.supxlabel("", fontsize=1)
    fig.supylabel(f"mean {score_axis_label(args.metric)}", fontsize=11)

    norm = Normalize(vmin=min(lags), vmax=max(lags))
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.82, pad=0.012)
    cbar.set_label("N word offset")
    cbar.set_ticks(lags)

    prefix = n_label(lags)
    file_label = boundary_file_label(args.boundary_level)
    directions_label = "-".join(directions)
    stem = args.output_dir / f"super_brain_{prefix}_{file_label}_locked_roi_panel_{directions_label}_{args.metric}"
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rois = [roi for roi in ROI_ORDER if roi in set(args.rois) and roi != "ALL"]
    directions = [direction for direction in ["past", "future"] if direction in set(args.directions)]
    lags = sorted({int(lag) for lag in args.lags if int(lag) > 0})
    if not rois:
        raise ValueError("No requested non-ALL ROIs to plot")
    if not directions:
        raise ValueError("No requested directions to plot")

    locked = collect_locked(args, rois, directions, lags)
    summary = summarize_locked(locked)

    prefix = n_label(lags)
    file_label = boundary_file_label(args.boundary_level)
    directions_label = "-".join(directions)
    locked.to_csv(
        args.output_dir / f"super_brain_{prefix}_{file_label}_locked_roi_panel_word_level_{directions_label}_{args.metric}.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / f"super_brain_{prefix}_{file_label}_locked_roi_panel_summary_{directions_label}_{args.metric}.csv",
        index=False,
    )
    plot_panel(summary, args, rois, directions, lags)

    print(f"Saved ROI boundary-locked panel to: {args.output_dir}", flush=True)
    print(summary.head().round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
