#!/usr/bin/env python3
"""Boundary-locked super-brain decoder performance by word offset."""

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

from common import DEFAULT_OUTPUT_ROOT, word_metric_scores_from_npz  # noqa: E402


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
    parser.add_argument("--roi", default="ALL")
    parser.add_argument("--direction", choices=["past", "future"], default="past")
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--word-window", type=int, default=10)
    parser.add_argument("--boundary-level", choices=["sentence", "event", "constituent"], default="sentence")
    parser.add_argument("--metric", choices=["point_r", "rank_identification"], default="point_r")
    return parser.parse_args()


def prediction_path(decoder_dir: Path, roi: str, direction: str, lag: int) -> Path:
    return decoder_dir / "predictions" / f"super_brain__{roi}__{direction}__lag-{int(lag):02d}.npz"


def sem(values: pd.Series) -> float:
    n = int(values.notna().sum())
    return float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan


def n_label(lags: list[int]) -> str:
    return f"n{lags[0]}" if len(lags) == 1 else f"n{min(lags)}-{max(lags)}"


def boundary_file_label(level: str) -> str:
    return f"{level}_boundary"


def boundary_display_name(level: str) -> str:
    return f"{level} boundary"


def load_boundary_end_indices(prepared_dir: Path, level: str) -> np.ndarray:
    boundaries_path = prepared_dir / "boundaries.csv"
    if boundaries_path.is_file():
        boundaries = pd.read_csv(boundaries_path)
        if {"level", "end_word_idx"}.issubset(boundaries.columns):
            vals = boundaries.loc[boundaries["level"].astype(str).eq(level), "end_word_idx"]
            if len(vals):
                return np.asarray(sorted(set(vals.astype(int).tolist())), dtype=int)

    words = pd.read_csv(prepared_dir / "words.csv")
    col = f"{level}_boundary_after"
    if col not in words.columns:
        raise ValueError(f"{prepared_dir / 'words.csv'} missing {col!r}")
    return np.flatnonzero(words[col].astype(bool).to_numpy())


def boundary_locked_prediction(npz_path: Path, boundaries: np.ndarray, word_window: int, metric: str) -> pd.DataFrame:
    npz = np.load(npz_path)
    current = npz["current_word_idx"].astype(int)
    scores = word_metric_scores_from_npz(npz, metric)
    perf = dict(zip(current.tolist(), scores.tolist()))

    rows = []
    for boundary_i, boundary_word_idx in enumerate(boundaries):
        for rel_word in range(-word_window, word_window + 1):
            current_word_idx = int(boundary_word_idx) + rel_word
            score = perf.get(current_word_idx)
            if score is None or not np.isfinite(score):
                continue
            rows.append(
                {
                    "boundary_i": int(boundary_i),
                    "boundary_word_idx": int(boundary_word_idx),
                    "relative_word": int(rel_word),
                    "word_idx": int(current_word_idx),
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def set_relative_word_ticks(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
    unique = np.asarray(sorted(set(int(x) for x in values)))
    if len(unique) <= 25:
        ax.set_xticks(unique)
        return
    max_abs = int(np.nanmax(np.abs(unique)))
    step = 10 if max_abs <= 50 else 20 if max_abs <= 100 else 50
    ticks = np.arange(-max_abs, max_abs + 1, step)
    if 0 not in ticks:
        ticks = np.sort(np.r_[ticks, 0])
    ax.set_xticks(ticks)


def score_axis_label(metric: str) -> str:
    if metric == "rank_identification":
        return "rank-identification accuracy"
    return "per-word true-vs-pred r"


def plot_lag_lines(summary: pd.DataFrame, args: argparse.Namespace, lags: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    palette = sns.color_palette("viridis", n_colors=len(lags))
    for color, lag in zip(palette, lags):
        lag_df = summary[summary["lag"] == lag].sort_values("relative_word")
        ax.plot(lag_df["relative_word"], lag_df["mean_score"], marker="o", lw=2.0, color=color, label=str(lag))
        ax.fill_between(
            lag_df["relative_word"],
            lag_df["mean_score"] - lag_df["sem_score"],
            lag_df["mean_score"] + lag_df["sem_score"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    ax.axvline(0, color="#c44e52", lw=1.2, alpha=0.9)
    baseline = 0.5 if args.metric == "rank_identification" else 0.0
    ax.axhline(baseline, color="0.25", lw=0.8)
    set_relative_word_ticks(ax, summary["relative_word"])
    label = boundary_display_name(args.boundary_level)
    ax.set_xlabel(f"word offset from {label}\n0 = final word before boundary, +1 = first word after boundary")
    ax.set_ylabel(f"mean {score_axis_label(args.metric)}")
    target = f"t-{min(lags)}..t-{max(lags)}" if args.direction == "past" else f"t+{min(lags)}..t+{max(lags)}"
    ax.set_title(f"{args.target_label}: {label}-locked decoding, {target}")

    norm = Normalize(vmin=min(lags), vmax=max(lags))
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location="right", shrink=0.78, pad=0.015)
    cbar.set_label("N word offset")
    cbar.set_ticks(lags)

    prefix = n_label(lags)
    file_label = boundary_file_label(args.boundary_level)
    suffix = f"{args.direction}_{args.metric}"
    out_stem = args.output_dir / f"super_brain_{prefix}_lines_{file_label}_locked_word_offsets_{suffix}"
    fig.savefig(out_stem.with_suffix(".png"), dpi=240)
    fig.savefig(out_stem.with_suffix(".pdf"))
    if args.boundary_level == "sentence" and args.direction == "past" and args.metric == "point_r":
        alias = args.output_dir / f"group_average_{prefix}_lines_boundary_locked_word_offsets.png"
        fig.savefig(alias, dpi=240)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lags = sorted({int(lag) for lag in args.lags if int(lag) > 0})
    boundaries = load_boundary_end_indices(args.prepared_dir, args.boundary_level)

    all_rows = []
    for lag in lags:
        path = prediction_path(args.decoder_dir, args.roi, args.direction, lag)
        if not path.is_file():
            raise FileNotFoundError(path)
        locked = boundary_locked_prediction(path, boundaries, args.word_window, args.metric)
        locked.insert(0, "metric", args.metric)
        locked.insert(0, "direction", args.direction)
        locked.insert(0, "lag", int(lag))
        locked.insert(0, "roi", args.roi)
        all_rows.append(locked)

    locked = pd.concat(all_rows, ignore_index=True)
    prefix = n_label(lags)
    file_label = boundary_file_label(args.boundary_level)
    suffix = f"{args.direction}_{args.metric}"
    locked.to_csv(args.output_dir / f"super_brain_{prefix}_{file_label}_locked_word_level_performance_{suffix}.csv", index=False)

    summary = (
        locked.groupby(["lag", "relative_word"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            sem_score=("score", sem),
            n=("score", "count"),
            n_boundaries=("boundary_i", "nunique"),
        )
        .sort_values(["lag", "relative_word"])
    )
    summary.to_csv(args.output_dir / f"super_brain_{prefix}_by_lag_{file_label}_locked_summary_{suffix}.csv", index=False)
    plot_lag_lines(summary, args, lags)

    print(f"Saved super-brain boundary-locked plot and tables to: {args.output_dir}", flush=True)
    print(summary.head().round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
