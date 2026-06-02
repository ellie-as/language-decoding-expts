#!/usr/bin/env python3
"""Check whether super-brain lag decoders predict the current word."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_OUTPUT_ROOT, boundary_prefix, crossed_between, row_corr  # noqa: E402


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
        default=DEFAULT_OUTPUT_ROOT / "super_brain_midpoint" / "gpt2_ctx32_layer8_pca20" / "current_word_confound",
    )
    parser.add_argument("--target-stem", default="gpt2_ctx32_layer8_pca20")
    parser.add_argument("--roi", default="ALL")
    parser.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--directions", nargs="+", choices=["past", "future"], default=["past", "future"])
    parser.add_argument("--boundary-level", default="sentence")
    return parser.parse_args()


def prediction_path(decoder_dir: Path, roi: str, direction: str, lag: int) -> Path:
    return decoder_dir / "predictions" / f"super_brain__{roi}__{direction}__lag-{int(lag):02d}.npz"


def load_prediction_rows(args: argparse.Namespace, word_scores: np.ndarray, boundary_after: np.ndarray) -> pd.DataFrame:
    prefix = boundary_prefix(boundary_after)
    rows = []
    directions = [str(direction) for direction in args.directions]
    for direction in directions:
        for lag in sorted({int(lag) for lag in args.lags if int(lag) > 0}):
            path = prediction_path(args.decoder_dir, args.roi, direction, lag)
            if not path.is_file():
                raise FileNotFoundError(path)
            npz = np.load(path)
            current_idx = npz["current_word_idx"].astype(np.int32)
            target_idx = npz["target_word_idx"].astype(np.int32)
            y_pred = npz["y_pred"].astype(np.float32)
            y_target = npz["y_true"].astype(np.float32)
            y_current = word_scores[current_idx].astype(np.float32)
            target_point_r = row_corr(y_target, y_pred)
            current_point_r = row_corr(y_current, y_pred)
            target_current_similarity = row_corr(y_target, y_current)
            crossed = crossed_between(prefix, current_idx, target_idx)
            valid = (
                np.isfinite(target_point_r)
                & np.isfinite(current_point_r)
                & np.isfinite(target_current_similarity)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "roi": args.roi,
                        "direction": direction,
                        "target_label": f"past target t-{lag}" if direction == "past" else f"future target t+{lag}",
                        "lag": int(lag),
                        "current_word_idx": current_idx[valid],
                        "target_word_idx": target_idx[valid],
                        f"{args.boundary_level}_boundary_crossed": crossed[valid].astype(int),
                        "condition": np.where(crossed[valid], f"across {args.boundary_level} boundary", f"within {args.boundary_level}"),
                        "target_point_r": target_point_r[valid],
                        "current_point_r": current_point_r[valid],
                        "target_minus_current_r": target_point_r[valid] - current_point_r[valid],
                        "target_gt_current": target_point_r[valid] > current_point_r[valid],
                        "target_current_similarity": target_current_similarity[valid],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def summarize(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_lag = (
        samples.groupby(["direction", "lag"], as_index=False)
        .agg(
            n=("target_point_r", "size"),
            target_point_r=("target_point_r", "mean"),
            current_point_r=("current_point_r", "mean"),
            target_minus_current_r=("target_minus_current_r", "mean"),
            frac_target_gt_current=("target_gt_current", "mean"),
            target_current_similarity=("target_current_similarity", "mean"),
        )
        .sort_values(["direction", "lag"])
    )
    by_condition = (
        samples.groupby(["direction", "lag", "condition"], as_index=False)
        .agg(
            n=("target_point_r", "size"),
            target_point_r=("target_point_r", "mean"),
            current_point_r=("current_point_r", "mean"),
            target_minus_current_r=("target_minus_current_r", "mean"),
            frac_target_gt_current=("target_gt_current", "mean"),
            target_current_similarity=("target_current_similarity", "mean"),
        )
        .sort_values(["direction", "lag", "condition"])
    )
    return by_lag, by_condition


def plot_target_vs_current(by_lag: pd.DataFrame, output_dir: Path, target_stem: str, roi: str) -> None:
    long = by_lag.melt(
        id_vars=["direction", "lag"],
        value_vars=["target_point_r", "current_point_r"],
        var_name="comparison",
        value_name="mean_point_r",
    )
    long["direction_label"] = long["direction"].map({"past": "past target t-N", "future": "future target t+N"})
    long["comparison_label"] = long["comparison"].map(
        {
            "target_point_r": "prediction vs target",
            "current_point_r": "prediction vs current t",
        }
    )
    long["line_label"] = long["direction_label"] + ": " + long["comparison_label"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.lineplot(data=long, x="lag", y="mean_point_r", hue="line_label", marker="o", ax=ax)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(sorted(by_lag["lag"].unique()))
    ax.set_xlabel("N word offset")
    ax.set_ylabel("mean pointwise r")
    ax.set_title(f"{target_stem} {roi}: target word vs current word")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    fig.savefig(output_dir / "super_brain_prediction_target_vs_current_word.png", dpi=240)
    fig.savefig(output_dir / "super_brain_prediction_target_vs_current_word.pdf")
    plt.close(fig)

    by_lag = by_lag.copy()
    by_lag["direction_label"] = by_lag["direction"].map({"past": "past target t-N", "future": "future target t+N"})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=by_lag,
        x="lag",
        y="target_minus_current_r",
        hue="direction_label",
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(sorted(by_lag["lag"].unique()))
    ax.set_xlabel("N word offset")
    ax.set_ylabel("mean r(pred,target) - r(pred,current)")
    ax.set_title("Positive values mean prediction is closer to the target word than to t")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    fig.savefig(output_dir / "super_brain_prediction_target_minus_current_word.png", dpi=240)
    fig.savefig(output_dir / "super_brain_prediction_target_minus_current_word.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    word_scores = np.load(args.prepared_dir / f"{args.target_stem}.npy").astype(np.float32)
    words = pd.read_csv(args.prepared_dir / "words.csv")
    boundary_col = f"{args.boundary_level}_boundary_after"
    if boundary_col not in words.columns:
        raise ValueError(f"{args.prepared_dir / 'words.csv'} missing {boundary_col!r}")
    samples = load_prediction_rows(args, word_scores, words[boundary_col].astype(bool).to_numpy())
    by_lag, by_condition = summarize(samples)
    samples.to_csv(args.output_dir / "super_brain_prediction_target_vs_current_samples.csv", index=False)
    by_lag.to_csv(args.output_dir / "super_brain_prediction_target_vs_current_by_lag.csv", index=False)
    by_condition.to_csv(args.output_dir / "super_brain_prediction_target_vs_current_by_condition.csv", index=False)
    plot_target_vs_current(by_lag, args.output_dir, args.target_stem, args.roi)
    print("\nBy lag", flush=True)
    print(by_lag.round(4).to_string(index=False), flush=True)
    print(f"\nSaved current-word confound outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
