#!/usr/bin/env python3
"""Plot top sentence-boundary high-gamma timecourses by response direction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PODCAST_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        default=PODCAST_DIR
        / "outputs"
        / "boundary_channel_tracking"
        / "sentence"
        / "sentence_boundary_channel_tracking.csv",
    )
    parser.add_argument(
        "--timecourses-npz",
        type=Path,
        default=PODCAST_DIR
        / "outputs"
        / "boundary_channel_tracking"
        / "sentence"
        / "sentence_boundary_channel_timecourses.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PODCAST_DIR / "outputs" / "clean_code_v2" / "sentence_boundary_channel_timecourses",
    )
    parser.add_argument("--direction", choices=["increase", "decrease"], default="decrease")
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--n-cols", type=int, default=4)
    parser.add_argument("--boundary-anchor", choices=["next_start", "prev_end", "midpoint"], default="next_start")
    parser.add_argument("--baseline-window", nargs=2, type=float, default=[-0.6, -0.1])
    parser.add_argument("--response-window", nargs=2, type=float, default=[0.0, 0.6])
    return parser.parse_args()


def ranked_channels(tracking: pd.DataFrame, direction: str, q_threshold: float, top_n: int) -> pd.DataFrame:
    sig = tracking["boundary_q"].to_numpy(dtype=float) <= float(q_threshold)
    effect = tracking["boundary_effect"].to_numpy(dtype=float)
    if direction == "increase":
        ranked = tracking[sig & (effect > 0)].sort_values("boundary_effect", ascending=False).head(top_n).copy()
    else:
        ranked = tracking[sig & (effect < 0)].sort_values("boundary_effect", ascending=True).head(top_n).copy()
    ranked["rank_within_direction"] = np.arange(1, len(ranked) + 1)
    return ranked


def load_timecourse_lookup(npz_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    times = data["time"].astype(float)
    traces = data["mean_highgamma_z"].astype(float)
    channel_ids = data["channel_id"].astype(str)
    if len(channel_ids) != traces.shape[0]:
        raise ValueError(f"{npz_path} has {len(channel_ids)} channel ids but {traces.shape[0]} traces")
    return times, {channel_id: traces[i] for i, channel_id in enumerate(channel_ids)}


def plot_direction_panel(
    ranked: pd.DataFrame,
    times: np.ndarray,
    trace_lookup: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Path:
    n = len(ranked)
    if n == 0:
        raise ValueError(f"No significant {args.direction} channels at q <= {args.q_threshold:g}")
    ncols = int(args.n_cols)
    nrows = int(math.ceil(n / ncols))
    color = "#B2182B" if args.direction == "increase" else "#2166AC"
    title_word = "increased" if args.direction == "increase" else "decreased"

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for ax, (_, row) in zip(axes.flat, ranked.iterrows(), strict=False):
        channel_id = str(row["channel_id"])
        if channel_id not in trace_lookup:
            raise KeyError(f"{channel_id} not found in {args.timecourses_npz}")
        ax.plot(times, trace_lookup[channel_id], color=color, lw=1.5)
        ax.axvline(0, color="0.15", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvspan(args.baseline_window[0], args.baseline_window[1], color="0.85", alpha=0.45, lw=0)
        ax.axvspan(args.response_window[0], args.response_window[1], color="#F58518", alpha=0.18, lw=0)
        ax.set_title(
            f"#{int(row['rank_within_direction'])} {row['subject']}:{row['channel']} {row['paper_roi']}\n"
            f"effect={row['boundary_effect']:.3f}, q={row['boundary_q']:.3g}",
            fontsize=9,
        )
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.supxlabel(f"time from sentence boundary ({args.boundary_anchor}, s)")
    fig.supylabel("mean high-gamma z, baseline subtracted")
    fig.suptitle(
        f"Top {n} channels that {title_word} most at sentence boundaries",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()

    out_stem = args.output_dir / f"top{n}_sentence_boundary_{args.direction}_channel_timecourses"
    fig.savefig(out_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_stem.with_suffix(".png")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"config_{args.direction}.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    tracking = pd.read_csv(args.tracking_csv)
    ranked = ranked_channels(tracking, args.direction, args.q_threshold, int(args.top_n))
    ranked.to_csv(args.output_dir / f"top{len(ranked)}_sentence_boundary_{args.direction}_channels.csv", index=False)

    times, trace_lookup = load_timecourse_lookup(args.timecourses_npz)
    out_png = plot_direction_panel(ranked, times, trace_lookup, args)

    print(f"Saved {args.direction} panel to: {out_png}", flush=True)
    print(
        ranked[["rank_within_direction", "channel_id", "paper_roi", "boundary_effect", "boundary_q"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
