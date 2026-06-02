#!/usr/bin/env python3
"""Screen ECoG channels for event-locked sentence-boundary responses.

For each channel, this script computes a boundary response:

    mean high-gamma in response window - mean high-gamma in baseline window

around sentence boundaries. Significance is estimated by circularly shifting
the sentence-boundary word indices along the transcript, preserving the number
and spacing of the boundary sequence while breaking its alignment with the
neural data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DEFAULT_BIDS_ROOT,
    DEFAULT_BOUNDARY_JSON,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROI_METRICS,
    highgamma_fif_path,
)
ROI_ORDER = ["EAC", "STG", "IFG", "PRC", "MFG", "TMP", "Other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--boundary-json", type=Path, default=DEFAULT_BOUNDARY_JSON)
    parser.add_argument("--roi-metrics", type=Path, default=DEFAULT_ROI_METRICS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "sentence_boundary_channel_timecourses" / "_tracking_intermediate",
    )
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--transcript-feature-space", default="gpt2-xl")
    parser.add_argument("--boundary-level", choices=["sentence", "event", "constituent"], default="sentence")
    parser.add_argument(
        "--words-csv",
        type=Path,
        default=None,
        help="Prepared words.csv with boundary columns (required for constituent level).",
    )
    parser.add_argument(
        "--boundary-anchor",
        choices=["next_start", "prev_end", "midpoint"],
        default="next_start",
        help="Time used for boundary events. next_start is the first word onset of the next sentence.",
    )
    parser.add_argument("--baseline-window", nargs=2, type=float, default=[-0.6, -0.1])
    parser.add_argument("--response-window", nargs=2, type=float, default=[0.0, 0.6])
    parser.add_argument("--plot-window", nargs=2, type=float, default=[-2.0, 2.0])
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--min-shift", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=24)
    return parser.parse_args()


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    pv = p[finite]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[finite] = restored
    return out


def load_words(root: Path, feature_space: str) -> pd.DataFrame:
    path = root / "stimuli" / feature_space / "transcript.tsv"
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


def load_boundary_indices(path: Path, level: str, n_words: int, words_csv: Path | None = None) -> np.ndarray:
    if level == "constituent":
        if words_csv is None or not words_csv.is_file():
            raise FileNotFoundError("Constituent boundaries require --words-csv pointing to prepared words.csv")
        words = pd.read_csv(words_csv)
        if "constituent_boundary_after" not in words.columns:
            raise ValueError(f"{words_csv} has no constituent_boundary_after column; run prepare_constituent_boundaries.py")
        indices = words.loc[words["constituent_boundary_after"].astype(bool), "word_idx"].astype(int).to_numpy()
        return indices[(indices >= 0) & (indices < n_words)]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    key = f"{level}_end_word_indices"
    if key not in data:
        raise KeyError(f"{path} has no {key}")
    indices = np.asarray(sorted(set(int(x) for x in data[key])), dtype=int)
    return indices[(indices >= 0) & (indices < n_words)]


def boundary_times(words: pd.DataFrame, boundary_end_indices: np.ndarray, anchor: str) -> tuple[np.ndarray, np.ndarray]:
    n_words = len(words)
    usable = boundary_end_indices[(boundary_end_indices >= 0) & (boundary_end_indices < n_words - 1)]
    prev_end = words.loc[usable, "end"].to_numpy(dtype=float)
    next_start = words.loc[usable + 1, "start"].to_numpy(dtype=float)
    if anchor == "prev_end":
        times = prev_end
    elif anchor == "midpoint":
        times = 0.5 * (prev_end + next_start)
    elif anchor == "next_start":
        times = next_start
    else:
        raise ValueError(anchor)
    return times, usable


def shifted_boundaries(boundary_end_indices: np.ndarray, n_words: int, shift: int) -> np.ndarray:
    return (boundary_end_indices + int(shift)) % max(1, n_words - 1)


def choose_shifts(n_words: int, n_permutations: int, min_shift: int, rng: np.random.Generator) -> np.ndarray:
    if n_permutations <= 0:
        return np.asarray([], dtype=int)
    max_index = max(2, n_words - 1)
    min_shift = min(int(min_shift), max(1, max_index // 3))
    possible = np.arange(min_shift, max_index - min_shift, dtype=int)
    if len(possible) == 0:
        possible = np.arange(1, max_index, dtype=int)
    replace = n_permutations > len(possible)
    return rng.choice(possible, size=n_permutations, replace=replace).astype(int)


def window_mean_from_csum(
    csum: np.ndarray,
    centers: np.ndarray,
    start_s: float,
    end_s: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_offset = int(round(start_s * sfreq))
    end_offset = int(round(end_s * sfreq))
    starts = centers + start_offset
    stops = centers + end_offset
    valid = (starts >= 0) & (stops <= csum.shape[1] - 1) & (stops > starts)
    means = np.full((len(centers), csum.shape[0]), np.nan, dtype=np.float32)
    if valid.any():
        seg_sum = csum[:, stops[valid]] - csum[:, starts[valid]]
        means[valid] = (seg_sum / (stops[valid] - starts[valid])[None, :]).T.astype(np.float32)
    return means, valid


def boundary_effect(
    csum: np.ndarray,
    centers: np.ndarray,
    sfreq: float,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
) -> tuple[np.ndarray, int]:
    baseline, valid_base = window_mean_from_csum(csum, centers, baseline_window[0], baseline_window[1], sfreq)
    response, valid_resp = window_mean_from_csum(csum, centers, response_window[0], response_window[1], sfreq)
    valid = valid_base & valid_resp
    if not valid.any():
        return np.full(csum.shape[0], np.nan, dtype=np.float32), 0
    event_effect = response[valid] - baseline[valid]
    return np.nanmean(event_effect, axis=0).astype(np.float32), int(valid.sum())


def mean_timecourse(
    data: np.ndarray,
    centers: np.ndarray,
    sfreq: float,
    plot_window: tuple[float, float],
    baseline_window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    start_offset = int(round(plot_window[0] * sfreq))
    end_offset = int(round(plot_window[1] * sfreq))
    offsets = np.arange(start_offset, end_offset + 1, dtype=int)
    valid = ((centers + offsets[0]) >= 0) & ((centers + offsets[-1]) < data.shape[1])
    times = offsets.astype(float) / sfreq
    if not valid.any():
        return np.full((data.shape[0], len(times)), np.nan, dtype=np.float32), times, 0
    idx = centers[valid, None] + offsets[None, :]
    epochs = data[:, idx]
    tc = epochs.mean(axis=1).astype(np.float32)
    base_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    if base_mask.any():
        tc = tc - np.nanmean(tc[:, base_mask], axis=1, keepdims=True).astype(np.float32)
    return tc, times, int(valid.sum())


def load_roi_table(path: Path, subjects: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    wanted = {f"sub-{subject}" for subject in subjects}
    table = table[table["subject"].isin(wanted)].copy()
    table["paper_roi"] = table["paper_roi"].fillna("Other")
    return table


def plot_roi_summary(summary: pd.DataFrame, output_dir: Path, q_threshold: float, level: str) -> None:
    summary = summary.copy()
    summary["is_significant"] = summary["boundary_q"] <= q_threshold
    roi_summary = (
        summary.groupby("paper_roi", dropna=False)
        .agg(
            n_channels=("channel_id", "size"),
            n_sig=("boundary_q", lambda x: int(np.sum(x <= q_threshold))),
            n_sig_positive=("sig_positive", "sum"),
            n_sig_negative=("sig_negative", "sum"),
            median_effect=("boundary_effect", "median"),
            median_abs_z=("boundary_abs_z", "median"),
        )
        .reset_index()
    )
    roi_summary["sig_fraction"] = roi_summary["n_sig"] / roi_summary["n_channels"].clip(lower=1)
    roi_summary.to_csv(output_dir / f"{level}_boundary_tracking_roi_summary.csv", index=False)

    order = [roi for roi in ROI_ORDER if roi in set(summary["paper_roi"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=roi_summary, x="paper_roi", y="sig_fraction", order=order, color="#4C78A8", ax=ax)
    ax.set_xlabel("paper ROI")
    ax.set_ylabel(f"fraction q <= {q_threshold:g}")
    ax.set_title(f"Channels with {level}-boundary-locked high-gamma responses")
    ax.set_ylim(0, max(0.05, roi_summary["sig_fraction"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(output_dir / f"{level}_boundary_tracking_fraction_by_roi.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=summary, x="paper_roi", y="boundary_effect", order=order, color="white", fliersize=0, ax=ax)
    sns.stripplot(
        data=summary,
        x="paper_roi",
        y="boundary_effect",
        order=order,
        hue="is_significant",
        palette={False: "0.65", True: "#D62728"},
        size=2.5,
        alpha=0.75,
        dodge=False,
        ax=ax,
    )
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("paper ROI")
    ax.set_ylabel("boundary response effect\nmean z(response) - mean z(baseline)")
    ax.set_title(f"{level.capitalize()}-boundary response effect by ROI")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles[:2], labels=[f"q>{q_threshold:g}", f"q<={q_threshold:g}"], title="", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{level}_boundary_effect_by_roi.png", dpi=240)
    plt.close(fig)


def plot_top_timecourses(
    summary: pd.DataFrame,
    timecourses: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    top_n: int,
    level: str,
) -> None:
    if len(summary) == 0:
        return
    ranked = summary.sort_values(["boundary_q", "boundary_abs_z"], ascending=[True, False]).head(top_n)
    n_cols = 4
    n_rows = int(np.ceil(len(ranked) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows), sharex=True, sharey=False)
    axes = np.ravel(np.atleast_1d(axes))
    for ax, (_, row) in zip(axes, ranked.iterrows(), strict=False):
        idx = int(row["timecourse_index"])
        ax.plot(times, timecourses[idx], color="#4C78A8", lw=1.5)
        ax.axvline(0, color="0.15", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvspan(-0.6, -0.1, color="0.85", alpha=0.45, lw=0)
        ax.axvspan(0.0, 0.6, color="#F58518", alpha=0.18, lw=0)
        ax.set_title(
            f"{row['subject']}:{row['channel']} {row['paper_roi']}\n"
            f"effect={row['boundary_effect']:.3f}, q={row['boundary_q']:.3g}",
            fontsize=9,
        )
    for ax in axes[len(ranked) :]:
        ax.axis("off")
    fig.supxlabel("time from boundary anchor (s)")
    fig.supylabel("mean high-gamma z, baseline subtracted")
    fig.tight_layout()
    fig.savefig(output_dir / f"top_{level}_boundary_channel_timecourses.png", dpi=240)
    plt.close(fig)


def plot_volcano(summary: pd.DataFrame, output_dir: Path, q_threshold: float, level: str) -> None:
    data = summary.copy()
    data["neg_log10_q"] = -np.log10(np.maximum(data["boundary_q"].to_numpy(dtype=float), 1e-12))
    fig, ax = plt.subplots(figsize=(7, 5))
    sig = data["boundary_q"] <= q_threshold
    ax.scatter(data.loc[~sig, "boundary_effect"], data.loc[~sig, "neg_log10_q"], s=12, color="0.65", alpha=0.7, label="not significant")
    ax.scatter(data.loc[sig, "boundary_effect"], data.loc[sig, "neg_log10_q"], s=18, color="#D62728", alpha=0.85, label=f"q <= {q_threshold:g}")
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_xlabel("boundary response effect")
    ax.set_ylabel("-log10(q)")
    ax.set_title(f"{level.capitalize()}-boundary channel screen")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{level}_boundary_tracking_volcano.png", dpi=240)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    words = load_words(args.bids_root, args.transcript_feature_space)
    boundary_indices = load_boundary_indices(args.boundary_json, args.boundary_level, len(words), args.words_csv)
    real_event_times, real_boundary_indices = boundary_times(words, boundary_indices, args.boundary_anchor)
    if len(real_event_times) == 0:
        raise ValueError("No usable boundaries after excluding final transcript boundary.")

    rng = np.random.default_rng(args.random_seed)
    shifts = choose_shifts(len(words), args.n_permutations, args.min_shift, rng)
    roi_table = load_roi_table(args.roi_metrics, args.subjects)

    rows = []
    tc_parts = []
    times_ref = None
    tc_index = 0

    for subject in args.subjects:
        subject_label = f"sub-{subject}"
        sub_table = roi_table[roi_table["subject"] == subject_label].copy()
        if sub_table.empty:
            continue
        raw = mne.io.read_raw_fif(highgamma_fif_path(args.bids_root, subject, args.task), preload=False, verbose=False)
        requested = sub_table["channel"].astype(str).drop_duplicates().tolist()
        available = [channel for channel in requested if channel in raw.ch_names]
        if not available:
            continue
        raw.pick(available)
        print(f"{subject_label}: reading {len(available)} channels at {raw.info['sfreq']:.1f} Hz", flush=True)
        data = raw.get_data().astype(np.float32)
        data -= np.nanmean(data, axis=1, keepdims=True).astype(np.float32)
        data /= np.maximum(np.nanstd(data, axis=1, keepdims=True).astype(np.float32), 1e-6)
        sfreq = float(raw.info["sfreq"])
        csum = np.concatenate([np.zeros((data.shape[0], 1), dtype=np.float64), np.cumsum(data.astype(np.float64), axis=1)], axis=1)

        real_centers = raw.time_as_index(real_event_times, use_rounding=True)
        obs, n_real = boundary_effect(
            csum,
            real_centers,
            sfreq,
            tuple(args.baseline_window),
            tuple(args.response_window),
        )
        null = np.full((len(shifts), len(available)), np.nan, dtype=np.float32)
        for perm_i, shift in enumerate(shifts):
            shifted = shifted_boundaries(real_boundary_indices, len(words), int(shift))
            shifted_times, _shifted_indices = boundary_times(words, shifted, args.boundary_anchor)
            centers = raw.time_as_index(shifted_times, use_rounding=True)
            null[perm_i], _n_null = boundary_effect(
                csum,
                centers,
                sfreq,
                tuple(args.baseline_window),
                tuple(args.response_window),
            )

        null_mean = np.nanmean(null, axis=0)
        null_std = np.nanstd(null, axis=0, ddof=1)
        z = (obs - null_mean) / np.maximum(null_std, 1e-8)
        p = (1.0 + np.sum(np.abs(null) >= np.abs(obs)[None, :], axis=0)) / (len(shifts) + 1.0)

        tc, times, n_tc = mean_timecourse(data, real_centers, sfreq, tuple(args.plot_window), tuple(args.baseline_window))
        if times_ref is None:
            times_ref = times
        elif len(times_ref) != len(times) or not np.allclose(times_ref, times):
            raise ValueError("Subjects have different time grids; resampling not implemented.")
        tc_parts.append(tc)

        test_mask = (times >= args.response_window[0]) & (times <= args.response_window[1])
        peak_local = np.nanargmax(np.abs(tc[:, test_mask]), axis=1)
        test_times = times[test_mask]
        test_tc = tc[:, test_mask]
        peak_time = test_times[peak_local]
        peak_value = test_tc[np.arange(len(available)), peak_local]

        channel_meta = sub_table.drop_duplicates("channel").set_index("channel")
        for channel_i, channel in enumerate(available):
            meta = channel_meta.loc[channel]
            rows.append(
                {
                    "subject": subject_label,
                    "channel": channel,
                    "channel_id": f"{subject_label}:{channel}",
                    "paper_roi": str(meta.get("paper_roi", "Other")),
                    "family": str(meta.get("family", "")),
                    "x": float(meta.get("x", np.nan)),
                    "y": float(meta.get("y", np.nan)),
                    "z_coord": float(meta.get("z", np.nan)),
                    "boundary_effect": float(obs[channel_i]),
                    "boundary_null_mean": float(null_mean[channel_i]),
                    "boundary_null_std": float(null_std[channel_i]),
                    "boundary_z": float(z[channel_i]),
                    "boundary_abs_z": float(abs(z[channel_i])),
                    "boundary_p": float(p[channel_i]),
                    "peak_time_s": float(peak_time[channel_i]),
                    "peak_mean_z": float(peak_value[channel_i]),
                    "n_boundaries": int(n_real),
                    "n_timecourse_boundaries": int(n_tc),
                    "n_permutations": int(len(shifts)),
                    "timecourse_index": int(tc_index),
                }
            )
            tc_index += 1

    summary = pd.DataFrame(rows)
    summary["boundary_q"] = fdr_bh(summary["boundary_p"].to_numpy(dtype=float))
    summary["sig_positive"] = (summary["boundary_q"] <= args.q_threshold) & (summary["boundary_effect"] > 0)
    summary["sig_negative"] = (summary["boundary_q"] <= args.q_threshold) & (summary["boundary_effect"] < 0)
    summary = summary.sort_values(["boundary_q", "boundary_abs_z"], ascending=[True, False]).reset_index(drop=True)
    summary.to_csv(args.output_dir / f"{args.boundary_level}_boundary_channel_tracking.csv", index=False)

    timecourses = np.vstack(tc_parts).astype(np.float32)
    np.savez_compressed(
        args.output_dir / f"{args.boundary_level}_boundary_channel_timecourses.npz",
        time=times_ref.astype(np.float32),
        mean_highgamma_z=timecourses,
        channel_id=summary.sort_values("timecourse_index")["channel_id"].astype(str).to_numpy(),
    )

    plot_roi_summary(summary, args.output_dir, args.q_threshold, args.boundary_level)
    plot_top_timecourses(summary, timecourses, times_ref, args.output_dir, args.top_n, args.boundary_level)
    plot_volcano(summary, args.output_dir, args.q_threshold, args.boundary_level)

    top_cols = ["subject", "channel", "paper_roi", "boundary_effect", "boundary_z", "boundary_p", "boundary_q", "peak_time_s"]
    print(f"\nTop {args.boundary_level}-boundary-tracking channels", flush=True)
    print(summary[top_cols].head(args.top_n).round(4).to_string(index=False), flush=True)
    print(f"\nSaved {args.boundary_level}-boundary channel tracking outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
