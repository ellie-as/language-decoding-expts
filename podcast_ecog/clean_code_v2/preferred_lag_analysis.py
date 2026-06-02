#!/usr/bin/env python3
"""Per-channel preferred-lag encoding analysis in **word position**.

This complements ``preferred_lag_time_analysis.py`` (peri-word time in seconds,
like ``run_encoding.py``). Here, for each channel we ask: which **word's**
embedding best predicts the current neural response?

Mechanics, kept consistent with ``train_decoders.py``:

  - neural features are the cached per-word high-gamma values from ``prepare_data.py``
    (one scalar per word per channel, e.g. the ``midpoint`` window cache)
  - the predictors are a prepared word representation selected with ``--target-stem``
    (static ``word_vectors_pca{N}`` word2vec, or contextual ``gpt2_ctx32_layer8_pca{N}``)
  - for a signed lag L, the embedding of word ``t + L`` is used to predict the
    neural response at word ``t``. A single cross-validated ridge model is fit
    over all channels at once; each channel's out-of-fold correlation across
    held-out words is the encoding performance at that lag.
  - the preferred lag for a channel is the lag with the highest encoding
    correlation (signed or absolute, see ``--metric``).

Lag sign convention (matches the embedding offset):

  - L = 0 : current word
  - L > 0 : a future word's embedding predicts current neural (leading / anticipatory)
  - L < 0 : a past word's embedding predicts current neural (the channel still
            carries information about earlier words)
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

from common import DEFAULT_OUTPUT_ROOT, clean_subject, subject_label  # noqa: E402
from preferred_lag_common import (  # noqa: E402
    channel_roi_labels,
    circular_shift_null_pvalues_scalar_lags,
    corr_across_words,
    cv_encode_predict,
    load_prepared_scalar_neural,
    load_roi_lookup,
    resolve_target_path,
    select_channels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "preferred_lag")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument(
        "--rois",
        nargs="+",
        default=["ALL"],
        help="Restrict to channels in these paper ROIs (union). ALL keeps every channel.",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        help="Signed word-level lags. L>0 future word embedding, L<0 past word embedding.",
    )
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument(
        "--target-stem",
        default=None,
        help=(
            "Stem of the prepared target .npy file, relative to --prepared-dir. "
            "Defaults to word_vectors_pca{n_components}. Use gpt2_ctx32_layer8_pca{n_components} for GPT-2."
        ),
    )
    parser.add_argument("--target-file", type=Path, default=None, help="Explicit target .npy path. Overrides --target-stem.")
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--metric", choices=["corr", "abs-corr"], default="corr", help="Pick preferred lag by max correlation or max |correlation|.")
    parser.add_argument(
        "--null-iters",
        type=int,
        default=0,
        help=(
            "Circular-shift null iterations for winning-lag significance. 0 disables. "
            "Each iteration rolls the held-out neural responses and takes the max correlation "
            "over lags, so the p-value is corrected for lag selection. Requires --sample-mode all_lags."
        ),
    )
    parser.add_argument("--min-shift", type=int, default=20, help="Minimum circular shift (words) for the null.")
    parser.add_argument("--significance-alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=["all_lags", "per_lag"],
        default="all_lags",
        help="all_lags uses the same current-word positions for every lag (comparable across lags).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def shared_current_indices(valid_neural: np.ndarray, finite_channels: np.ndarray, word_scores: np.ndarray, lags: list[int]) -> np.ndarray:
    """Current-word positions valid for the neural data and every requested lag."""
    n_words = len(word_scores)
    current = np.flatnonzero(valid_neural & finite_channels)
    keep = np.ones(len(current), dtype=bool)
    for lag in lags:
        emb_idx = current + int(lag)
        in_range = (emb_idx >= 0) & (emb_idx < n_words)
        finite = np.zeros(len(current), dtype=bool)
        finite[in_range] = np.isfinite(word_scores[emb_idx[in_range]]).all(axis=1)
        keep &= in_range & finite
    return current[keep]


def per_lag_indices(valid_neural: np.ndarray, finite_channels: np.ndarray, word_scores: np.ndarray, lag: int) -> np.ndarray:
    n_words = len(word_scores)
    current = np.flatnonzero(valid_neural & finite_channels)
    emb_idx = current + int(lag)
    in_range = (emb_idx >= 0) & (emb_idx < n_words)
    finite = np.zeros(len(current), dtype=bool)
    finite[in_range] = np.isfinite(word_scores[emb_idx[in_range]]).all(axis=1)
    keep = in_range & finite
    return current[keep]


def median_word_spacing(words_csv: Path) -> float:
    if not words_csv.is_file():
        return float("nan")
    starts = pd.read_csv(words_csv)["start"].to_numpy(dtype=float)
    diffs = np.diff(np.sort(starts))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else float("nan")


def save_lag_profile_plot(lags: list[int], corr_by_lag: np.ndarray, out_path: Path, title: str) -> None:
    mean = np.nanmean(corr_by_lag, axis=1)
    n = np.sum(np.isfinite(corr_by_lag), axis=1)
    err = np.nanstd(corr_by_lag, axis=1) / np.sqrt(np.maximum(n, 1))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(lags, mean, color="black", marker="o")
    ax.fill_between(lags, mean - err, mean + err, alpha=0.15, color="black")
    ax.axvline(0, c=(0.85, 0.85, 0.85), ls="--")
    ax.axhline(0, c=(0.85, 0.85, 0.85), ls="--")
    ax.set_xlabel("word lag (embedding offset)")
    ax.set_ylabel("encoding performance (r +/- sem over channels)")
    ax.set_title(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = [clean_subject(s) for s in args.subjects]
    lags = [int(lag) for lag in args.lags]
    target_path, target_name = resolve_target_path(args.prepared_dir, args.target_stem, args.target_file, args.n_components)
    word_scores = np.load(target_path).astype(np.float32)
    roi_lookup = load_roi_lookup(args.prepared_dir)
    spacing = median_word_spacing(args.prepared_dir / "words.csv")
    rng = np.random.default_rng(args.random_seed)
    run_null = args.null_iters > 0
    if run_null and args.sample_mode != "all_lags":
        raise ValueError("--null-iters requires --sample-mode all_lags so the shifted target is shared across lags.")
    print(f"Using target features: {target_path} shape={word_scores.shape}", flush=True)

    summary_csv = args.output_dir / f"preferred_lag__{target_name}.csv"
    if summary_csv.is_file() and not args.overwrite:
        print(f"Output already exists (use --overwrite): {summary_csv}", flush=True)
        return 0

    rows = []
    all_corr_by_lag: list[np.ndarray] = []
    for subject in subjects:
        x_all, valid, channels = load_prepared_scalar_neural(args.prepared_dir, subject)
        roi_labels = channel_roi_labels(channels, roi_lookup, subject)
        cols = select_channels(channels, roi_labels, args.rois)
        if len(cols) == 0:
            print(f"{subject_label(subject)}: no channels for ROIs {args.rois}, skipping", flush=True)
            continue
        x_roi = x_all[:, cols].astype(np.float32)
        finite_channels = np.isfinite(x_roi).all(axis=1)

        if args.sample_mode == "all_lags":
            shared = shared_current_indices(valid, finite_channels, word_scores, lags)
            print(f"{subject_label(subject)}: {len(cols)} channels, {len(shared)} shared words", flush=True)

        corr_by_lag = np.full((len(lags), len(cols)), np.nan, dtype=np.float32)
        n_samples_by_lag = np.zeros(len(lags), dtype=int)
        preds_by_lag: list[np.ndarray | None] = [None] * len(lags)
        for li, lag in enumerate(lags):
            current = shared if args.sample_mode == "all_lags" else per_lag_indices(valid, finite_channels, word_scores, lag)
            if len(current) <= args.outer_splits:
                continue
            y = x_roi[current].astype(np.float32)
            x = word_scores[current + lag].astype(np.float32)
            y_pred = cv_encode_predict(x, y, args.ridge_alpha, args.outer_splits)
            corr_by_lag[li] = corr_across_words(y, y_pred)
            n_samples_by_lag[li] = len(current)
            if run_null:
                preds_by_lag[li] = y_pred

        score = np.abs(corr_by_lag) if args.metric == "abs-corr" else corr_by_lag
        valid_lag = np.isfinite(score).any(axis=0)
        preferred_idx = np.full(len(cols), -1, dtype=int)
        preferred_idx[valid_lag] = np.nanargmax(score[:, valid_lag], axis=0)

        best_corr_p = np.full(len(cols), np.nan, dtype=np.float32)
        if run_null:
            observed_best = np.nan_to_num(np.nanmax(corr_by_lag, axis=0), nan=-np.inf).astype(np.float32)
            y_shared = x_roi[shared].astype(np.float32)
            best_corr_p = circular_shift_null_pvalues_scalar_lags(
                y_shared, preds_by_lag, observed_best, args.null_iters, args.min_shift, rng
            )
            n_sig = int((best_corr_p <= args.significance_alpha).sum())
            print(f"{subject_label(subject)}: {n_sig}/{len(cols)} channels significant (p<={args.significance_alpha})", flush=True)

        lags_arr = np.asarray(lags, dtype=int)
        all_corr_by_lag.append(corr_by_lag)
        for ci, col in enumerate(cols):
            if preferred_idx[ci] < 0:
                continue
            pl = int(lags_arr[preferred_idx[ci]])
            row = {
                "subject": subject_label(subject),
                "channel": channels[col],
                "paper_roi": roi_labels[col],
                "n_channels_subject": int(len(cols)),
                "preferred_lag_words": pl,
                "preferred_lag_sec_approx": float(pl * spacing) if np.isfinite(spacing) else float("nan"),
                "corr_at_preferred_lag": float(corr_by_lag[preferred_idx[ci], ci]),
                "max_abs_corr": float(np.nanmax(np.abs(corr_by_lag[:, ci]))),
            }
            if run_null:
                row["best_corr_p"] = float(best_corr_p[ci])
                row["significant_best"] = bool(best_corr_p[ci] <= args.significance_alpha)
            for li, lag in enumerate(lags):
                row[f"corr_lag_{lag:+d}"] = float(corr_by_lag[li, ci])
            rows.append(row)

        prof_path = args.output_dir / f"lag_profile__{target_name}__{subject_label(subject)}.png"
        save_lag_profile_plot(lags, corr_by_lag, prof_path, f"{subject_label(subject)} {target_name}")

    if not rows:
        print("No channels produced results.", flush=True)
        return 1

    table = pd.DataFrame(rows)
    table.to_csv(summary_csv, index=False)
    print(f"Saved per-channel preferred lag: {summary_csv}", flush=True)

    stacked = np.concatenate(all_corr_by_lag, axis=1) if all_corr_by_lag else np.zeros((len(lags), 0))
    save_lag_profile_plot(lags, stacked, args.output_dir / f"lag_profile__{target_name}__ALL.png", f"all channels {target_name}")

    counts = table["preferred_lag_words"].value_counts().sort_index()
    summary = {
        "analysis": "word_position",
        "target_name": target_name,
        "target_path": str(target_path),
        "subjects": subjects,
        "rois": args.rois,
        "lags": lags,
        "metric": args.metric,
        "sample_mode": args.sample_mode,
        "ridge_alpha": float(args.ridge_alpha),
        "outer_splits": int(args.outer_splits),
        "median_word_spacing_sec": spacing,
        "n_channels_total": int(len(table)),
        "preferred_lag_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "mean_corr_at_preferred_lag": float(table["corr_at_preferred_lag"].mean()),
    }
    if "significant_best" in table.columns:
        sig = table[table["significant_best"]]
        sig_counts = sig["preferred_lag_words"].value_counts().sort_index()
        summary.update(
            {
                "null_iters": int(args.null_iters),
                "min_shift": int(args.min_shift),
                "significance_alpha": float(args.significance_alpha),
                "n_significant_best": int(len(sig)),
                "preferred_lag_counts_significant": {str(int(k)): int(v) for k, v in sig_counts.items()},
            }
        )
    with (args.output_dir / f"preferred_lag_summary__{target_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Preferred-lag counts (words):", flush=True)
    print(counts.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
