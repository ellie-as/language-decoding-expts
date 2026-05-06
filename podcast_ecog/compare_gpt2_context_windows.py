#!/usr/bin/env python3
"""Compare concatenated GPT-2 embeddings from the last N words."""

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

from compare_encoding_models import evaluate_model_window, load_or_build_targets  # noqa: E402
from run_all_subject_window_preferences import bids_root, load_words_and_gpt2_features  # noqa: E402
from run_text_window_encoding import DEFAULT_MOUNT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "gpt2_context_window_comparison")
    parser.add_argument("--subject", default="03")
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--target-mode", default="preferred-lag", choices=["preferred-lag"])
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--gpt2-layer", type=int, default=24)
    parser.add_argument("--context-windows", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def concat_last_n(features: np.ndarray, n_words: int) -> np.ndarray:
    n_rows, dim = features.shape
    out = np.zeros((n_rows, dim * int(n_words)), dtype=np.float32)
    for i in range(n_rows):
        for offset in range(int(n_words)):
            source = i - offset
            if source < 0:
                continue
            out[i, offset * dim : (offset + 1) * dim] = features[source]
    return out


def summarize(results: dict[int, np.ndarray], output_dir: Path, channel_names: np.ndarray) -> pd.DataFrame:
    rows = []
    channel_rows = []
    for window, corrs in results.items():
        mean_by_channel = corrs.mean(axis=0)
        rows.append(
            {
                "gpt2_context_words": int(window),
                "feature_dim": int(window) * 1600,
                "mean_r": float(mean_by_channel.mean()),
                "median_r": float(np.median(mean_by_channel)),
                "q75_r": float(np.quantile(mean_by_channel, 0.75)),
                "max_r": float(mean_by_channel.max()),
                "positive_share": float((mean_by_channel > 0).mean()),
            }
        )
        for channel, score in zip(channel_names, mean_by_channel):
            channel_rows.append({"gpt2_context_words": int(window), "channel": str(channel), "r": float(score)})

    summary = pd.DataFrame(rows).sort_values("mean_r", ascending=False)
    channels = pd.DataFrame(channel_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "gpt2_context_window_summary.csv", index=False)
    channels.to_csv(output_dir / "gpt2_context_window_channel_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    plot_df = summary.sort_values("gpt2_context_words")
    ax.plot(plot_df["gpt2_context_words"], plot_df["mean_r"], marker="o", color="#4C78A8")
    ax.set_xlabel("concatenated GPT-2 words")
    ax.set_ylabel("mean channel correlation r")
    ax.set_title("GPT-2 context window comparison")
    fig.savefig(output_dir / "gpt2_context_window_mean_r.png", dpi=220)
    plt.close(fig)

    (output_dir / "gpt2_context_window_comparison_summary.md").write_text(
        "\n".join(["# GPT-2 Context Window Comparison", "", summary.to_markdown(index=False)]) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_payload = load_or_build_targets(args)
    y = target_payload["targets"].astype(np.float32)
    selection = target_payload["selection"].astype(np.int64)
    channel_names = target_payload["channel_names"].astype(str)

    root = bids_root(args)
    _words, gpt2_features = load_words_and_gpt2_features(root, args.gpt2_layer)

    results = {}
    for window in args.context_windows:
        print(f"Evaluating GPT-2 context window={window}", flush=True)
        x_all = concat_last_n(gpt2_features, int(window))
        x = x_all[selection].astype(np.float32)
        results[int(window)] = evaluate_model_window("ridge", int(window), x, y, args)

    np.savez_compressed(
        args.output_dir / "gpt2_context_window_comparison_results.npz",
        context_windows=np.asarray(list(results.keys())),
        corrs=np.stack([results[key] for key in results.keys()]),
        channel_names=channel_names,
    )
    config = vars(args).copy()
    config["mounted_root"] = str(args.mounted_root)
    config["bids_root"] = None if args.bids_root is None else str(args.bids_root)
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = summarize(results, args.output_dir, channel_names)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
