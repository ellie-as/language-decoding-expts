#!/usr/bin/env python3
"""Compare GPT-2 word features with sentence/text-window embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
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
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "feature_space_comparison")
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=THIS_DIR / "outputs" / "text_window_encoding" / "cache",
    )
    parser.add_argument("--subject", default="03")
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--target-mode", default="preferred-lag", choices=["preferred-lag"])
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--gpt2-layer", type=int, default=24)
    parser.add_argument("--sentence-windows", nargs="+", type=int, default=[1, 10, 100, 200])
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def find_embedding_cache(cache_dir: Path, window_size: int) -> Path:
    matches = sorted(cache_dir.glob(f"text_window_w{window_size}__*.npz"))
    if not matches:
        raise FileNotFoundError(f"No cached MiniLM embeddings for window {window_size} under {cache_dir}")
    return matches[-1]


def load_sentence_embeddings(cache_dir: Path, window_size: int, selection: np.ndarray) -> np.ndarray:
    path = find_embedding_cache(cache_dir, window_size)
    print(f"Loading sentence embeddings w={window_size}: {path}", flush=True)
    return np.load(path)["embeddings"].astype(np.float32)[selection]


def summarize(results: dict[str, np.ndarray], output_dir: Path, channel_names: np.ndarray) -> pd.DataFrame:
    rows = []
    channel_rows = []
    for feature_name, corrs in results.items():
        mean_by_channel = corrs.mean(axis=0)
        rows.append(
            {
                "feature_space": feature_name,
                "mean_r": float(mean_by_channel.mean()),
                "median_r": float(np.median(mean_by_channel)),
                "q75_r": float(np.quantile(mean_by_channel, 0.75)),
                "max_r": float(mean_by_channel.max()),
                "positive_share": float((mean_by_channel > 0).mean()),
            }
        )
        for channel, score in zip(channel_names, mean_by_channel):
            channel_rows.append({"feature_space": feature_name, "channel": str(channel), "r": float(score)})

    summary = pd.DataFrame(rows).sort_values("mean_r", ascending=False)
    channels = pd.DataFrame(channel_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "feature_space_summary.csv", index=False)
    channels.to_csv(output_dir / "feature_space_channel_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    ax.bar(summary["feature_space"], summary["mean_r"], color="#4C78A8")
    ax.set_ylabel("mean channel correlation r")
    ax.set_title("GPT-2 vs sentence embedding features")
    ax.tick_params(axis="x", labelrotation=30)
    fig.savefig(output_dir / "feature_space_mean_r.png", dpi=220)
    plt.close(fig)

    (output_dir / "feature_space_comparison_summary.md").write_text(
        "\n".join(
            [
                "# Feature Space Comparison",
                "",
                summary.to_markdown(index=False),
            ]
        )
        + "\n",
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
    words, gpt2_features = load_words_and_gpt2_features(root, args.gpt2_layer)
    x_gpt2 = gpt2_features[selection].astype(np.float32)

    results = {
        f"gpt2_layer{args.gpt2_layer}": evaluate_model_window(
            "ridge", 0, x_gpt2, y, args
        )
    }
    for window in args.sentence_windows:
        x_sent = load_sentence_embeddings(args.embedding_cache_dir, int(window), selection)
        results[f"minilm_w{int(window)}"] = evaluate_model_window("ridge", int(window), x_sent, y, args)

    np.savez_compressed(
        args.output_dir / "feature_space_comparison_results.npz",
        feature_spaces=np.asarray(list(results.keys())),
        corrs=np.stack([results[key] for key in results.keys()]),
        channel_names=channel_names,
    )
    config = vars(args).copy()
    config["mounted_root"] = str(args.mounted_root)
    config["bids_root"] = None if args.bids_root is None else str(args.bids_root)
    config["output_dir"] = str(args.output_dir)
    config["embedding_cache_dir"] = str(args.embedding_cache_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = summarize(results, args.output_dir, channel_names)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
