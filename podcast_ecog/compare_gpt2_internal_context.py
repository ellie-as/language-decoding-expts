#!/usr/bin/env python3
"""Recompute one-word GPT-2 features with different internal context lengths."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from compare_encoding_models import evaluate_model_window, load_or_build_targets  # noqa: E402
from run_all_subject_window_preferences import bids_root  # noqa: E402
from run_text_window_encoding import DEFAULT_MOUNT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "gpt2_internal_context_comparison")
    parser.add_argument("--subject", default="03")
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--target-mode", default="preferred-lag", choices=["preferred-lag"])
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden state layer. -1 uses final layer.")
    parser.add_argument("--context-token-lengths", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_token_table(root: Path, tokenizer) -> pd.DataFrame:
    transcript_path = root / "stimuli" / "podcast_transcript.csv"
    transcript = pd.read_csv(transcript_path)
    transcript.insert(0, "word_idx", transcript.index.values)
    transcript["hftoken"] = transcript.word.apply(lambda word: tokenizer.tokenize(" " + str(word)))
    table = transcript.explode("hftoken", ignore_index=True)
    table["token_id"] = table.hftoken.apply(tokenizer.convert_tokens_to_ids).astype(int)
    return table


def cache_file(output_dir: Path, model_name: str, layer: int, context_len: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")
    return output_dir / "cache" / f"{safe_model}_layer{layer}_ctx{context_len}_word_features.npz"


def extract_word_features(
    *,
    model,
    token_table: pd.DataFrame,
    model_name: str,
    layer: int,
    context_len: int,
    batch_size: int,
    device: str,
    output_dir: Path,
) -> np.ndarray:
    path = cache_file(output_dir, model_name, layer, context_len)
    if path.is_file():
        print(f"Loading cached GPT-2 context features: {path}", flush=True)
        return np.load(path)["features"].astype(np.float32)

    token_ids = token_table["token_id"].to_numpy(dtype=np.int64)
    pad_id = 0 if model.config.pad_token_id is None else int(model.config.pad_token_id)
    seq_len = int(context_len) + 1
    data = np.full((len(token_ids), seq_len), pad_id, dtype=np.int64)
    for i in range(len(token_ids)):
        segment = token_ids[max(0, i - int(context_len)) : i + 1]
        data[i, -len(segment) :] = segment

    selected_layer = layer
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.tensor(data[start : start + batch_size], dtype=torch.long, device=device)
            output = model(batch, output_hidden_states=True)
            states = output.hidden_states[selected_layer][:, -1, :].detach().cpu().numpy().astype(np.float32)
            embeddings.append(states)
    token_features = np.vstack(embeddings)

    word_features = []
    for _word_idx, group in token_table.groupby("word_idx", sort=True):
        word_features.append(token_features[group.index.to_numpy()].mean(axis=0))
    features = np.vstack(word_features).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, context_len=np.array(context_len), model_name=np.array(model_name))
    print(f"Saved GPT-2 context features: {path}", flush=True)
    return features


def summarize(results: dict[int, np.ndarray], output_dir: Path, channel_names: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    channel_rows = []
    for context_len, corrs in results.items():
        mean_by_channel = corrs.mean(axis=0)
        rows.append(
            {
                "context_tokens": int(context_len),
                "mean_r": float(mean_by_channel.mean()),
                "median_r": float(np.median(mean_by_channel)),
                "q75_r": float(np.quantile(mean_by_channel, 0.75)),
                "max_r": float(mean_by_channel.max()),
                "positive_share": float((mean_by_channel > 0).mean()),
            }
        )
        for channel, score in zip(channel_names, mean_by_channel):
            channel_rows.append({"context_tokens": int(context_len), "channel": str(channel), "r": float(score)})
    summary = pd.DataFrame(rows).sort_values("mean_r", ascending=False)
    channels = pd.DataFrame(channel_rows)
    summary.to_csv(output_dir / "gpt2_internal_context_summary.csv", index=False)
    channels.to_csv(output_dir / "gpt2_internal_context_channel_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    plot_df = summary.sort_values("context_tokens")
    ax.plot(plot_df["context_tokens"], plot_df["mean_r"], marker="o", color="#4C78A8")
    ax.set_xlabel("previous GPT-2 tokens available")
    ax.set_ylabel("mean channel correlation r")
    ax.set_title(f"One-word {model_name} features by internal context")
    fig.savefig(output_dir / "gpt2_internal_context_mean_r.png", dpi=220)
    plt.close(fig)

    (output_dir / "gpt2_internal_context_comparison_summary.md").write_text(
        "\n".join(["# GPT-2 Internal Context Comparison", "", summary.to_markdown(index=False)]) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Loading {args.model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    root = bids_root(args)
    token_table = load_token_table(root, tokenizer)
    target_payload = load_or_build_targets(args)
    y = target_payload["targets"].astype(np.float32)
    selection = target_payload["selection"].astype(np.int64)
    channel_names = target_payload["channel_names"].astype(str)

    results = {}
    for context_len in args.context_token_lengths:
        features = extract_word_features(
            model=model,
            token_table=token_table,
            model_name=args.model_name,
            layer=args.layer,
            context_len=int(context_len),
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
        )
        x = features[selection].astype(np.float32)
        print(f"Evaluating context_len={context_len}, X={x.shape}", flush=True)
        results[int(context_len)] = evaluate_model_window("ridge", int(context_len), x, y, args)

    np.savez_compressed(
        args.output_dir / "gpt2_internal_context_comparison_results.npz",
        context_token_lengths=np.asarray(list(results.keys())),
        corrs=np.stack([results[key] for key in results.keys()]),
        channel_names=channel_names,
        model_name=np.array(args.model_name),
        layer=np.array(args.layer),
    )
    config = vars(args).copy()
    config["mounted_root"] = str(args.mounted_root)
    config["bids_root"] = None if args.bids_root is None else str(args.bids_root)
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = summarize(results, args.output_dir, channel_names, args.model_name)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
