#!/usr/bin/env python3
"""Sweep LLM hidden-state features over models and internal context lengths.

This script is intended for an allocated server node. It reads the Podcast ECoG
dataset from a BIDS/OpenNeuro checkout, extracts one feature vector per current
word from several causal language models, and evaluates each feature space with
the same ridge encoding setup.

Example:

    python podcast_ecog/run_llm_context_sweep.py \
      --bids-root /ceph/behrens/ellie/language-decoding-expts/podcast_ecog/data/ds005574 \
      --models gpt2 gpt2-medium gpt2-large openai-community/gpt2-xl \
      --context-token-lengths 0 1 2 4 8 16 32 64 \
      --subjects 03
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
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
from run_text_window_encoding import DEFAULT_MOUNT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "llm_context_sweep")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "gpt2-medium", "gpt2-large"],
        help="Hugging Face causal LM ids.",
    )
    parser.add_argument("--context-token-lengths", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["final"],
        help="Hidden layers to evaluate: final, all, or integer layer indices.",
    )
    parser.add_argument("--subjects", nargs="+", default=["03"])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--target-mode", default="preferred-lag", choices=["preferred-lag"])
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extract-only", action="store_true", help="Only cache LLM features; skip ECoG ridge evaluation.")
    parser.add_argument("--force-features", action="store_true", help="Recompute feature caches.")
    parser.add_argument("--force-targets", action="store_true", help="Recompute ECoG target caches.")
    return parser.parse_args()


def resolve_bids_root(args: argparse.Namespace) -> Path:
    return args.bids_root or args.mounted_root / "podcast_ecog" / "data" / "ds005574"


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def load_token_table(bids_root: Path, tokenizer) -> pd.DataFrame:
    transcript_path = bids_root / "stimuli" / "podcast_transcript.csv"
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    transcript = pd.read_csv(transcript_path)
    if "start" in transcript.columns:
        transcript = transcript.sort_values("start").reset_index(drop=True)
    transcript.insert(0, "word_idx", transcript.index.values)
    transcript["hftoken"] = transcript.word.apply(lambda word: tokenizer.tokenize(" " + str(word)))
    table = transcript.explode("hftoken", ignore_index=True)
    table["token_id"] = table.hftoken.apply(tokenizer.convert_tokens_to_ids).astype(int)
    return table


def parse_layers(requested: list[str], n_hidden_layers: int) -> list[int]:
    if requested == ["all"]:
        return list(range(n_hidden_layers + 1))
    out = []
    for item in requested:
        if item == "final":
            out.append(n_hidden_layers)
        else:
            layer = int(item)
            if layer < 0:
                layer = n_hidden_layers + 1 + layer
            out.append(layer)
    return sorted(set(out))


def feature_cache_path(output_dir: Path, model_name: str, context_len: int, layer: int, max_words: int | None) -> Path:
    word_tag = f"maxwords{max_words}" if max_words is not None else "allwords"
    return (
        output_dir
        / "features"
        / safe_name(model_name)
        / f"ctx{int(context_len)}"
        / f"layer{int(layer)}__{word_tag}.npz"
    )


def build_context_matrix(token_ids: np.ndarray, context_len: int, pad_id: int) -> np.ndarray:
    seq_len = int(context_len) + 1
    data = np.full((len(token_ids), seq_len), pad_id, dtype=np.int64)
    for i in range(len(token_ids)):
        segment = token_ids[max(0, i - int(context_len)) : i + 1]
        data[i, -len(segment) :] = segment
    return data


def pool_token_to_word(token_features: np.ndarray, token_table: pd.DataFrame, max_words: int | None) -> np.ndarray:
    features = []
    for word_idx, group in token_table.groupby("word_idx", sort=True):
        if max_words is not None and int(word_idx) >= int(max_words):
            break
        features.append(token_features[group.index.to_numpy()].mean(axis=0))
    return np.vstack(features).astype(np.float32)


def extract_features_for_context(
    *,
    model,
    tokenizer,
    token_table: pd.DataFrame,
    model_name: str,
    context_len: int,
    layers: list[int],
    batch_size: int,
    device: str,
    output_dir: Path,
    max_words: int | None,
    force: bool,
) -> dict[int, Path]:
    paths = {
        layer: feature_cache_path(output_dir, model_name, context_len, layer, max_words)
        for layer in layers
    }
    if not force and all(path.is_file() for path in paths.values()):
        for path in paths.values():
            print(f"Feature cache exists: {path}", flush=True)
        return paths

    if max_words is None:
        token_table_used = token_table
    else:
        token_table_used = token_table[token_table["word_idx"] < int(max_words)].copy()

    token_ids = token_table_used["token_id"].to_numpy(dtype=np.int64)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    data = build_context_matrix(token_ids, int(context_len), int(pad_id))

    layer_chunks = {layer: [] for layer in layers}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), int(batch_size)):
            batch = torch.tensor(data[start : start + int(batch_size)], dtype=torch.long, device=device)
            output = model(batch, output_hidden_states=True)
            for layer in layers:
                values = output.hidden_states[layer][:, -1, :].detach().float().cpu().numpy()
                layer_chunks[layer].append(values.astype(np.float32))
            del output, batch

    for layer in layers:
        token_features = np.vstack(layer_chunks[layer])
        word_features = pool_token_to_word(token_features, token_table_used, max_words=max_words)
        path = paths[layer]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=word_features.astype(np.float32),
            model_name=np.array(model_name),
            context_len=np.array(context_len),
            layer=np.array(layer),
            max_words=np.array(-1 if max_words is None else max_words),
        )
        print(f"Saved {path} shape={word_features.shape}", flush=True)
    return paths


def subject_reference_path(args: argparse.Namespace, subject: str) -> Path | None:
    if args.reference_results is not None and len(args.subjects) == 1:
        return args.reference_results
    default = (
        args.mounted_root
        / "podcast_ecog"
        / "outputs_all_channels"
        / f"sub-{subject}_gpt2-xl_layer-24_encoding_results.npz"
    )
    return default if default.is_file() else None


def target_cache_dir(args: argparse.Namespace, subject: str) -> Path:
    return args.output_dir / "targets" / f"sub-{subject}"


def load_subject_targets(args: argparse.Namespace, subject: str):
    target_dir = target_cache_dir(args, subject)
    target_file = target_dir / "preferred_lag_targets.npz"
    if target_file.is_file() and not args.force_targets:
        print(f"Loading target cache: {target_file}", flush=True)
        with np.load(target_file, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    # Reuse the existing target-builder by making a shallow Namespace with
    # per-subject output/cache settings.
    from argparse import Namespace

    target_args = Namespace(**vars(args))
    target_args.subject = subject
    target_args.output_dir = target_dir
    target_args.reference_results = subject_reference_path(args, subject)
    if target_args.reference_results is None:
        raise FileNotFoundError(
            f"No preferred-lag reference result found for sub-{subject}. "
            "Run a GPT-2 lag reference first or pass --reference-results for a single subject."
        )
    payload = load_or_build_targets(target_args)
    source_file = target_dir / "cache" / f"sub-{subject}_allwords_allch_._preferred_lag_targets.npz"
    # The exact cache name changes with max/picks flags; find it robustly.
    matches = sorted((target_dir / "cache").glob("*preferred_lag_targets.npz"))
    if matches:
        source_file = matches[-1]
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if source_file.is_file() and source_file != target_file:
        with np.load(source_file, allow_pickle=True) as data:
            np.savez_compressed(target_file, **{key: data[key] for key in data.files})
    return payload


def evaluate_feature_path(
    *,
    args: argparse.Namespace,
    subject: str,
    feature_path: Path,
    targets: dict,
) -> dict:
    data = np.load(feature_path)
    features = data["features"].astype(np.float32)
    selection = targets["selection"].astype(np.int64)
    y = targets["targets"].astype(np.float32)
    if int(selection.max()) >= len(features):
        raise ValueError(f"Selection max {selection.max()} exceeds feature rows {features.shape} in {feature_path}")
    x = features[selection]
    corrs = evaluate_model_window("ridge", 0, x, y, args)
    mean_by_channel = corrs.mean(axis=0)
    return {
        "subject": f"sub-{subject}",
        "feature_path": str(feature_path),
        "mean_r": float(mean_by_channel.mean()),
        "median_r": float(np.median(mean_by_channel)),
        "q75_r": float(np.quantile(mean_by_channel, 0.75)),
        "max_r": float(mean_by_channel.max()),
        "positive_share": float((mean_by_channel > 0).mean()),
        "corrs": corrs,
    }


def write_summary(args: argparse.Namespace, rows: list[dict], corr_payloads: dict[str, np.ndarray]) -> None:
    if not rows:
        return
    summary = pd.DataFrame([{k: v for k, v in row.items() if k != "corrs"} for row in rows])
    summary = summary.sort_values(["mean_r", "median_r"], ascending=False)
    summary.to_csv(args.output_dir / "llm_context_sweep_summary.csv", index=False)
    np.savez_compressed(
        args.output_dir / "llm_context_sweep_corrs.npz",
        **corr_payloads,
    )

    plot_df = summary.copy()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for (model_name, layer), group in plot_df.groupby(["model", "layer"], sort=True):
        group = group.sort_values("context_tokens")
        ax.plot(group["context_tokens"], group["mean_r"], marker="o", label=f"{model_name} L{layer}")
    ax.set_xlabel("previous model tokens available")
    ax.set_ylabel("mean channel correlation r")
    ax.set_title("LLM internal context sweep")
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(args.output_dir / "llm_context_sweep_mean_r.png", dpi=220)
    plt.close(fig)

    (args.output_dir / "llm_context_sweep_summary.md").write_text(
        "\n".join(["# LLM Context Sweep", "", summary.to_markdown(index=False)]) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bids = resolve_bids_root(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    config = vars(args).copy()
    config["mounted_root"] = str(args.mounted_root)
    config["bids_root"] = str(bids)
    config["output_dir"] = str(args.output_dir)
    config["torch_dtype"] = str(dtype)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    all_rows = []
    corr_payloads: dict[str, np.ndarray] = {}
    targets_by_subject = None
    if not args.extract_only:
        targets_by_subject = {subject: load_subject_targets(args, subject) for subject in args.subjects}

    for model_name in args.models:
        t0 = time.time()
        print(f"=== Loading model {model_name} on {device} dtype={dtype} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            torch_dtype=dtype,
        ).to(device)
        token_table = load_token_table(bids, tokenizer)
        layers = parse_layers(args.layers, int(model.config.num_hidden_layers))
        print(f"Layers: {layers}; hidden size={model.config.hidden_size}", flush=True)

        for context_len in args.context_token_lengths:
            feature_paths = extract_features_for_context(
                model=model,
                tokenizer=tokenizer,
                token_table=token_table,
                model_name=model_name,
                context_len=int(context_len),
                layers=layers,
                batch_size=args.batch_size,
                device=device,
                output_dir=args.output_dir,
                max_words=args.max_words,
                force=args.force_features,
            )
            if args.extract_only:
                continue
            for layer, feature_path in feature_paths.items():
                for subject in args.subjects:
                    print(f"Evaluating subject={subject} model={model_name} ctx={context_len} layer={layer}", flush=True)
                    result = evaluate_feature_path(
                        args=args,
                        subject=subject,
                        feature_path=feature_path,
                        targets=targets_by_subject[subject],
                    )
                    row = {
                        "subject": f"sub-{subject}",
                        "model": model_name,
                        "context_tokens": int(context_len),
                        "layer": int(layer),
                        "feature_dim": int(np.load(feature_path)["features"].shape[1]),
                        **{k: result[k] for k in ["mean_r", "median_r", "q75_r", "max_r", "positive_share"]},
                        "feature_path": str(feature_path),
                    }
                    all_rows.append(row)
                    key = f"{safe_name(model_name)}__ctx{context_len}__layer{layer}__sub{subject}"
                    corr_payloads[key] = result["corrs"]

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"=== Finished {model_name} in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    write_summary(args, all_rows, corr_payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
