#!/usr/bin/env python3
"""Run GPT-2 context-size and layer encoding sweeps for all Podcast ECoG subjects.

The script is intended for an allocated cluster node. It saves intermediate
feature caches, per-subject target caches, fold correlations, and tidy summary
tables for later local analysis.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from compare_encoding_models import evaluate_model_window  # noqa: E402
from run_all_subject_window_preferences import (  # noqa: E402
    compute_gpt2_lag_reference,
    fif_path,
    load_words_and_gpt2_features,
)
from run_llm_context_sweep import (  # noqa: E402
    extract_features_for_context,
    load_token_table,
    parse_layers,
    resolve_device,
    resolve_dtype,
    safe_name,
)
from run_text_window_encoding import DEFAULT_MOUNT_ROOT, load_epochs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounted-root", type=Path, default=DEFAULT_MOUNT_ROOT)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "outputs" / "gpt2_all_subject_context_layer_sweep",
    )
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--picks-regex", default=".*")
    parser.add_argument("--model", default="gpt2", help="Hugging Face causal LM id. Intended default: gpt2.")
    parser.add_argument("--reference-gpt2-layer", type=int, default=24)
    parser.add_argument("--context-token-lengths", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["all"],
        help="GPT-2 hidden layers to evaluate: all, final, or integer layer indices. Layer 0 is embeddings.",
    )
    parser.add_argument("--resample-sfreq", type=float, default=32.0)
    parser.add_argument("--lag-tmin", type=float, default=-2.0, help="Start of epoch window, in seconds, for lag selection.")
    parser.add_argument("--lag-tmax", type=float, default=2.0, help="End of epoch window, in seconds, for lag selection.")
    parser.add_argument("--outer-splits", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=100000.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-targets", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument(
        "--no-append-existing",
        action="store_true",
        help="Start fresh summary/correlation outputs instead of merging with files already in --output-dir.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-evaluate combinations that already exist in the summary outputs.",
    )
    return parser.parse_args()


def bids_root(args: argparse.Namespace) -> Path:
    return args.bids_root or args.mounted_root / "podcast_ecog" / "data" / "ds005574"


def cache_tag(args: argparse.Namespace) -> str:
    word_tag = f"maxwords{args.max_words}" if args.max_words is not None else "allwords"
    chan_tag = f"maxch{args.max_channels}" if args.max_channels is not None else "allch"
    picks = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.picks_regex).strip("-") or "all"
    lag_tag = ""
    if float(args.lag_tmin) != -2.0 or float(args.lag_tmax) != 2.0:
        lag_tmin = str(float(args.lag_tmin)).replace("-", "m").replace(".", "p")
        lag_tmax = str(float(args.lag_tmax)).replace("-", "m").replace(".", "p")
        lag_tag = f"_lag{lag_tmin}to{lag_tmax}"
    return f"{word_tag}_{chan_tag}_{picks}{lag_tag}"


def reference_cache_path(args: argparse.Namespace, subject: str) -> Path:
    return args.output_dir / "references" / f"sub-{subject}_{cache_tag(args)}_gpt2xl-layer{args.reference_gpt2_layer}_lag_reference.npz"


def target_cache_path(args: argparse.Namespace, subject: str) -> Path:
    return args.output_dir / "targets" / f"sub-{subject}_{cache_tag(args)}_preferred_lag_targets.npz"


def load_or_build_subject_targets(
    *,
    args: argparse.Namespace,
    subject: str,
    words: pd.DataFrame,
    reference_features: np.ndarray,
) -> dict[str, np.ndarray]:
    target_path = target_cache_path(args, subject)
    if target_path.is_file() and not args.force_targets:
        print(f"Loading target cache: {target_path}", flush=True)
        with np.load(target_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    root = bids_root(args)
    subject_words = words
    subject_features = reference_features
    if args.max_words is not None:
        subject_words = words.iloc[: args.max_words].reset_index(drop=True)
        subject_features = reference_features[: args.max_words]

    epochs = load_epochs(
        fif_path=fif_path(root, subject, args.task),
        words=subject_words,
        picks_regex=args.picks_regex,
        max_channels=args.max_channels,
        tmin=args.lag_tmin,
        tmax=args.lag_tmax,
        resample_sfreq=args.resample_sfreq,
    )
    selection = epochs.selection.astype(np.int64)
    epoch_data = epochs.get_data(copy=True).astype(np.float32)
    x_ref = subject_features[selection].astype(np.float32)

    ref_path = reference_cache_path(args, subject)
    if ref_path.is_file() and not args.force_reference:
        print(f"Loading lag reference: {ref_path}", flush=True)
        with np.load(ref_path, allow_pickle=True) as ref:
            ref_corrs = ref["corrs"]
    else:
        print(f"Building lag reference for sub-{subject}", flush=True)
        ref_corrs = compute_gpt2_lag_reference(
            x=x_ref,
            epoch_data=epoch_data,
            alpha=args.ridge_alpha,
            outer_splits=args.outer_splits,
        )
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ref_path,
            corrs=ref_corrs.astype(np.float32),
            lags=epochs.times.copy(),
            channel_names=np.asarray(epochs.info["ch_names"]).astype(str),
            coords=np.vstack([ch["loc"][:3] for ch in epochs.info["chs"]]).astype(float) * 1000.0,
        )
        print(f"Saved lag reference: {ref_path}", flush=True)

    mean_ref = ref_corrs.mean(axis=0)
    preferred_epoch_idx = mean_ref.argmax(axis=-1).astype(np.int64)
    preferred_lag_s = epochs.times[preferred_epoch_idx].astype(np.float32)
    targets = epoch_data[:, np.arange(epoch_data.shape[1]), preferred_epoch_idx].astype(np.float32)
    payload = {
        "targets": targets,
        "selection": selection,
        "channel_names": np.asarray(epochs.info["ch_names"]).astype(str),
        "preferred_lag_s": preferred_lag_s,
        "preferred_epoch_idx": preferred_epoch_idx,
        "coords": np.vstack([ch["loc"][:3] for ch in epochs.info["chs"]]).astype(float) * 1000.0,
        "reference_path": np.asarray(ref_path.as_posix()),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, **payload)
    print(f"Saved target cache: {target_path}", flush=True)
    return payload


def feature_dim(feature_path: Path) -> int:
    with np.load(feature_path) as data:
        return int(data["features"].shape[1])


def evaluate_feature_for_subject(
    *,
    args: argparse.Namespace,
    subject: str,
    feature_path: Path,
    targets: dict[str, np.ndarray],
) -> dict[str, object]:
    with np.load(feature_path) as data:
        features = data["features"].astype(np.float32)
    selection = targets["selection"].astype(np.int64)
    if int(selection.max()) >= len(features):
        raise ValueError(f"Selection max {selection.max()} exceeds feature rows {features.shape} in {feature_path}")
    x = features[selection]
    y = targets["targets"].astype(np.float32)
    corrs = evaluate_model_window("ridge", 0, x, y, args)
    mean_by_channel = corrs.mean(axis=0)
    return {
        "corrs": corrs.astype(np.float32),
        "mean_by_channel": mean_by_channel.astype(np.float32),
        "mean_r": float(mean_by_channel.mean()),
        "median_r": float(np.median(mean_by_channel)),
        "q75_r": float(np.quantile(mean_by_channel, 0.75)),
        "max_r": float(mean_by_channel.max()),
        "positive_share": float((mean_by_channel > 0).mean()),
    }


def load_existing_outputs(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    if args.no_append_existing:
        return [], [], {}

    summary_path = args.output_dir / "gpt2_context_layer_summary.csv"
    channel_path = args.output_dir / "gpt2_context_layer_channel_scores.csv"
    corr_path = args.output_dir / "gpt2_context_layer_corrs.npz"

    rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    corr_payloads: dict[str, np.ndarray] = {}
    if summary_path.is_file():
        rows = pd.read_csv(summary_path).to_dict("records")
        print(f"Loaded existing summary rows: {len(rows)} from {summary_path}", flush=True)
    if channel_path.is_file():
        channel_rows = pd.read_csv(channel_path).to_dict("records")
        print(f"Loaded existing channel rows: {len(channel_rows)} from {channel_path}", flush=True)
    if corr_path.is_file():
        with np.load(corr_path, allow_pickle=True) as data:
            corr_payloads = {key: data[key] for key in data.files}
        print(f"Loaded existing correlation arrays: {len(corr_payloads)} from {corr_path}", flush=True)
    return rows, channel_rows, corr_payloads


def summary_key(row: dict[str, object]) -> tuple[str, int, int]:
    return (str(row["subject"]), int(row["context_tokens"]), int(row["layer"]))


def drop_existing_combination(
    *,
    rows: list[dict[str, object]],
    channel_rows: list[dict[str, object]],
    corr_payloads: dict[str, np.ndarray],
    key: tuple[str, int, int],
    corr_key: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    subject, context_tokens, layer = key
    rows = [
        row
        for row in rows
        if not (
            str(row.get("subject")) == subject
            and int(row.get("context_tokens")) == context_tokens
            and int(row.get("layer")) == layer
        )
    ]
    channel_rows = [
        row
        for row in channel_rows
        if not (
            str(row.get("subject")) == subject
            and int(row.get("context_tokens")) == context_tokens
            and int(row.get("layer")) == layer
        )
    ]
    corr_payloads.pop(corr_key, None)
    return rows, channel_rows, corr_payloads


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    channel_rows: list[dict[str, object]],
    corr_payloads: dict[str, np.ndarray],
) -> None:
    summary = pd.DataFrame(rows).sort_values(["mean_r", "median_r"], ascending=False)
    channel_scores = pd.DataFrame(channel_rows)
    summary.to_csv(args.output_dir / "gpt2_context_layer_summary.csv", index=False)
    channel_scores.to_csv(args.output_dir / "gpt2_context_layer_channel_scores.csv", index=False)
    np.savez_compressed(args.output_dir / "gpt2_context_layer_corrs.npz", **corr_payloads)

    context_summary = (
        summary.groupby(["subject", "context_tokens"], as_index=False)
        .agg(mean_r=("mean_r", "mean"), max_mean_r=("mean_r", "max"), median_r=("median_r", "mean"))
        .sort_values(["subject", "context_tokens"])
    )
    layer_summary = (
        summary.groupby(["subject", "layer"], as_index=False)
        .agg(mean_r=("mean_r", "mean"), max_mean_r=("mean_r", "max"), median_r=("median_r", "mean"))
        .sort_values(["subject", "layer"])
    )
    context_summary.to_csv(args.output_dir / "gpt2_context_summary_by_subject.csv", index=False)
    layer_summary.to_csv(args.output_dir / "gpt2_layer_summary_by_subject.csv", index=False)

    overall = (
        summary.groupby(["context_tokens", "layer"], as_index=False)
        .agg(mean_r=("mean_r", "mean"), median_r=("median_r", "mean"), subjects=("subject", "nunique"))
        .sort_values(["mean_r", "median_r"], ascending=False)
    )
    overall.to_csv(args.output_dir / "gpt2_context_layer_overall_summary.csv", index=False)
    print("Top context/layer settings across subjects:", flush=True)
    print(overall.head(25).to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = bids_root(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    config = vars(args).copy()
    config.update({"mounted_root": str(args.mounted_root), "bids_root": str(root), "device_resolved": device, "dtype_resolved": str(dtype)})
    config_path = args.output_dir / "config.json"
    if config_path.exists() and not args.no_append_existing:
        config_path = args.output_dir / f"append_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print("Loading GPT-2 XL reference features for lag selection", flush=True)
    words, reference_features = load_words_and_gpt2_features(root, args.reference_gpt2_layer)

    targets_by_subject = {}
    for subject in args.subjects:
        t0 = time.time()
        print(f"=== Building/loading targets for sub-{subject} ===", flush=True)
        targets_by_subject[subject] = load_or_build_subject_targets(
            args=args,
            subject=subject,
            words=words,
            reference_features=reference_features,
        )
        print(f"=== Targets sub-{subject} ready in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    print(f"=== Loading {args.model} on {device} dtype={dtype} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(device)
    token_table = load_token_table(root, tokenizer)
    layers = parse_layers(args.layers, int(model.config.num_hidden_layers))
    print(f"Evaluating layers: {layers}", flush=True)

    rows, channel_rows, corr_payloads = load_existing_outputs(args)
    model_tag = safe_name(args.model)

    for context_len in args.context_token_lengths:
        t0 = time.time()
        print(f"=== Extracting/evaluating context={context_len} ===", flush=True)
        feature_paths = extract_features_for_context(
            model=model,
            tokenizer=tokenizer,
            token_table=token_table,
            model_name=args.model,
            context_len=int(context_len),
            layers=layers,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
            max_words=args.max_words,
            force=args.force_features,
        )
        for layer in layers:
            feature_path = feature_paths[layer]
            dim = feature_dim(feature_path)
            for subject in args.subjects:
                row_key = (f"sub-{subject}", int(context_len), int(layer))
                corr_key = f"{model_tag}__ctx{int(context_len)}__layer{int(layer)}__sub{subject}"
                if not args.force_eval and row_key in {summary_key(row) for row in rows}:
                    print(f"Skipping existing sub-{subject} context={context_len} layer={layer}", flush=True)
                    continue
                if args.force_eval:
                    rows, channel_rows, corr_payloads = drop_existing_combination(
                        rows=rows,
                        channel_rows=channel_rows,
                        corr_payloads=corr_payloads,
                        key=row_key,
                        corr_key=corr_key,
                    )
                print(f"Evaluating sub-{subject} context={context_len} layer={layer}", flush=True)
                result = evaluate_feature_for_subject(
                    args=args,
                    subject=subject,
                    feature_path=feature_path,
                    targets=targets_by_subject[subject],
                )
                row = {
                    "subject": f"sub-{subject}",
                    "model": args.model,
                    "context_tokens": int(context_len),
                    "layer": int(layer),
                    "feature_dim": dim,
                    "mean_r": result["mean_r"],
                    "median_r": result["median_r"],
                    "q75_r": result["q75_r"],
                    "max_r": result["max_r"],
                    "positive_share": result["positive_share"],
                    "feature_path": str(feature_path),
                }
                rows.append(row)
                channel_names = targets_by_subject[subject]["channel_names"].astype(str)
                preferred_lag_s = targets_by_subject[subject]["preferred_lag_s"].astype(float)
                coords = targets_by_subject[subject]["coords"].astype(float)
                for i, channel in enumerate(channel_names):
                    channel_rows.append(
                        {
                            "subject": f"sub-{subject}",
                            "channel": str(channel),
                            "model": args.model,
                            "context_tokens": int(context_len),
                            "layer": int(layer),
                            "r": float(result["mean_by_channel"][i]),
                            "preferred_lag_s": float(preferred_lag_s[i]),
                            "x": float(coords[i, 0]),
                            "y": float(coords[i, 1]),
                            "z": float(coords[i, 2]),
                        }
                    )
                corr_payloads[corr_key] = result["corrs"]
        write_outputs(args=args, rows=rows, channel_rows=channel_rows, corr_payloads=corr_payloads)
        print(f"=== Context {context_len} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_outputs(args=args, rows=rows, channel_rows=channel_rows, corr_payloads=corr_payloads)
    print(f"Saved sweep outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
