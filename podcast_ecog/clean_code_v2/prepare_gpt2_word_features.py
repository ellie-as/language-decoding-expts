#!/usr/bin/env python3
"""Prepare contextual GPT-2 word features for the clean decoding pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_BIDS_ROOT, DEFAULT_OUTPUT_ROOT, compute_word_pca  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--context-token-length", type=int, default=32)
    parser.add_argument("--layer", type=int, default=8, help="Hidden-state index. For GPT-2, 0 is embeddings and 1-12 are transformer blocks.")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def target_stem(model: str, context_len: int, layer: int, n_components: int) -> str:
    safe_model = model.replace("/", "-")
    return f"{safe_model}_ctx{int(context_len)}_layer{int(layer)}_pca{int(n_components)}"


def load_transcript(bids_root: Path) -> pd.DataFrame:
    path = bids_root / "stimuli" / "podcast_transcript.csv"
    if path.is_file():
        transcript = pd.read_csv(path)
    else:
        token_path = bids_root / "stimuli" / "gpt2-xl" / "transcript.tsv"
        if not token_path.is_file():
            token_path = bids_root / "stimuli" / "en_core_web_lg" / "transcript.tsv"
        if not token_path.is_file():
            raise FileNotFoundError(f"Neither {path} nor a token transcript TSV exists under {bids_root / 'stimuli'}")
        token_table = pd.read_csv(token_path, sep="\t")
        transcript = (
            token_table.groupby("word_idx", sort=True)
            .agg(word=("word", "first"), start=("start", "first"), end=("end", "last"))
            .reset_index(drop=True)
        )
    required = {"word", "start", "end"}
    missing = required.difference(transcript.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    transcript = transcript.sort_values("start").reset_index(drop=True)
    transcript.insert(0, "word_idx", transcript.index.values.astype(int))
    return transcript


def build_token_table(transcript: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []
    for _, row in transcript.iterrows():
        tokens = tokenizer.tokenize(" " + str(row["word"]))
        if not tokens:
            tokens = [tokenizer.unk_token or tokenizer.eos_token]
        for token in tokens:
            rows.append(
                {
                    "word_idx": int(row["word_idx"]),
                    "word": str(row["word"]),
                    "hftoken": token,
                    "token_id": int(tokenizer.convert_tokens_to_ids(token)),
                }
            )
    return pd.DataFrame(rows)


def build_context_matrix(token_ids: np.ndarray, context_len: int, pad_id: int) -> np.ndarray:
    seq_len = int(context_len) + 1
    data = np.full((len(token_ids), seq_len), int(pad_id), dtype=np.int64)
    for i in range(len(token_ids)):
        segment = token_ids[max(0, i - int(context_len)) : i + 1]
        data[i, -len(segment) :] = segment
    return data


def pool_token_to_word(token_features: np.ndarray, token_table: pd.DataFrame, n_words: int) -> np.ndarray:
    features = np.full((int(n_words), token_features.shape[1]), np.nan, dtype=np.float32)
    for word_idx, group in token_table.groupby("word_idx", sort=True):
        features[int(word_idx)] = token_features[group.index.to_numpy()].mean(axis=0)
    return features


def extract_gpt2_features(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    transcript = load_transcript(args.bids_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_table = build_token_table(transcript, tokenizer)
    token_ids = token_table["token_id"].to_numpy(dtype=np.int64)
    contexts = build_context_matrix(token_ids, args.context_token_length, tokenizer.pad_token_id)

    device = resolve_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    n_hidden_states = int(model.config.n_layer) + 1
    if args.layer < 0 or args.layer >= n_hidden_states:
        raise ValueError(f"--layer must be in [0, {n_hidden_states - 1}] for {args.model}, got {args.layer}")

    chunks = []
    with torch.no_grad():
        for start in range(0, len(contexts), int(args.batch_size)):
            batch = torch.tensor(contexts[start : start + int(args.batch_size)], dtype=torch.long, device=device)
            output = model(batch, output_hidden_states=True)
            values = output.hidden_states[int(args.layer)][:, -1, :].detach().float().cpu().numpy()
            chunks.append(values.astype(np.float32))
            del output, batch
    token_features = np.vstack(chunks)
    word_features = pool_token_to_word(token_features, token_table, n_words=len(transcript))
    return word_features.astype(np.float32), token_table


def main() -> int:
    args = parse_args()
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    stem = target_stem(args.model, args.context_token_length, args.layer, args.n_components)
    raw_path = args.prepared_dir / f"{stem}_raw.npy"
    pca_path = args.prepared_dir / f"{stem}.npy"
    token_table_path = args.prepared_dir / f"{stem}_tokens.csv"
    explained_path = args.prepared_dir / f"{stem}_explained_variance.csv"
    config_path = args.prepared_dir / f"{stem}_config.json"

    if pca_path.is_file() and raw_path.is_file() and not args.force:
        print(f"GPT-2 target already exists: {pca_path}", flush=True)
        return 0

    features, token_table = extract_gpt2_features(args)
    scores, explained = compute_word_pca(features, args.n_components)
    np.save(raw_path, features.astype(np.float32))
    np.save(pca_path, scores.astype(np.float32))
    token_table.to_csv(token_table_path, index=False)
    explained.to_csv(explained_path, index=False)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "context_token_length": int(args.context_token_length),
                "layer": int(args.layer),
                "n_components": int(args.n_components),
                "target_stem": stem,
                "raw_path": str(raw_path),
                "pca_path": str(pca_path),
            },
            handle,
            indent=2,
        )
    print(f"Saved raw GPT-2 word features: {raw_path} shape={features.shape}", flush=True)
    print(f"Saved PCA GPT-2 word features: {pca_path} shape={scores.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
