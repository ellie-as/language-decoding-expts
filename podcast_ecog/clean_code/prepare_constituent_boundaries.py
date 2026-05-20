#!/usr/bin/env python3
"""Add constituency-parser boundary annotations to the clean prepared cache."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import DEFAULT_BIDS_ROOT, DEFAULT_OUTPUT_ROOT, segment_ids_from_boundary_after  # noqa: E402


DEFAULT_MAJOR_LABELS = ["NP", "VP", "PP", "ADJP", "ADVP", "SBAR", "S", "SINV", "SQ"]
PUNCT_POS = {".", ",", ":", "``", "''", "-LRB-", "-RRB-", "HYPH", "NFP"}
PUNCT_TOKENS = set(string.punctuation) | {"...", "``", "''", "“", "”", "‘", "’"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "prepared")
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--syntactic-feature-space", default="syntactic")
    parser.add_argument("--model", default="benepar_en3")
    parser.add_argument("--major-labels", nargs="+", default=DEFAULT_MAJOR_LABELS)
    parser.add_argument("--min-span-words", type=int, default=2)
    parser.add_argument("--include-sentence-final", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_token(token: object) -> str:
    text = str(token)
    if text == "nan":
        return ""
    return text.strip()


def is_punctuation(token: str, pos: str | None = None) -> bool:
    if pos in PUNCT_POS:
        return True
    stripped = token.strip()
    if not stripped:
        return True
    if stripped in PUNCT_TOKENS:
        return True
    return bool(re.fullmatch(r"\W+", stripped))


def load_syntactic_tokens(bids_root: Path, feature_space: str) -> pd.DataFrame:
    path = bids_root / "stimuli" / feature_space / "transcript.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    tokens = pd.read_csv(path, sep="\t")
    required = {"word_idx", "token"}
    missing = required.difference(tokens.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    tokens["token"] = tokens["token"].map(clean_token)
    if "pos" not in tokens.columns:
        tokens["pos"] = None
    return tokens


def iter_tree_spans(tree, start: int = 0, depth: int = 0):
    if isinstance(tree, str):
        return start + 1, []
    cursor = start
    rows = []
    for child in tree:
        cursor, child_rows = iter_tree_spans(child, cursor, depth + 1)
        rows.extend(child_rows)
    rows.append({"label": str(tree.label()), "start_token": int(start), "end_token": int(cursor), "depth": int(depth)})
    return cursor, rows


def sentence_token_rows(words: pd.DataFrame, tokens: pd.DataFrame, sentence_id: int) -> pd.DataFrame:
    word_ids = set(words.loc[words["sentence_id"] == sentence_id, "word_idx"].astype(int).tolist())
    sent_tokens = tokens[tokens["word_idx"].astype(int).isin(word_ids)].copy()
    keep = [
        not is_punctuation(token, pos)
        for token, pos in zip(sent_tokens["token"].astype(str), sent_tokens["pos"].astype(str), strict=False)
    ]
    sent_tokens = sent_tokens.loc[keep].copy()
    sent_tokens = sent_tokens[sent_tokens["token"].astype(str).str.len() > 0].copy()
    return sent_tokens.reset_index(drop=True)


def parse_sentence(parser, sent_tokens: pd.DataFrame):
    from benepar import InputSentence

    return parser.parse(InputSentence(words=sent_tokens["token"].astype(str).tolist()))


def compute_constituents(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    import benepar

    words = pd.read_csv(args.prepared_dir / "words.csv")
    tokens = load_syntactic_tokens(args.bids_root, args.syntactic_feature_space)
    parser = benepar.Parser(args.model)

    major_labels = set(args.major_labels)
    boundary_after = np.zeros(len(words), dtype=bool)
    boundary_strength = np.zeros(len(words), dtype=np.int16)
    labels_by_end: list[set[str]] = [set() for _ in range(len(words))]
    span_rows = []

    sentence_ids = sorted(words["sentence_id"].astype(int).unique())
    for sentence_id in sentence_ids:
        sent_word_rows = words[words["sentence_id"] == sentence_id]
        if sent_word_rows.empty:
            continue
        final_word_idx = int(sent_word_rows["word_idx"].max())
        sent_tokens = sentence_token_rows(words, tokens, sentence_id)
        if len(sent_tokens) < 2:
            continue
        try:
            tree = parse_sentence(parser, sent_tokens)
        except Exception as exc:
            print(f"sentence {sentence_id}: parser failed: {exc}", flush=True)
            continue

        _, spans = iter_tree_spans(tree)
        n_tokens = len(sent_tokens)
        for span in spans:
            label = span["label"]
            if label == "TOP":
                continue
            start_token = int(span["start_token"])
            end_token = int(span["end_token"])
            if end_token <= start_token or start_token < 0 or end_token > n_tokens:
                continue
            token_slice = sent_tokens.iloc[start_token:end_token]
            start_word_idx = int(token_slice["word_idx"].astype(int).min())
            end_word_idx = int(token_slice["word_idx"].astype(int).max())
            n_words = int(token_slice["word_idx"].astype(int).nunique())
            full_sentence = start_token == 0 and end_token == n_tokens
            sentence_final = end_word_idx == final_word_idx
            is_major = label in major_labels and n_words >= int(args.min_span_words)
            used = bool(is_major and not full_sentence and (args.include_sentence_final or not sentence_final))
            if used:
                boundary_after[end_word_idx] = True
                boundary_strength[end_word_idx] += 1
                labels_by_end[end_word_idx].add(label)
            span_rows.append(
                {
                    "sentence_id": int(sentence_id),
                    "label": label,
                    "start_word_idx": start_word_idx,
                    "end_word_idx": end_word_idx,
                    "n_words": n_words,
                    "start_token": start_token,
                    "end_token": end_token,
                    "n_tokens": int(end_token - start_token),
                    "depth": int(span["depth"]),
                    "full_sentence": bool(full_sentence),
                    "sentence_final": bool(sentence_final),
                    "is_major": bool(is_major),
                    "used_as_boundary": used,
                    "text": " ".join(token_slice["token"].astype(str).tolist()),
                }
            )
        if sentence_id % 25 == 0:
            print(f"parsed sentence {sentence_id}/{sentence_ids[-1]}", flush=True)

    words = words.copy()
    words["constituent_boundary_after"] = boundary_after
    words["constituent_boundary_strength"] = boundary_strength
    words["constituent_boundary_labels"] = [";".join(sorted(labels)) for labels in labels_by_end]
    words["constituent_id"] = segment_ids_from_boundary_after(boundary_after)
    spans = pd.DataFrame(span_rows)
    return words, spans


def update_boundary_table(prepared_dir: Path, words: pd.DataFrame) -> None:
    path = prepared_dir / "boundaries.csv"
    if path.is_file():
        existing = pd.read_csv(path)
        existing = existing[existing["level"] != "constituent"].copy()
    else:
        existing = pd.DataFrame(columns=["level", "end_word_idx", "time", "word"])

    rows = []
    for _, row in words[words["constituent_boundary_after"].astype(bool)].iterrows():
        rows.append(
            {
                "level": "constituent",
                "end_word_idx": int(row["word_idx"]),
                "time": float(row["end"]),
                "word": str(row["word"]),
                "boundary_strength": int(row["constituent_boundary_strength"]),
                "labels": str(row["constituent_boundary_labels"]),
            }
        )
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    words_path = args.prepared_dir / "words.csv"
    spans_path = args.prepared_dir / "constituent_spans.csv"
    metadata_path = args.prepared_dir / "constituent_boundary_config.json"
    if not words_path.is_file():
        raise FileNotFoundError(words_path)
    if spans_path.is_file() and not args.overwrite:
        print(f"Constituent spans already exist: {spans_path}", flush=True)
        print("Use --overwrite to recompute.", flush=True)
        return 0

    words, spans = compute_constituents(args)
    words.to_csv(words_path, index=False)
    spans.to_csv(spans_path, index=False)
    update_boundary_table(args.prepared_dir, words)
    metadata = {
        "model": args.model,
        "major_labels": args.major_labels,
        "min_span_words": int(args.min_span_words),
        "include_sentence_final": bool(args.include_sentence_final),
        "n_constituent_boundaries": int(words["constituent_boundary_after"].sum()),
        "mean_boundary_strength_nonzero": float(words.loc[words["constituent_boundary_after"], "constituent_boundary_strength"].mean()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {len(spans)} parsed constituent spans to: {spans_path}", flush=True)
    print(f"Constituent boundaries: {metadata['n_constituent_boundaries']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
