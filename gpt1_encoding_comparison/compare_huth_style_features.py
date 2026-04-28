#!/usr/bin/env python3
"""Train Huth-style encoding models with alternate short-window text features.

This script keeps the paper-code encoding pipeline fixed:

* same perceived-speech training stories
* same word-to-TR interpolation and FIR delays via ``get_stim``
* same ``bootstrap_ridge(..., use_corr=False)`` fitting
* same top-voxel selection
* optional same leave-one-story-out noise model

Only the per-word text feature extractor changes. The default conditions are:

* ``gpt2``: GPT-2 small layer-9 hidden state at the current word's final BPE token
* ``sentence``: sentence-transformer embedding of the same short word window

Both use the Huth short linguistic window: current word plus
``config.GPT_WORDS`` previous words.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
from compare_gpt1_encoding import (  # noqa: E402
    DEFAULT_SESSIONS,
    estimate_noise_model,
    load_stories,
)
from utils_resp import get_resp  # noqa: E402
from utils_ridge.ridge import bootstrap_ridge  # noqa: E402
from utils_stim import get_stim  # noqa: E402


np.random.seed(42)

CONDITION_GPT2 = "gpt2"
CONDITION_SENTENCE = "sentence"


def configure_huggingface_downloads():
    """Use longer HF timeouts for clusters with slow egress."""
    import os

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


class GPT2ShortWindowFeatures:
    """GPT-2 small features over the same short word window as Huth GPT-1."""

    def __init__(self, model_name: str, device: str, batch_size: int, pool_current_word: bool = False):
        configure_huggingface_downloads()
        from transformers import GPT2Model, GPT2Tokenizer

        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.pool_current_word = bool(pool_current_word)
        self.layer = config.GPT_LAYER
        self.context_words = config.GPT_WORDS
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name).eval().to(device)
        self.hidden_size = int(self.model.config.n_embd)

    def _encode_context(self, words):
        token_ids = []
        current_word_start = 0
        for word_index, word in enumerate(words):
            clean = str(word).strip()
            if not clean:
                continue
            piece = (" " + clean) if token_ids else clean
            ids = self.tokenizer.encode(piece, add_special_tokens=False)
            if word_index == len(words) - 1:
                current_word_start = len(token_ids)
            token_ids.extend(ids)

        if not token_ids:
            return [self.tokenizer.eos_token_id], 0, 1

        max_positions = int(getattr(self.model.config, "n_positions", 1024))
        if len(token_ids) > max_positions:
            drop = len(token_ids) - max_positions
            token_ids = token_ids[-max_positions:]
            current_word_start = max(0, current_word_start - drop)
        current_word_stop = len(token_ids)
        return token_ids, current_word_start, current_word_stop

    def _embed_contexts(self, contexts):
        vecs = np.zeros((len(contexts), self.hidden_size), dtype=np.float32)
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start:start + self.batch_size]
            max_len = max(len(ids) for ids, _cur_start, _cur_stop in batch)
            input_ids = np.full(
                (len(batch), max_len),
                self.tokenizer.pad_token_id,
                dtype=np.int64,
            )
            attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            spans = []
            for row, (ids, cur_start, cur_stop) in enumerate(batch):
                input_ids[row, :len(ids)] = ids
                attention_mask[row, :len(ids)] = 1
                spans.append((cur_start, cur_stop))

            input_t = torch.tensor(input_ids, device=self.device)
            mask_t = torch.tensor(attention_mask, device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_t,
                    attention_mask=mask_t,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[self.layer]
            for row, (cur_start, cur_stop) in enumerate(spans):
                if self.pool_current_word:
                    vec = hidden[row, cur_start:cur_stop].mean(dim=0)
                else:
                    vec = hidden[row, cur_stop - 1]
                vecs[start + row] = vec.detach().cpu().numpy().astype(np.float32)
        return vecs

    def make_stim(self, words):
        words = [str(word) for word in words]
        contexts = []
        for word_index in range(len(words)):
            start = max(0, word_index - self.context_words)
            contexts.append(self._encode_context(words[start:word_index + 1]))
        return self._embed_contexts(contexts)

    def extend(self, extensions, verbose=False):
        contexts = [
            self._encode_context([str(word) for word in extension[-(self.context_words + 1):]])
            for extension in extensions
        ]
        if verbose:
            print(contexts)
        return self._embed_contexts(contexts)

    def close(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SentenceShortWindowFeatures:
    """Sentence-transformer embeddings over the same short word window."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        configure_huggingface_downloads()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.layer = -1
        self.context_words = config.GPT_WORDS
        self.model = SentenceTransformer(model_name, device=device)
        self.hidden_size = int(self.model.get_sentence_embedding_dimension())

    def _window_texts(self, words):
        words = [str(word) for word in words]
        texts = []
        for word_index in range(len(words)):
            start = max(0, word_index - self.context_words)
            texts.append(" ".join(word.strip() for word in words[start:word_index + 1] if word.strip()))
        return texts

    def make_stim(self, words):
        texts = self._window_texts(words)
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)

    def extend(self, extensions, verbose=False):
        texts = [
            " ".join(str(word).strip() for word in extension[-(self.context_words + 1):] if str(word).strip())
            for extension in extensions
        ]
        if verbose:
            print(texts)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[CONDITION_GPT2, CONDITION_SENTENCE],
        choices=[CONDITION_GPT2, CONDITION_SENTENCE],
    )
    parser.add_argument("--gpt2-model", default="openai-community/gpt2")
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=config.GPT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpt2-pool-current-word", action="store_true")
    parser.add_argument("--nboots", type=int, default=config.NBOOTS)
    parser.add_argument("--voxel-count", type=int, default=config.VOXELS)
    parser.add_argument("--skip-noise-model", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "huth_style_feature_outputs"),
    )
    return parser.parse_args()


def make_features(condition, args):
    if condition == CONDITION_GPT2:
        return GPT2ShortWindowFeatures(
            model_name=args.gpt2_model,
            device=args.device,
            batch_size=args.batch_size,
            pool_current_word=args.gpt2_pool_current_word,
        )
    if condition == CONDITION_SENTENCE:
        return SentenceShortWindowFeatures(
            model_name=args.sentence_model,
            device=args.device,
            batch_size=args.batch_size,
        )
    raise ValueError(f"Unknown condition: {condition}")


def train_condition(subject, stories, condition, args, out_path):
    features = make_features(condition, args)
    try:
        print(f"\n[{subject} / {condition}] extracting short-window features")
        rstim, tr_stats, word_stats = get_stim(stories, features)
        rresp = get_resp(subject, stories, stack=True)
        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))

        print(
            f"[{subject} / {condition}] bootstrap ridge: "
            f"stim={rstim.shape}, resp={rresp.shape}, nboots={args.nboots}"
        )
        weights, alphas, bscorrs = bootstrap_ridge(
            rstim,
            rresp,
            use_corr=False,
            alphas=config.ALPHAS,
            nboots=args.nboots,
            chunklen=config.CHUNKLEN,
            nchunks=nchunks,
        )
        bootstrap_scores = bscorrs.mean(2).max(0)
        voxel_count = min(int(args.voxel_count), bootstrap_scores.shape[0])
        vox = np.sort(np.argsort(bootstrap_scores)[-voxel_count:])
        del rstim, rresp, bscorrs

        if args.skip_noise_model:
            noise_model = np.array([], dtype=np.float32)
        else:
            print(f"[{subject} / {condition}] estimating noise model")
            noise_model = estimate_noise_model(
                subject=subject,
                stories=stories,
                features=features,
                tr_stats=tr_stats,
                vox=vox,
                alphas=alphas,
            )

        metadata = {
            "condition": condition,
            "subject": subject,
            "model_name": features.model_name,
            "hidden_size": features.hidden_size,
            "layer": features.layer,
            "context_words": features.context_words,
            "sessions": args.sessions,
            "nboots": args.nboots,
            "voxel_count": voxel_count,
            "skip_noise_model": args.skip_noise_model,
            "huth_style_short_window": True,
        }
        np.savez(
            out_path,
            weights=weights,
            noise_model=noise_model,
            alphas=alphas,
            voxels=vox,
            stories=np.array(stories),
            tr_stats=np.array(tr_stats, dtype=object),
            word_stats=np.array(word_stats, dtype=object),
            bootstrap_scores=bootstrap_scores,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        return {
            "condition": condition,
            "path": str(out_path),
            "model_name": features.model_name,
            "hidden_size": features.hidden_size,
            "mean_bootstrap_score": float(bootstrap_scores.mean()),
            "median_bootstrap_score": float(np.median(bootstrap_scores)),
            "max_bootstrap_score": float(bootstrap_scores.max()),
            "n_score_gt_0_1": int((bootstrap_scores > 0.1).sum()),
            "voxels": vox,
            "bootstrap_scores": bootstrap_scores,
        }
    finally:
        features.close()


def load_existing_condition(path):
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"].reshape(-1)[0]))
    scores = data["bootstrap_scores"]
    return {
        "condition": metadata["condition"],
        "path": str(path),
        "model_name": metadata["model_name"],
        "hidden_size": int(metadata["hidden_size"]),
        "mean_bootstrap_score": float(scores.mean()),
        "median_bootstrap_score": float(np.median(scores)),
        "max_bootstrap_score": float(scores.max()),
        "n_score_gt_0_1": int((scores > 0.1).sum()),
        "voxels": data["voxels"],
        "bootstrap_scores": scores,
    }


def summarize(results_by_condition):
    summary = {
        condition: {
            key: value
            for key, value in result.items()
            if key not in {"voxels", "bootstrap_scores"}
        }
        for condition, result in results_by_condition.items()
    }

    conditions = sorted(results_by_condition)
    comparisons = {}
    for index, left in enumerate(conditions):
        for right in conditions[index + 1:]:
            lres = results_by_condition[left]
            rres = results_by_condition[right]
            lvox = set(map(int, lres["voxels"]))
            rvox = set(map(int, rres["voxels"]))
            overlap = lvox & rvox
            comparisons[f"{left}_vs_{right}"] = {
                "mean_delta_right_minus_left": (
                    rres["mean_bootstrap_score"] - lres["mean_bootstrap_score"]
                ),
                "max_delta_right_minus_left": (
                    rres["max_bootstrap_score"] - lres["max_bootstrap_score"]
                ),
                "selected_voxel_overlap": len(overlap),
                "selected_voxel_overlap_fraction_of_left": len(overlap) / max(1, len(lvox)),
                "per_voxel_score_map_correlation": float(
                    np.corrcoef(lres["bootstrap_scores"], rres["bootstrap_scores"])[0, 1]
                ),
            }
    if comparisons:
        summary["comparisons"] = comparisons
    return summary


def main():
    args = parse_args()
    config.GPT_DEVICE = args.device
    config.EM_DEVICE = args.device
    config.SM_DEVICE = args.device

    stories = load_stories(args.sessions)
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        subject_dir = out_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Subject {subject}: {len(stories)} stories ===")

        results = {}
        for condition in args.conditions:
            suffix = condition
            if condition == CONDITION_GPT2 and args.gpt2_pool_current_word:
                suffix = "gpt2_pool_current_word"
            out_path = subject_dir / f"encoding_model_{suffix}.npz"
            if args.skip_existing and out_path.exists():
                print(f"[{subject} / {condition}] reusing {out_path}")
                result = load_existing_condition(out_path)
            else:
                result = train_condition(subject, stories, condition, args, out_path)
            results[suffix] = result

        summary = summarize(results)
        summary_path = subject_dir / "huth_style_feature_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[{subject}] wrote {summary_path}")


if __name__ == "__main__":
    main()
