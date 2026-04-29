#!/usr/bin/env python3
"""Paper-style encoding comparison for finetuned vs pretrained GPT-1.

The finetuned condition mirrors ``decoding/train_EM.py`` exactly, using the
local Huth/Tang ``data_lm/perceived`` checkpoint and word-level vocabulary.

The pretrained condition uses size-matched Hugging Face ``openai-gpt`` with the
Hugging Face tokenizer. Word-level features are aligned by taking the hidden
state at the final BPE token of the current stimulus word.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
from GPT import GPT  # noqa: E402
from StimulusModel import LMFeatures  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_ridge.ridge import bootstrap_ridge, ridge  # noqa: E402
from utils_stim import get_stim  # noqa: E402


np.random.seed(42)


DEFAULT_SESSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
CONDITION_FINETUNED = "finetuned"
CONDITION_PRETRAINED = "pretrained"


class HuthFinetunedGPT1Features:
    """Feature adapter for the local Huth/Tang GPT-1 checkpoint."""

    def __init__(self, checkpoint: str, device: str):
        vocab_path = Path(config.DATA_LM_DIR) / checkpoint / "vocab.json"
        model_path = Path(config.DATA_LM_DIR) / checkpoint / "model"
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        self.gpt = GPT(path=str(model_path), vocab=vocab, device=device)
        self.features = LMFeatures(
            model=self.gpt,
            layer=config.GPT_LAYER,
            context_words=config.GPT_WORDS,
        )
        self.model_name = checkpoint
        self.layer = config.GPT_LAYER
        self.context_words = config.GPT_WORDS
        self.hidden_size = int(getattr(self.gpt.model.config, "n_embd", 0))

    def make_stim(self, words):
        return self.features.make_stim(words).astype(np.float32, copy=False)

    def extend(self, extensions, verbose=False):
        return self.features.extend(extensions, verbose=verbose)

    def close(self):
        del self.features
        del self.gpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFOpenAIGPTFeatures:
    """Word-aligned layer features for Hugging Face ``openai-gpt``."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.layer = config.GPT_LAYER
        self.context_words = config.GPT_WORDS
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
        self.hidden_size = int(self.model.config.n_embd)

    def _pad_token_id(self):
        return (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.unk_token_id
            if self.tokenizer.unk_token_id is not None
            else 0
        )

    def _encode_context(self, words):
        """Tokenize a word window and return token ids plus current-word index."""
        token_ids = []
        for word_index, word in enumerate(words):
            clean = str(word).strip()
            if not clean:
                continue
            piece = (" " + clean) if token_ids else clean
            ids = self.tokenizer.encode(piece, add_special_tokens=False)
            token_ids.extend(ids)

        if not token_ids:
            unk = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
            return [unk], 0

        max_positions = int(
            getattr(
                self.model.config,
                "n_positions",
                getattr(self.model.config, "max_position_embeddings", 512),
            )
        )
        if len(token_ids) > max_positions:
            token_ids = token_ids[-max_positions:]
        return token_ids, len(token_ids) - 1

    def make_stim(self, words):
        """Return one layer-9 vector per stimulus word."""
        words = [str(word) for word in words]
        contexts = []
        for word_index in range(len(words)):
            start = max(0, word_index - self.context_words)
            contexts.append(self._encode_context(words[start:word_index + 1]))

        vecs = np.zeros((len(contexts), self.hidden_size), dtype=np.float32)
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start:start + self.batch_size]
            max_len = max(len(ids) for ids, _cur_start in batch)
            input_ids = np.full(
                (len(batch), max_len),
                self._pad_token_id(),
                dtype=np.int64,
            )
            attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            last_indices = []

            for row, (ids, cur_start) in enumerate(batch):
                input_ids[row, :len(ids)] = ids
                attention_mask[row, :len(ids)] = 1
                last_indices.append(max(cur_start, len(ids) - 1))

            input_t = torch.tensor(input_ids, device=self.device)
            mask_t = torch.tensor(attention_mask, device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_t,
                    attention_mask=mask_t,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[self.layer]
            row_idx = torch.arange(hidden.shape[0], device=self.device)
            col_idx = torch.tensor(last_indices, device=self.device)
            vecs[start:start + len(batch)] = (
                hidden[row_idx, col_idx].detach().cpu().numpy().astype(np.float32)
            )

        return vecs

    def extend(self, extensions, verbose=False):
        """Return the current-word vector for each candidate word extension."""
        contexts = [
            self._encode_context([str(word) for word in extension[-(self.context_words + 1):]])
            for extension in extensions
        ]
        if verbose:
            print(contexts)

        vecs = np.zeros((len(contexts), self.hidden_size), dtype=np.float32)
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start:start + self.batch_size]
            max_len = max(len(ids) for ids, _cur_start in batch)
            input_ids = np.full(
                (len(batch), max_len),
                self._pad_token_id(),
                dtype=np.int64,
            )
            attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            last_indices = []

            for row, (ids, cur_start) in enumerate(batch):
                input_ids[row, :len(ids)] = ids
                attention_mask[row, :len(ids)] = 1
                last_indices.append(max(cur_start, len(ids) - 1))

            input_t = torch.tensor(input_ids, device=self.device)
            mask_t = torch.tensor(attention_mask, device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_t,
                    attention_mask=mask_t,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[self.layer]
            row_idx = torch.arange(hidden.shape[0], device=self.device)
            col_idx = torch.tensor(last_indices, device=self.device)
            vecs[start:start + len(batch)] = (
                hidden[row_idx, col_idx].detach().cpu().numpy().astype(np.float32)
            )
        return vecs

    def close(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[CONDITION_FINETUNED, CONDITION_PRETRAINED],
        choices=[CONDITION_FINETUNED, CONDITION_PRETRAINED],
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        default="perceived",
        help="Local Huth/Tang checkpoint under data_lm/.",
    )
    parser.add_argument(
        "--pretrained-model",
        default="openai-gpt",
        help="Hugging Face GPT-1 model id. Default is size-matched GPT-1.",
    )
    parser.add_argument("--device", default=config.GPT_DEVICE)
    parser.add_argument("--pretrained-batch-size", type=int, default=64)
    parser.add_argument("--nboots", type=int, default=config.NBOOTS)
    parser.add_argument("--voxel-count", type=int, default=config.VOXELS)
    parser.add_argument("--skip-noise-model", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "gpt1_encoding_comparison" / "outputs"),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Optional mounted project root containing data_train/, data_lm/, "
            "data_test/, and models/. Use this when code is local but data are mounted."
        ),
    )
    parser.add_argument(
        "--local-compute-mode",
        action="store_true",
        help=(
            "Read data from --mounted-project-root / --data-root but keep outputs "
            "under this local repo."
        ),
    )
    parser.add_argument(
        "--mounted-project-root",
        default="smb://ceph-gw02.hpc.swc.ucl.ac.uk/behrens/ellie/language-decoding-expts",
        help=(
            "Mounted project root used by --local-compute-mode. SMB URLs are "
            "resolved to likely macOS /Volumes paths."
        ),
    )
    return parser.parse_args()


def resolve_mounted_root(raw_root):
    """Resolve a normal path or an SMB URL to a mounted local filesystem path."""
    raw = str(raw_root)
    if not raw.startswith("smb://"):
        return Path(raw).expanduser().resolve()

    without_scheme = raw[len("smb://"):]
    parts = without_scheme.split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse SMB URL: {raw}")
    share = parts[1]
    tail = parts[2:]
    candidates = [Path("/Volumes") / share / Path(*tail)]
    if tail:
        candidates.append(Path("/Volumes") / tail[0] / Path(*tail[1:]))
        candidates.append(Path("/Volumes") / tail[-1])
    candidates.append(Path(raw))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    formatted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "SMB URL is not mounted as a local path I can access.\n"
        f"URL: {raw}\n"
        "Tried:\n"
        f"  {formatted}\n"
        "Mount the share in Finder first, or pass --data-root with the actual "
        "/Volumes/... path."
    )


def configure_data_root(data_root):
    """Point decoding config at a mounted data mirror, if provided."""
    if data_root is None:
        return

    root = resolve_mounted_root(data_root)
    config.DATA_TRAIN_DIR = str(root / "data_train")
    config.DATA_LM_DIR = str(root / "data_lm")
    config.DATA_TEST_DIR = str(root / "data_test")
    config.MODEL_DIR = str(root / "models")

    print(f"Using mounted data root: {root}")
    print(f"  DATA_TRAIN_DIR -> {config.DATA_TRAIN_DIR}")
    print(f"  MODEL_DIR      -> {config.MODEL_DIR}")
    print(f"  DATA_LM_DIR    -> {config.DATA_LM_DIR}")
    print(f"  DATA_TEST_DIR  -> {config.DATA_TEST_DIR}")


def load_stories(sessions):
    with open(Path(config.DATA_TRAIN_DIR) / "sess_to_story.json", encoding="utf-8") as f:
        sess_to_story = json.load(f)
    stories = []
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def make_features(condition, args):
    if condition == CONDITION_FINETUNED:
        return HuthFinetunedGPT1Features(
            checkpoint=args.finetuned_checkpoint,
            device=args.device,
        )
    if condition == CONDITION_PRETRAINED:
        return HFOpenAIGPTFeatures(
            model_name=args.pretrained_model,
            device=args.device,
            batch_size=args.pretrained_batch_size,
        )
    raise ValueError(f"Unknown condition: {condition}")


def estimate_noise_model(subject, stories, features, tr_stats, vox, alphas):
    """Replicate the leave-one-story-out residual covariance from train_EM.py."""
    stim_dict = {
        story: get_stim([story], features, tr_stats=tr_stats)
        for story in stories
    }
    resp_dict = get_resp(subject, stories, stack=False, vox=vox)
    noise_model = np.zeros((len(vox), len(vox)), dtype=np.float64)

    for heldout_story in stories:
        train_stim = np.vstack(
            [stim_dict[story] for story in stories if story != heldout_story]
        )
        heldout_stim = stim_dict[heldout_story]
        train_resp = np.vstack(
            [resp_dict[story] for story in stories if story != heldout_story]
        )
        heldout_resp = resp_dict[heldout_story]
        bs_weights = ridge(train_stim, train_resp, alphas[vox])
        residuals = heldout_resp - heldout_stim.dot(bs_weights)
        story_noise = residuals.T.dot(residuals)
        noise_model += story_noise / np.diag(story_noise).mean() / len(stories)

    return noise_model.astype(np.float32, copy=False)


def train_condition(subject, stories, condition, args, out_path):
    features = make_features(condition, args)
    try:
        print(f"\n[{subject} / {condition}] extracting stimulus features")
        rstim, tr_stats, word_stats = get_stim(stories, features)
        print(f"[{subject} / {condition}] loading fMRI responses")
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
        bootstrap_corrs = bscorrs.mean(2).max(0)
        voxel_count = min(int(args.voxel_count), bootstrap_corrs.shape[0])
        vox = np.sort(np.argsort(bootstrap_corrs)[-voxel_count:])
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
            bootstrap_corrs=bootstrap_corrs,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        return {
            "condition": condition,
            "path": str(out_path),
            "model_name": features.model_name,
            "hidden_size": features.hidden_size,
            "mean_bootstrap_corr": float(bootstrap_corrs.mean()),
            "median_bootstrap_corr": float(np.median(bootstrap_corrs)),
            "max_bootstrap_corr": float(bootstrap_corrs.max()),
            "n_corr_gt_0_1": int((bootstrap_corrs > 0.1).sum()),
            "voxels": vox,
            "bootstrap_corrs": bootstrap_corrs,
        }
    finally:
        features.close()


def load_existing_condition(path):
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"].reshape(-1)[0]))
    corrs = data["bootstrap_corrs"]
    return {
        "condition": metadata["condition"],
        "path": str(path),
        "model_name": metadata["model_name"],
        "hidden_size": int(metadata["hidden_size"]),
        "mean_bootstrap_corr": float(corrs.mean()),
        "median_bootstrap_corr": float(np.median(corrs)),
        "max_bootstrap_corr": float(corrs.max()),
        "n_corr_gt_0_1": int((corrs > 0.1).sum()),
        "voxels": data["voxels"],
        "bootstrap_corrs": corrs,
    }


def compare_results(results_by_condition):
    summary = {
        condition: {
            key: value
            for key, value in result.items()
            if key not in {"voxels", "bootstrap_corrs"}
        }
        for condition, result in results_by_condition.items()
    }

    if {CONDITION_FINETUNED, CONDITION_PRETRAINED}.issubset(results_by_condition):
        fine = results_by_condition[CONDITION_FINETUNED]
        pre = results_by_condition[CONDITION_PRETRAINED]
        fine_vox = set(map(int, fine["voxels"]))
        pre_vox = set(map(int, pre["voxels"]))
        overlap = fine_vox & pre_vox
        score_corr = float(np.corrcoef(fine["bootstrap_corrs"], pre["bootstrap_corrs"])[0, 1])
        summary["comparison"] = {
            "mean_corr_delta_pretrained_minus_finetuned": (
                pre["mean_bootstrap_corr"] - fine["mean_bootstrap_corr"]
            ),
            "max_corr_delta_pretrained_minus_finetuned": (
                pre["max_bootstrap_corr"] - fine["max_bootstrap_corr"]
            ),
            "selected_voxel_overlap": len(overlap),
            "selected_voxel_overlap_fraction_of_finetuned": len(overlap) / max(1, len(fine_vox)),
            "per_voxel_bootstrap_corr_map_correlation": score_corr,
        }
    return summary


def main():
    args = parse_args()
    data_root = args.data_root
    if args.local_compute_mode and data_root is None:
        data_root = args.mounted_project_root
    configure_data_root(data_root)

    if args.pretrained_model != "openai-gpt":
        print(
            "Warning: --pretrained-model is not 'openai-gpt'. "
            "Model size may no longer match GPT-1."
        )

    stories = load_stories(args.sessions)
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        subject_dir = out_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Subject {subject}: {len(stories)} stories ===")

        results = {}
        for condition in args.conditions:
            path = subject_dir / f"encoding_model_{condition}.npz"
            if args.skip_existing and path.exists():
                print(f"[{subject} / {condition}] reusing {path}")
                result = load_existing_condition(path)
            else:
                result = train_condition(subject, stories, condition, args, path)
            results[condition] = result

        summary = compare_results(results)
        summary_path = subject_dir / "comparison_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[{subject}] wrote {summary_path}")


if __name__ == "__main__":
    main()
