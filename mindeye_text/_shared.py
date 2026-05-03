"""Self-contained helpers for the MindEye-style text decoder.

These were inlined from ``27-04-expts/train_5tr_chunk_nn.py`` and
``27-04-expts/train_lagged_text_pca_mlp.py`` so ``mindeye_text/`` only depends
on the always-present top-level project modules (``run_summaries_encoding``,
``run_summary_decoding``, and ``decoding/``).

The chunk-embedding cache layout matches the legacy 27-04-expts pipeline, so
existing ``chunk5tr_text_embeddings__*.pkl`` files keep working.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from run_summary_decoding import EMBEDDING_MODELS, load_encoder, retrieval_metrics  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


log = logging.getLogger("mindeye_text._shared")

TRIM_START = 5 + config.TRIM


# ---------------------------------------------------------------------------
# Path / story / ROI resolution
# ---------------------------------------------------------------------------


def resolve_response_root(args: argparse.Namespace) -> tuple[str, Path | None]:
    """Mirror of ``train_lagged_text_pca_mlp.resolve_response_root``."""
    mounted_root = None
    if getattr(args, "local_compute_mode", False):
        try:
            mounted_root = rse.configure_local_compute_mode(args)
        except FileNotFoundError as err:
            log.warning("%s", err)
            log.warning("Falling back to local cache only.")

    local_cache_root = Path(args.local_cache_root).expanduser().resolve()
    cached_base = local_cache_root / "data_train"
    fallback = str(cached_base) if cached_base.exists() else config.DATA_TRAIN_DIR
    return fallback, mounted_root


def load_stories(args: argparse.Namespace, response_root: str) -> List[str]:
    """Mirror of ``train_lagged_text_pca_mlp.load_stories``."""
    try:
        return rse.load_story_list(args)
    except (FileNotFoundError, OSError) as err:
        log.warning("%s", err)
        subj_dir = Path(response_root) / "train_response" / args.subject
        if not subj_dir.exists():
            raise
        stories = sorted(p.stem for p in subj_dir.glob("*.hf5"))
        log.warning("Falling back to cached response stories (%d stories).", len(stories))
        return stories


def resolve_roi(args: argparse.Namespace, total_voxels: int) -> tuple[str, np.ndarray]:
    """Mirror of ``train_lagged_text_pca_mlp.resolve_roi``."""
    roi = args.roi
    if roi in {"all", "whole", "all_voxels"}:
        return "all", np.arange(total_voxels, dtype=int)

    uts_id = rse.SUBJECT_TO_UTS.get(args.subject)
    if not uts_id:
        raise ValueError(f"Unknown subject {args.subject!r}")

    if roi == "full_frontal":
        roi_path = Path(args.ba_dir) / uts_id / "BA_full_frontal.json"
        roi_name = "full_frontal"
    else:
        roi_name = roi if roi.startswith("BA_") or roi == "BROCA" else f"BA_{roi}"
        roi_path = Path(args.ba_dir) / uts_id / f"{roi_name}.json"

    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    with open(roi_path, encoding="utf-8") as f:
        vox = np.asarray(next(iter(json.load(f).values())), dtype=int)
    vox = np.sort(vox[vox < total_voxels])
    if vox.size == 0:
        raise ValueError(f"ROI {roi_name} has no voxels under total voxel count {total_voxels}")
    return roi_name, vox


# ---------------------------------------------------------------------------
# Train / val grouped split, torch device, cosine helper
# ---------------------------------------------------------------------------


def grouped_train_val_split(groups: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Story-grouped split, mirror of the 27-04-expts helper."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_groups = set(unique[:n_val].tolist())
    val_mask = np.asarray([g in val_groups for g in groups])
    train_idx = np.nonzero(~val_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    if train_idx.size == 0:
        perm = rng.permutation(len(groups))
        n_val_rows = max(1, int(round(len(groups) * val_frac)))
        return perm[n_val_rows:], perm[:n_val_rows]
    log.info("Validation stories: %s", ", ".join(sorted(str(g) for g in val_groups)))
    return train_idx, val_idx


def resolve_torch_device(pref: str):
    import torch

    if pref == "cuda":
        return torch.device("cuda")
    if pref == "mps":
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_cosine(true: np.ndarray, pred: np.ndarray) -> float:
    denom = np.linalg.norm(true, axis=1) * np.linalg.norm(pred, axis=1)
    cos = np.divide((true * pred).sum(axis=1), denom, out=np.zeros(true.shape[0], dtype=np.float32), where=denom != 0)
    return float(np.nan_to_num(cos).mean())


# ---------------------------------------------------------------------------
# Response loading
# ---------------------------------------------------------------------------


def load_responses_by_story(
    stories: List[str],
    subject: str,
    vox: np.ndarray,
    response_root: str,
) -> Dict[str, np.ndarray]:
    return {
        story: get_resp(subject, [story], stack=True, vox=vox, response_root=response_root).astype(np.float32)
        for story in stories
    }


# ---------------------------------------------------------------------------
# 5-TR chunk text-embedding cache (compatible with 27-04-expts/cache layout)
# ---------------------------------------------------------------------------


def chunk_cache_key(
    subject: str,
    stories: List[str],
    feature_model: str,
    chunk_trs: int,
    lag_trs: int,
    embedding_model: str | None = None,
) -> str:
    payload = {
        "subject": subject,
        "stories": stories,
        "feature_model": feature_model,
        "chunk_trs": int(chunk_trs),
        "lag_trs": int(lag_trs),
        "alignment": "text_at_i_brain_at_i_plus_lag_v2",
    }
    if embedding_model is not None:
        payload["embedding_model"] = embedding_model
        payload["version"] = 2
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def ensure_stimulus_data_root(response_root: str, raise_if_missing: bool = True) -> bool:
    """Point TextGrid loading at the resolved data root when available."""
    current = Path(config.DATA_TRAIN_DIR)
    if (current / "train_stimulus").is_dir():
        return True

    candidate = Path(response_root)
    if (candidate / "train_stimulus").is_dir():
        config.DATA_TRAIN_DIR = str(candidate)
        return True

    if raise_if_missing:
        raise FileNotFoundError(
            "Cannot build 5-TR text chunks because train_stimulus/*.TextGrid is missing. "
            f"Checked {current / 'train_stimulus'} and {candidate / 'train_stimulus'}. "
            "Mount or copy the full data_train directory, not just train_response."
        )
    return False


def text_for_tr_chunk(wordseq, response_start: int, chunk_trs: int) -> str:
    """Transcript words inside a response-aligned TR chunk."""
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    words = np.asarray(wordseq.data)
    tr = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 2.0

    stim_start = TRIM_START + response_start
    stim_stop = stim_start + chunk_trs - 1
    if stim_stop >= len(tr_times):
        raise ValueError(f"TR chunk stop {stim_stop} exceeds wordseq TR count {len(tr_times)}")

    start_t = tr_times[stim_start] - tr / 2.0
    stop_t = tr_times[stim_stop] + tr / 2.0
    mask = (word_times >= start_t) & (word_times < stop_t)
    return " ".join(str(w).strip() for w in words[mask] if str(w).strip())


def build_chunk_texts_by_story(
    stories: List[str],
    resp_lengths: Dict[str, int],
    chunk_trs: int,
    lag_trs: int,
) -> Dict[str, List[str]]:
    wordseqs = get_story_wordseqs(stories)
    texts_by_story: Dict[str, List[str]] = {}
    for story in stories:
        n_chunks = resp_lengths[story] - lag_trs - chunk_trs + 1
        if n_chunks <= 0:
            raise ValueError(
                f"{story}: response length {resp_lengths[story]} is too short for "
                f"chunk_trs={chunk_trs}, lag_trs={lag_trs}"
            )
        texts_by_story[story] = [
            text_for_tr_chunk(wordseqs[story], start, chunk_trs)
            for start in range(n_chunks)
        ]
    return texts_by_story


def embed_chunk_texts(
    encoder,
    emb_dim: int,
    stories: List[str],
    texts_by_story: Dict[str, List[str]],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    flat_texts: List[str] = []
    story_slices: Dict[str, tuple[int, int]] = {}
    cursor = 0
    for story in stories:
        texts = texts_by_story[story]
        flat_texts.extend(texts)
        story_slices[story] = (cursor, cursor + len(texts))
        cursor += len(texts)

    out = np.zeros((len(flat_texts), emb_dim), dtype=np.float32)
    nonempty = [i for i, text in enumerate(flat_texts) if text.strip()]
    if nonempty:
        enc = encoder.encode(
            [flat_texts[i] for i in nonempty],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        out[nonempty] = enc

    return {
        story: out[start:end].astype(np.float32)
        for story, (start, end) in story_slices.items()
    }


def load_or_build_chunk_embeddings(
    args: argparse.Namespace,
    stories: List[str],
    resp_lengths: Dict[str, int],
    response_root: str | None = None,
) -> tuple[Dict[str, np.ndarray], int, str]:
    """Load (or build) the per-subject 5-TR text embedding cache.

    Cache file path matches the layout produced by ``27-04-expts/train_5tr_chunk_nn.py``,
    so existing files are picked up here without rebuilding.
    """
    cache_dir = Path(args.embedding_cache_dir) / args.subject
    cache_dir.mkdir(parents=True, exist_ok=True)
    explicit_embedding_model = getattr(args, "embedding_model", None)
    key = chunk_cache_key(
        args.subject,
        stories,
        args.feature_model,
        args.chunk_trs,
        args.lag_trs,
        explicit_embedding_model,
    )
    cache_path = cache_dir / (
        f"chunk{args.chunk_trs}tr_text_embeddings__{args.feature_model}"
        f"__lag{args.lag_trs}__{key}.pkl"
    )

    if cache_path.is_file():
        log.info("Loading cached chunk embeddings: %s", cache_path)
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload["embeddings_by_story"], int(payload["embedding_dim"]), str(cache_path)

    if response_root is not None:
        ensure_stimulus_data_root(response_root)
    log.info("Building 5-TR chunk texts and embeddings")
    texts_by_story = build_chunk_texts_by_story(stories, resp_lengths, args.chunk_trs, args.lag_trs)
    emb_model_name = explicit_embedding_model
    if not emb_model_name:
        emb_model_name, _ = EMBEDDING_MODELS[args.feature_model]
    encoder, emb_dim = load_encoder(
        emb_model_name,
        device=getattr(args, "embedding_device", "cpu"),
    )
    embeddings_by_story = embed_chunk_texts(
        encoder,
        emb_dim,
        stories,
        texts_by_story,
        args.embed_batch_size,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "embeddings_by_story": embeddings_by_story,
                "embedding_model": emb_model_name,
                "embedding_dim": emb_dim,
                "stories": stories,
                "chunk_trs": args.chunk_trs,
                "lag_trs": args.lag_trs,
            },
            f,
        )
    log.info("Wrote chunk embedding cache: %s", cache_path)
    return embeddings_by_story, emb_dim, str(cache_path)
