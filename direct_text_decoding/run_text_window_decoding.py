#!/usr/bin/env python3
"""Decode embeddings of the actual recently heard story text from brain data.

This is intentionally separate from the summary-decoding scripts. For each
response TR, it builds a target string from the words in the previous N stimulus
TRs (optionally shifted by a BOLD/text lag), embeds that raw text with a
sentence-transformer, and trains brain->text-embedding decoders.

The main sweep axes are:
  - --window-trs: how many recent stimulus TRs of text to include
  - --target-lags: how far before the current response TR the text window ends

Example:
    python -m direct_text_decoding.run_text_window_decoding \
        --subject S1 --roi full_frontal \
        --window-trs 1 2 3 5 8 10 \
        --target-lags 1 2 3 4 \
        --target-pca 50 --brain-pca 512 \
        --skip-sklearn --loss infonce
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from run_h20_decoder_sweep import (  # noqa: E402
    dim_r,
    pls_fit_predict,
    ridge_fit_predict,
    ridge_rankk_predict_from_model,
    story_retrieval_metrics,
    torch_fit_predict,
)
from run_summary_decoding import (  # noqa: E402
    EMBEDDING_MODELS,
    TRIM_END,
    TRIM_START,
    load_encoder,
    retrieval_metrics,
)
from utils_resp import get_resp  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("direct_text_decoding")

WORD_RE = re.compile(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$")


def _safe_text(words) -> str:
    text = " ".join(str(w).strip() for w in words if str(w).strip())
    return text if text else ""


def normalize_word(word: str) -> str:
    """Normalize TextGrid tokens for lookup in word-vector vocabularies."""
    return WORD_RE.sub("", str(word).lower()).strip()


def build_recent_texts_for_story(wordseq, response_len: int, window_trs: int, target_lag: int):
    """Build one raw-text window for every response TR in a story.

    Response files in this repo are already trimmed to align with stimulus TRs
    `TRIM_START : -TRIM_END`. For response row i, the corresponding untrimmed
    stimulus TR index is `TRIM_START + i`. `target_lag` shifts the text window
    earlier, so lag=3 means the window ends three TRs before the response row.
    """
    words = np.asarray(wordseq.data)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
    if len(tr_times) < TRIM_START + response_len:
        raise ValueError(
            f"wordseq has only {len(tr_times)} TRs but response_len={response_len} "
            f"requires at least {TRIM_START + response_len}"
        )

    # Usually 2s, but infer it from the data rather than hard-coding.
    tr = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 2.0
    half_tr = tr / 2.0
    texts = []
    for i in range(response_len):
        stim_idx = TRIM_START + i - int(target_lag)
        if stim_idx < 0:
            texts.append("")
            continue
        end_t = tr_times[stim_idx] + half_tr
        start_t = end_t - float(window_trs) * tr
        mask = (word_times >= start_t) & (word_times < end_t)
        texts.append(_safe_text(words[mask]))
    return texts


def embed_texts(model, texts, emb_dim: int, batch_size: int):
    """Embed possibly-empty text windows; empty windows get an all-zero vector."""
    out = np.zeros((len(texts), emb_dim), dtype=np.float32)
    idx = [i for i, t in enumerate(texts) if t.strip()]
    if not idx:
        return out
    enc = model.encode(
        [texts[i] for i in idx],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    out[idx] = enc
    return out


def collect_story_vocab(wordseqs, stories):
    vocab = set()
    for story in stories:
        for word in wordseqs[story].data:
            w = normalize_word(word)
            if w:
                vocab.add(w)
    return vocab


def load_word_vectors_for_vocab(path, needed_vocab):
    """Load only needed word vectors from a text .vec/.txt embedding file.

    Supports standard word2vec/GloVe text format. If the first line looks like
    a word2vec header ("N D"), it is skipped.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Word-vector file not found: {path}")
    needed_vocab = set(needed_vocab)
    vectors = {}
    dim = None
    log.info("Loading word vectors from %s for %d needed words", path, len(needed_vocab))
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip().split()
            if not parts:
                continue
            if line_no == 1 and len(parts) == 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    continue
                except ValueError:
                    pass
            word = normalize_word(parts[0])
            if word not in needed_vocab:
                continue
            try:
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            if dim is None:
                dim = int(vec.shape[0])
            if vec.shape[0] != dim:
                continue
            vectors[word] = vec
            if len(vectors) == len(needed_vocab):
                break
    if not vectors:
        raise ValueError(f"No matching word vectors found in {path}")
    log.info(
        "Loaded %d/%d needed word vectors (dim=%d)",
        len(vectors), len(needed_vocab), dim,
    )
    return vectors, int(dim)


def load_gensim_vectors_for_vocab(model_name, needed_vocab):
    """Load vectors via gensim.downloader, keeping only story words."""
    try:
        import gensim.downloader as api
    except Exception as e:
        raise RuntimeError(
            "gensim is not installed. Install gensim or pass --word-vector-path "
            "to a local GloVe/word2vec text file."
        ) from e
    log.info("Loading gensim word-vector model: %s", model_name)
    kv = api.load(model_name)
    vectors = {}
    for word in needed_vocab:
        if word in kv:
            vectors[word] = np.asarray(kv[word], dtype=np.float32)
    if not vectors:
        raise ValueError(f"No story words found in gensim model {model_name}")
    dim = int(next(iter(vectors.values())).shape[0])
    log.info("Loaded %d/%d needed word vectors (dim=%d)", len(vectors), len(needed_vocab), dim)
    return vectors, dim


def load_word_vectors(args, wordseqs, stories):
    needed = collect_story_vocab(wordseqs, stories)
    if args.word_vector_path:
        return load_word_vectors_for_vocab(args.word_vector_path, needed)
    return load_gensim_vectors_for_vocab(args.word_vector_model, needed)


def build_text_targets(
    stories,
    wordseqs,
    resp_lengths,
    encoder,
    emb_dim: int,
    window_trs: int,
    target_lag: int,
    batch_size: int,
):
    """Return stacked target embeddings and text strings for the requested stories."""
    Ys = []
    all_texts = []
    groups = []
    for story in stories:
        texts = build_recent_texts_for_story(
            wordseqs[story],
            response_len=resp_lengths[story],
            window_trs=window_trs,
            target_lag=target_lag,
        )
        y = embed_texts(encoder, texts, emb_dim, batch_size=batch_size)
        Ys.append(y)
        all_texts.extend(texts)
        groups.extend([story] * len(texts))
    return np.vstack(Ys).astype(np.float32), all_texts, np.asarray(groups)


def build_word2vec_mean_targets_for_story(
    wordseq,
    response_len: int,
    window_trs: int,
    target_lag: int,
    word_vectors,
    emb_dim: int,
):
    """Mean word-vector target over words in the recent text window."""
    words = np.asarray(wordseq.data)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
    if len(tr_times) < TRIM_START + response_len:
        raise ValueError(
            f"wordseq has only {len(tr_times)} TRs but response_len={response_len} "
            f"requires at least {TRIM_START + response_len}"
        )
    tr = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 2.0
    half_tr = tr / 2.0

    # Pre-normalize lookup to avoid doing string work inside every window.
    per_word = []
    for w in words:
        vec = word_vectors.get(normalize_word(w))
        per_word.append(vec)

    targets = np.zeros((response_len, emb_dim), dtype=np.float32)
    texts = []
    for i in range(response_len):
        stim_idx = TRIM_START + i - int(target_lag)
        if stim_idx < 0:
            texts.append("")
            continue
        end_t = tr_times[stim_idx] + half_tr
        start_t = end_t - float(window_trs) * tr
        idx = np.nonzero((word_times >= start_t) & (word_times < end_t))[0]
        vecs = [per_word[j] for j in idx if per_word[j] is not None]
        if vecs:
            targets[i] = np.mean(vecs, axis=0).astype(np.float32)
        texts.append(_safe_text(words[idx]))
    return targets, texts


def build_word2vec_mean_targets(
    stories,
    wordseqs,
    resp_lengths,
    word_vectors,
    emb_dim: int,
    window_trs: int,
    target_lag: int,
):
    Ys = []
    all_texts = []
    groups = []
    for story in stories:
        y, texts = build_word2vec_mean_targets_for_story(
            wordseqs[story],
            response_len=resp_lengths[story],
            window_trs=window_trs,
            target_lag=target_lag,
            word_vectors=word_vectors,
            emb_dim=emb_dim,
        )
        Ys.append(y)
        all_texts.extend(texts)
        groups.extend([story] * y.shape[0])
    return np.vstack(Ys).astype(np.float32), all_texts, np.asarray(groups)


def zscore_train_test(train, test):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1
    return (
        np.nan_to_num((train - mu) / sd).astype(np.float32),
        np.nan_to_num((test - mu) / sd).astype(np.float32),
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def load_roi_voxels(args, total_voxels: int):
    """Resolve --roi to voxel indices."""
    roi_name = args.roi
    if roi_name in ("all", "whole", "all_voxels"):
        return "all", np.arange(total_voxels, dtype=int)

    uts_id = rse.SUBJECT_TO_UTS.get(args.subject)
    if not uts_id:
        raise ValueError(f"Unknown subject {args.subject}")
    if roi_name == "full_frontal":
        roi_json = Path(args.ba_dir) / uts_id / "BA_full_frontal.json"
    else:
        base = roi_name if roi_name.startswith("BA_") else f"BA_{roi_name}"
        roi_json = Path(args.ba_dir) / uts_id / f"{base}.json"
        roi_name = base
    if not roi_json.is_file():
        raise FileNotFoundError(f"ROI file not found: {roi_json}")
    with open(roi_json, encoding="utf-8") as f:
        roi_data = json.load(f)
    vox = np.sort(np.asarray(list(roi_data.values())[0], dtype=int))
    vox = vox[vox < total_voxels]
    return roi_name, vox


def build_brain_and_groups(subject: str, stories, vox, response_root, resp_lengths):
    Xs = []
    groups = []
    for story in stories:
        x = get_resp(subject, [story], stack=True, vox=vox, response_root=response_root).astype(np.float32)
        if x.shape[0] != resp_lengths[story]:
            raise ValueError(f"{story}: brain TRs {x.shape[0]} != expected {resp_lengths[story]}")
        Xs.append(x)
        groups.extend([story] * x.shape[0])
    return np.vstack(Xs).astype(np.float32), np.asarray(groups)


def maybe_pca(train, test, k: int, label: str):
    if not k or k <= 0:
        return train, test, None
    from sklearn.decomposition import PCA

    k_eff = int(min(k, train.shape[0] - 1, train.shape[1]))
    log.info("Applying %s PCA: %s -> %d", label, train.shape, k_eff)
    pca = PCA(n_components=k_eff, svd_solver="randomized", random_state=0)
    train_p = pca.fit_transform(train).astype(np.float32)
    test_p = pca.transform(test).astype(np.float32)
    log.info("  %s PCA explained variance: %.3f", label, float(pca.explained_variance_ratio_.sum()))
    return train_p, test_p, pca


def paragraph_retrieval_metrics(true_emb, pred_emb, groups, window_trs: int = 15, stride_trs: int = 5):
    """Retrieval after averaging embeddings over sliding windows within each story."""
    true_chunks = []
    pred_chunks = []
    chunk_groups = []
    groups = np.asarray(groups)
    for story in np.unique(groups):
        idx = np.nonzero(groups == story)[0]
        if len(idx) < window_trs:
            continue
        for start in range(0, len(idx) - window_trs + 1, stride_trs):
            sel = idx[start:start + window_trs]
            true_chunks.append(true_emb[sel].mean(axis=0))
            pred_chunks.append(pred_emb[sel].mean(axis=0))
            chunk_groups.append(story)
    if not true_chunks:
        return {
            "paragraph_top1": np.nan,
            "paragraph_mrr": np.nan,
            "paragraph_mean_rank": np.nan,
            "paragraph_story_top1": np.nan,
            "paragraph_n": 0,
        }
    true_chunks = np.vstack(true_chunks).astype(np.float32)
    pred_chunks = np.vstack(pred_chunks).astype(np.float32)
    chunk_groups = np.asarray(chunk_groups)
    top1, mrr, mean_rank = retrieval_metrics(true_chunks, pred_chunks)
    story = story_retrieval_metrics(true_chunks, pred_chunks, chunk_groups)
    return {
        "paragraph_top1": float(top1),
        "paragraph_mrr": float(mrr),
        "paragraph_mean_rank": float(mean_rank),
        "paragraph_story_top1": float(story["story_top1"]),
        "paragraph_story_mrr": float(story["story_mrr"]),
        "paragraph_argmax_in_correct": float(story["argmax_in_correct_story"]),
        "paragraph_n": int(true_chunks.shape[0]),
    }


def evaluate_predictions(true_test, pred_test, groups_test, paragraph_window, paragraph_stride):
    tr_top1, tr_mrr, tr_mean_rank = retrieval_metrics(true_test, pred_test)
    story = story_retrieval_metrics(true_test, pred_test, groups_test)
    para = paragraph_retrieval_metrics(
        true_test, pred_test, groups_test,
        window_trs=paragraph_window,
        stride_trs=paragraph_stride,
    )
    return {
        "dim_r_test": float(dim_r(true_test, pred_test)),
        "tr_top1": float(tr_top1),
        "tr_mrr": float(tr_mrr),
        "tr_mean_rank": float(tr_mean_rank),
        "story_top1": float(story["story_top1"]),
        "story_mrr": float(story["story_mrr"]),
        "story_argmax": float(story["argmax_in_correct_story"]),
        "story_mean_rank": float(story["story_mean_rank"]),
        **para,
    }


def circular_shift_within_stories(true_emb, groups, rng, min_shift: int = 25):
    """Break temporal alignment while preserving each story's target autocorrelation.

    This is a stricter null than random row shuffling: each held-out story keeps
    its internal target trajectory, but the trajectory is circularly shifted
    relative to the brain predictions. Story identity is preserved, temporal
    alignment is not.
    """
    shifted = np.empty_like(true_emb)
    groups = np.asarray(groups)
    for story in np.unique(groups):
        idx = np.nonzero(groups == story)[0]
        n = len(idx)
        if n <= 1:
            shifted[idx] = true_emb[idx]
            continue
        valid = np.arange(1, n)
        far = valid[(valid >= min_shift) & (valid <= n - min_shift)]
        if len(far) == 0:
            far = valid
        shift = int(rng.choice(far))
        shifted[idx] = np.roll(true_emb[idx], shift=shift, axis=0)
    return shifted


def add_null_metrics(row, true_test, pred_test, groups_test, args, window_trs, target_lag, model_name):
    """Attach circular-shift null statistics to one result row."""
    if args.null_iters <= 0:
        return
    seed = (
        int(args.seed)
        + 1009 * int(window_trs)
        + 9173 * int(target_lag)
        + sum(ord(c) for c in str(model_name))
    )
    rng = np.random.default_rng(seed)
    null_dim = []
    null_tr_top1 = []
    null_story = []
    null_para_story = []
    null_para_top1 = []
    for _ in range(args.null_iters):
        shifted_true = circular_shift_within_stories(
            true_test, groups_test, rng, min_shift=args.null_min_shift,
        )
        if args.null_eval == "dim":
            null_dim.append(dim_r(shifted_true, pred_test))
        else:
            met = evaluate_predictions(
                shifted_true, pred_test, groups_test,
                paragraph_window=args.paragraph_window_trs,
                paragraph_stride=args.paragraph_stride_trs,
            )
            null_dim.append(met["dim_r_test"])
            null_tr_top1.append(met["tr_top1"])
            null_story.append(met["story_top1"])
            null_para_story.append(met["paragraph_story_top1"])
            null_para_top1.append(met["paragraph_top1"])

    def summarize(prefix, observed, values):
        values = np.asarray(values, dtype=np.float64)
        row[f"{prefix}_null_mean"] = float(values.mean())
        row[f"{prefix}_null_p95"] = float(np.percentile(values, 95))
        row[f"{prefix}_null_p"] = float((1 + np.sum(values >= observed)) / (len(values) + 1))

    summarize("dim_r", row["dim_r_test"], null_dim)
    if args.null_eval == "all":
        summarize("tr_top1", row["tr_top1"], null_tr_top1)
        summarize("story_top1", row["story_top1"], null_story)
        summarize("paragraph_story_top1", row["paragraph_story_top1"], null_para_story)
        summarize("paragraph_top1", row["paragraph_top1"], null_para_top1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--stories", nargs="*", default=None)
    p.add_argument(
        "--sessions", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    p.add_argument("--holdout-stories", nargs="*", default=None)
    p.add_argument("--holdout-count", type=int, default=rse.DEFAULT_HOLDOUT_COUNT)
    p.add_argument("--no-story-holdout", action="store_true")
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    p.add_argument("--local-cache-root", default=str(rse.DEFAULT_LOCAL_CACHE_ROOT))
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument(
        "--roi",
        default="full_frontal",
        help="Voxel set: all, full_frontal, BA_10, BA_45, etc.",
    )
    p.add_argument(
        "--focused",
        action="store_true",
        help="Shortcut for the current best raw-text region: --window-trs 2 3 "
             "--target-lags 2. Explicit --window-trs/--target-lags override this.",
    )
    p.add_argument("--window-trs", nargs="+", type=int, default=None)
    p.add_argument("--target-lags", nargs="+", type=int, default=None)
    p.add_argument(
        "--target-kind",
        default="sentence",
        choices=["sentence", "word2vec_mean"],
        help="'sentence' embeds the whole text window with a sentence-transformer. "
             "'word2vec_mean' averages one static word vector per word in the window.",
    )
    p.add_argument("--feature-model", default="embedding", choices=list(EMBEDDING_MODELS.keys()))
    p.add_argument(
        "--word-vector-path",
        default=None,
        help="Path to a local GloVe/word2vec text file for --target-kind word2vec_mean. "
             "If omitted, tries gensim.downloader with --word-vector-model.",
    )
    p.add_argument(
        "--word-vector-model",
        default="glove-wiki-gigaword-300",
        help="gensim.downloader model name used when --target-kind word2vec_mean and "
             "--word-vector-path is omitted.",
    )
    p.add_argument("--embed-batch-size", type=int, default=128)
    p.add_argument("--brain-pca", type=int, default=512)
    p.add_argument("--target-pca", type=int, default=50)
    p.add_argument("--loss", default="infonce", choices=["mse", "cosine", "infonce"])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--torch-device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--skip-torch", action="store_true")
    p.add_argument("--skip-sklearn", action="store_true")
    p.add_argument(
        "--sklearn-models",
        nargs="+",
        default=["ridge", "ridge_rankk", "pls"],
        choices=["ridge", "ridge_rankk", "pls"],
        help="Sklearn baselines to run when --skip-sklearn is not set.",
    )
    p.add_argument("--ridge-alphas", nargs="+", type=float, default=[1e2, 1e3, 1e4, 1e5, 1e6])
    p.add_argument("--ridge-ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--pls-components", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument(
        "--null-iters",
        type=int,
        default=0,
        help="If >0, compute circular-shift null metrics for every fitted model.",
    )
    p.add_argument(
        "--null-min-shift",
        type=int,
        default=25,
        help="Minimum within-story circular shift in TRs for the null.",
    )
    p.add_argument(
        "--null-eval",
        default="dim",
        choices=["dim", "all"],
        help="'dim' computes only dim_r nulls (fast). 'all' also recomputes retrieval nulls.",
    )
    p.add_argument("--paragraph-window-trs", type=int, default=15)
    p.add_argument("--paragraph-stride-trs", type=int, default=5)
    p.add_argument("--output-dir", default="direct_text_decoding/results")
    p.add_argument("--seed", type=int, default=0)
    # Present only so rse.configure_local_compute_mode can mutate defaults.
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.window_trs is None:
        args.window_trs = [2, 3] if args.focused else [1, 2, 3, 5, 8, 10]
    if args.target_lags is None:
        args.target_lags = [2] if args.focused else [1, 2, 3, 4]
    if args.focused:
        log.info(
            "Focused sweep: window_trs=%s target_lags=%s",
            args.window_trs, args.target_lags,
        )

    mounted_root = None
    if args.local_compute_mode:
        try:
            mounted_root = rse.configure_local_compute_mode(args)
        except FileNotFoundError as e:
            log.warning("%s", e)
            log.warning("Falling back to local cache only.")

    local_cache_root = Path(args.local_cache_root).expanduser().resolve()
    cached_base = local_cache_root / "data_train"
    fallback_response_root = str(cached_base) if cached_base.exists() else config.DATA_TRAIN_DIR

    try:
        stories = rse.load_story_list(args)
    except FileNotFoundError as e:
        log.warning("%s", e)
        subj_dir = Path(fallback_response_root) / "train_response" / args.subject
        if not subj_dir.exists():
            raise
        stories = sorted(p.stem for p in subj_dir.glob("*.hf5"))
        log.warning("Falling back to story list from cached responses (%d stories).", len(stories))

    train_stories, test_stories = rse.split_story_list(stories, args)
    if not test_stories:
        raise ValueError("Need held-out stories for evaluation.")
    log.info("Train stories: %d, test stories: %d", len(train_stories), len(test_stories))

    if args.local_compute_mode and mounted_root is not None:
        response_root = rse.stage_local_response_cache(
            args.subject, stories,
            mounted_data_train_dir=config.DATA_TRAIN_DIR,
            cache_root=local_cache_root,
        )
    else:
        response_root = fallback_response_root

    train_resp_lengths, total_voxels = rse.load_resp_info(args.subject, train_stories, data_train_dir=response_root)
    test_resp_lengths, _ = rse.load_resp_info(args.subject, test_stories, data_train_dir=response_root)

    roi_name, vox = load_roi_voxels(args, total_voxels)
    log.info("Using %s voxels: %d", roi_name, len(vox))

    log.info("Loading word sequences for raw story text...")
    wordseqs = get_story_wordseqs(stories)

    encoder = None
    word_vectors = None
    if args.target_kind == "sentence":
        emb_model_name, emb_dim = EMBEDDING_MODELS[args.feature_model]
        encoder, emb_dim = load_encoder(emb_model_name, device="cpu")
        target_label = args.feature_model
    else:
        word_vectors, emb_dim = load_word_vectors(args, wordseqs, stories)
        target_label = (
            Path(args.word_vector_path).stem if args.word_vector_path
            else args.word_vector_model
        )

    log.info("Loading brain responses...")
    X_train_raw, g_train = build_brain_and_groups(
        args.subject, train_stories, vox, response_root, train_resp_lengths,
    )
    X_test_raw, g_test = build_brain_and_groups(
        args.subject, test_stories, vox, response_root, test_resp_lengths,
    )
    X_train, X_test, _, _ = zscore_train_test(X_train_raw, X_test_raw)
    del X_train_raw, X_test_raw
    X_train_model, X_test_model, _ = maybe_pca(X_train, X_test, args.brain_pca, "brain")
    del X_train, X_test

    from run_h20_decoder_sweep import _resolve_torch_device

    device = _resolve_torch_device(args.torch_device)
    log.info("Torch device: %s", device)

    results = []
    for window_trs in args.window_trs:
        for target_lag in args.target_lags:
            log.info("=" * 72)
            log.info("Target: previous %d TRs of raw text, lag=%d", window_trs, target_lag)
            log.info("=" * 72)

            if args.target_kind == "sentence":
                Y_train, _, _ = build_text_targets(
                    train_stories, wordseqs, train_resp_lengths, encoder, emb_dim,
                    window_trs=window_trs, target_lag=target_lag,
                    batch_size=args.embed_batch_size,
                )
                Y_test, test_texts, y_groups_test = build_text_targets(
                    test_stories, wordseqs, test_resp_lengths, encoder, emb_dim,
                    window_trs=window_trs, target_lag=target_lag,
                    batch_size=args.embed_batch_size,
                )
            else:
                Y_train, _, _ = build_word2vec_mean_targets(
                    train_stories, wordseqs, train_resp_lengths, word_vectors, emb_dim,
                    window_trs=window_trs, target_lag=target_lag,
                )
                Y_test, test_texts, y_groups_test = build_word2vec_mean_targets(
                    test_stories, wordseqs, test_resp_lengths, word_vectors, emb_dim,
                    window_trs=window_trs, target_lag=target_lag,
                )
            if not np.array_equal(y_groups_test, g_test):
                raise ValueError("Target and brain group labels are misaligned.")

            Y_train_z, Y_test_z, y_mu, y_sd = zscore_train_test(Y_train, Y_test)
            Y_train_model, Y_test_model, target_pca = maybe_pca(
                Y_train_z, Y_test_z, args.target_pca, "target",
            )

            def invert_target(pred_model):
                pred_z = pred_model
                if target_pca is not None:
                    pred_z = target_pca.inverse_transform(pred_model).astype(np.float32)
                return (pred_z * y_sd + y_mu).astype(np.float32)

            def add_result(model_name, params, pred_model):
                pred = invert_target(pred_model)
                row = {
                    "model": model_name,
                    "window_trs": int(window_trs),
                    "target_lag": int(target_lag),
                    "target_kind": args.target_kind,
                    "target_label": target_label,
                    "roi": roi_name,
                    "n_voxels": int(len(vox)),
                    "brain_pca": int(args.brain_pca),
                    "target_pca": int(args.target_pca),
                    **params,
                    **evaluate_predictions(
                        Y_test, pred, g_test,
                        paragraph_window=args.paragraph_window_trs,
                        paragraph_stride=args.paragraph_stride_trs,
                    ),
                }
                add_null_metrics(
                    row, Y_test, pred, g_test, args,
                    window_trs=window_trs,
                    target_lag=target_lag,
                    model_name=model_name,
                )
                results.append(row)
                null_txt = ""
                if args.null_iters > 0:
                    null_txt = (
                        f" dim_null_mean={row['dim_r_null_mean']:.4f}"
                        f" p={row['dim_r_null_p']:.3f}"
                    )
                log.info(
                    "  %-12s dim_r=%.4f TR_top1=%.3f story_top1=%.3f "
                    "para_top1=%.3f para_story=%.3f%s params=%s",
                    model_name, row["dim_r_test"], row["tr_top1"], row["story_top1"],
                    row["paragraph_top1"], row["paragraph_story_top1"], null_txt, params,
                )

            if not args.skip_sklearn:
                if "ridge" in args.sklearn_models or "ridge_rankk" in args.sklearn_models:
                    fitted_ridges = {}
                    for alpha in args.ridge_alphas:
                        log.info("Fitting ridge alpha=%g", alpha)
                        ridge_model, pred_model = ridge_fit_predict(
                            X_train_model, Y_train_model, X_test_model, alpha,
                        )
                        fitted_ridges[float(alpha)] = ridge_model
                        if "ridge" in args.sklearn_models:
                            add_result("ridge", {"alpha": float(alpha)}, pred_model)
                        if "ridge_rankk" in args.sklearn_models:
                            for rank in args.ridge_ranks:
                                if rank > min(X_train_model.shape[1], Y_train_model.shape[1]):
                                    continue
                                log.info("  Applying ridge rank-k alpha=%g rank=%d", alpha, rank)
                                pred_rank = ridge_rankk_predict_from_model(ridge_model, X_test_model, rank)
                                add_result(
                                    "ridge_rankk",
                                    {"alpha": float(alpha), "rank": int(rank)},
                                    pred_rank,
                                )
                if "pls" in args.sklearn_models:
                    for n_comp in args.pls_components:
                        if n_comp > min(X_train_model.shape[0] - 1, X_train_model.shape[1], Y_train_model.shape[1]):
                            continue
                        log.info("Fitting PLS n_components=%d", n_comp)
                        _, pred_model = pls_fit_predict(
                            X_train_model, Y_train_model, X_test_model, n_comp,
                        )
                        add_result("pls", {"n_components": int(n_comp)}, pred_model)

            if not args.skip_torch:
                torch_configs = [
                    ("linear", {"weight_decay": 1e-3}),
                    ("linear", {"weight_decay": 1e-2}),
                    ("lowrank", {"rank": 16, "weight_decay": 1e-3}),
                    ("lowrank", {"rank": 32, "weight_decay": 1e-3}),
                    ("mlp", {"hidden": 64, "dropout": 0.3, "weight_decay": 1e-3}),
                    ("mlp", {"hidden": 128, "dropout": 0.3, "weight_decay": 1e-3}),
                ]
                for arch, cfg in torch_configs:
                    log.info("Fitting torch_%s params=%s", arch, cfg)
                    _, pred_model = torch_fit_predict(
                        X_train_model, Y_train_model, X_test_model,
                        arch=arch,
                        device=device,
                        groups=g_train,
                        loss=args.loss,
                        temperature=args.temperature,
                        lr=1e-3,
                        max_epochs=1000,
                        patience=50,
                        batch_size=256,
                        seed=args.seed,
                        **cfg,
                    )
                    add_result(f"torch_{arch}", cfg, pred_model)

            # Keep memory bounded across sweep cells.
            del Y_train, Y_test, Y_train_z, Y_test_z, Y_train_model, Y_test_model, test_texts

    results_sorted = sorted(
        results,
        key=lambda r: (
            np.nan_to_num(r.get("paragraph_story_top1", -1.0), nan=-1.0),
            np.nan_to_num(r.get("dim_r_test", -1.0), nan=-1.0),
        ),
        reverse=True,
    )

    out_dir = Path(args.output_dir) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"text_window__{roi_name}__{args.target_kind}__{target_label}"
        f"__brainpca-{args.brain_pca}__targetpca-{args.target_pca}"
        f"__loss-{args.loss}"
    )
    if args.focused:
        tag += "__focused"
    if args.null_iters > 0:
        tag += f"__null-{args.null_iters}-{args.null_eval}"
    out_csv = out_dir / f"{tag}.csv"
    all_keys = []
    seen = set()
    for r in results_sorted:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results_sorted)
    log.info("Saved results to %s", out_csv)
    log.info("Top models (paragraph/story metrics are the key readout):")
    for row in results_sorted[:10]:
        null_txt = ""
        if "dim_r_null_mean" in row:
            null_txt = f" dim_null={row['dim_r_null_mean']:.4f} p={row['dim_r_null_p']:.3f}"
        log.info(
            "  %-12s win=%d lag=%d dim_r=%.4f TR_top1=%.3f "
            "story=%.3f para_top1=%.3f para_story=%.3f%s params=%s",
            row["model"], row["window_trs"], row["target_lag"], row["dim_r_test"],
            row["tr_top1"], row["story_top1"], row["paragraph_top1"],
            row["paragraph_story_top1"], null_txt,
            {k: row[k] for k in ("alpha", "rank", "hidden", "dropout", "weight_decay") if k in row},
        )


if __name__ == "__main__":
    main()
