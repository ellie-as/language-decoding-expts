#!/usr/bin/env python3
"""Preferred-lag RSA for local transcript representations.

For each ROI and candidate lag, this script compares the representational
geometry of held-out fMRI patterns with the representational geometry of recent
transcript-window embeddings.

Default target representation is `word2vec_mean`, matching the lightweight lag
analysis. RSA is computed within held-out stories by default, then averaged
across stories so story identity does not dominate the result.
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

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_ridge.util import make_delayed  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM
DEFAULT_ROIS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA", "BA_full_frontal"]
ROI_COMPONENTS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]
WORD_RE = re.compile(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
    "she", "so", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preferred_lag_rsa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument(
        "--sessions", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    parser.add_argument("--holdout-stories", nargs="+", default=None)
    parser.add_argument("--holdout-count", type=int, default=rse.DEFAULT_HOLDOUT_COUNT)
    parser.add_argument("--no-story-holdout", action="store_true")
    parser.add_argument("--local-compute-mode", action="store_true")
    parser.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    parser.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    parser.add_argument("--output-dir", default="preferred_lag_rsa_results")

    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    parser.add_argument("--local-window-trs", type=int, default=5)
    parser.add_argument(
        "--target-representation",
        default="word2vec_mean",
        choices=["sentence_transformer", "word2vec_mean", "content_word2vec_mean", "idf_word2vec_mean"],
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--word-vector-path", default=None)
    parser.add_argument("--word-vector-model", default="glove-wiki-gigaword-300")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--lags", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--tr-sec", type=float, default=2.0)
    parser.add_argument("--metric", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--max-trs-per-story", type=int, default=350)
    parser.add_argument("--null-iters", type=int, default=200)
    parser.add_argument("--min-shift-trs", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    mounted_root = None
    if args.local_compute_mode:
        mounted_root = rse.configure_local_compute_mode(args)
    return rse.resolve_output_dir(args, mounted_root)


def subject_to_uts(subject: str) -> str:
    if subject in rse.SUBJECT_TO_UTS:
        return rse.SUBJECT_TO_UTS[subject]
    if subject.startswith("UTS"):
        return subject
    raise ValueError(f"Cannot map subject {subject!r} to UTS ID")


def load_roi_indices(subject: str, roi: str, ba_dir: str | Path, total_voxels: int) -> np.ndarray:
    roi = "BA_full_frontal" if roi == "full_frontal" else roi
    subj_dir = Path(ba_dir).expanduser().resolve() / subject_to_uts(subject)
    roi_path = subj_dir / f"{roi}.json"
    if roi_path.exists():
        with open(roi_path, encoding="utf-8") as f:
            idx = np.asarray(next(iter(json.load(f).values())), dtype=np.int64)
    elif roi == "BA_full_frontal":
        full: set[int] = set()
        for component in ROI_COMPONENTS:
            with open(subj_dir / f"{component}.json", encoding="utf-8") as f:
                full.update(int(v) for v in next(iter(json.load(f).values())))
        idx = np.asarray(sorted(full), dtype=np.int64)
    else:
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    idx = np.sort(idx[idx < total_voxels])
    if idx.size == 0:
        raise ValueError(f"ROI {roi} has no voxels under total voxel count {total_voxels}")
    return idx


def normalize_word(word: str) -> str:
    return WORD_RE.sub("", str(word).lower()).strip()


def is_content_word(word: str) -> bool:
    word = normalize_word(word)
    return bool(word) and any(ch.isalpha() for ch in word) and word not in STOPWORDS


def collect_story_vocab(wordseqs: dict, stories: list[str]) -> set[str]:
    vocab = set()
    for story in stories:
        for word in wordseqs[story].data:
            normalized = normalize_word(word)
            if normalized:
                vocab.add(normalized)
    return vocab


def load_word_vectors_for_vocab(path: str | Path, needed_vocab: set[str]) -> tuple[dict[str, np.ndarray], int]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Word-vector file not found: {path}")
    vectors = {}
    dim = None
    log.info("Loading word vectors from %s for %d story words", path, len(needed_vocab))
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
    if not vectors or dim is None:
        raise ValueError(f"No matching word vectors found in {path}")
    log.info("Loaded %d/%d word vectors (dim=%d)", len(vectors), len(needed_vocab), dim)
    return vectors, dim


def load_gensim_vectors_for_vocab(model_name: str, needed_vocab: set[str]) -> tuple[dict[str, np.ndarray], int]:
    try:
        import gensim.downloader as api
    except Exception as err:
        raise RuntimeError("Install gensim or pass --word-vector-path to a local GloVe/word2vec file.") from err
    log.info("Loading gensim word-vector model: %s", model_name)
    kv = api.load(model_name)
    vectors = {word: np.asarray(kv[word], dtype=np.float32) for word in needed_vocab if word in kv}
    if not vectors:
        raise ValueError(f"No story words found in gensim model {model_name}")
    dim = int(next(iter(vectors.values())).shape[0])
    log.info("Loaded %d/%d word vectors (dim=%d)", len(vectors), len(needed_vocab), dim)
    return vectors, dim


def compute_story_idf(wordseqs: dict, train_stories: list[str]) -> dict[str, float]:
    df = {}
    for story in train_stories:
        words = {normalize_word(w) for w in wordseqs[story].data if normalize_word(w)}
        for word in words:
            df[word] = df.get(word, 0) + 1
    n_docs = max(1, len(train_stories))
    return {word: float(np.log((1 + n_docs) / (1 + count)) + 1.0) for word, count in df.items()}


def embed_texts(model, texts: list[str], dim: int, batch_size: int) -> np.ndarray:
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    idx = [i for i, text in enumerate(texts) if text.strip()]
    if not idx:
        return vecs
    enc = model.encode(
        [texts[i] for i in idx],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    vecs[idx] = enc
    return vecs


def build_text_windows(wordseq, response_len: int, window_trs: int) -> list[str]:
    words = np.asarray(wordseq.data)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
    tr = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 2.0
    half_tr = tr / 2.0
    texts = []
    for i in range(response_len):
        stim_idx = TRIM_START + i
        end_t = tr_times[stim_idx] + half_tr
        start_t = end_t - float(window_trs) * tr
        mask = (word_times >= start_t) & (word_times < end_t)
        texts.append(" ".join(str(w).strip() for w in words[mask] if str(w).strip()))
    return texts


def build_word2vec_features_for_story(
    wordseq,
    response_len: int,
    window_trs: int,
    word_vectors: dict[str, np.ndarray],
    emb_dim: int,
    target_representation: str,
    idf_weights: dict[str, float] | None = None,
) -> np.ndarray:
    words = np.asarray(wordseq.data)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
    tr = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 2.0
    half_tr = tr / 2.0
    per_word = []
    per_weight = []
    for word in words:
        normalized = normalize_word(word)
        vec = word_vectors.get(normalized)
        keep = True
        weight = 1.0
        if target_representation == "content_word2vec_mean":
            keep = is_content_word(normalized)
        elif target_representation == "idf_word2vec_mean":
            weight = float((idf_weights or {}).get(normalized, 1.0))
        if not keep:
            vec = None
        per_word.append(vec)
        per_weight.append(weight)

    targets = np.zeros((response_len, emb_dim), dtype=np.float32)
    for i in range(response_len):
        stim_idx = TRIM_START + i
        end_t = tr_times[stim_idx] + half_tr
        start_t = end_t - float(window_trs) * tr
        idx = np.nonzero((word_times >= start_t) & (word_times < end_t))[0]
        vecs = [per_word[j] for j in idx if per_word[j] is not None]
        if not vecs:
            continue
        weights = np.asarray([per_weight[j] for j in idx if per_word[j] is not None], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-12)
        targets[i] = np.sum(np.vstack(vecs) * weights[:, None], axis=0).astype(np.float32)
    return targets


def build_base_features(
    args: argparse.Namespace,
    stories: list[str],
    train_stories: list[str],
    resp_lengths: dict[str, int],
) -> dict[str, np.ndarray]:
    wordseqs = get_story_wordseqs(stories)
    features = {}
    if args.target_representation == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s", args.embedding_model)
        encoder = SentenceTransformer(args.embedding_model, device=args.device)
        dim = int(encoder.get_sentence_embedding_dimension())
        for story in stories:
            texts = build_text_windows(wordseqs[story], resp_lengths[story], args.local_window_trs)
            features[story] = embed_texts(encoder, texts, dim, args.embedding_batch_size)
        return features

    needed = collect_story_vocab(wordseqs, stories)
    if args.word_vector_path:
        word_vectors, dim = load_word_vectors_for_vocab(args.word_vector_path, needed)
    else:
        word_vectors, dim = load_gensim_vectors_for_vocab(args.word_vector_model, needed)
    idf_weights = None
    if args.target_representation == "idf_word2vec_mean":
        idf_weights = compute_story_idf(wordseqs, train_stories)
    for story in stories:
        features[story] = build_word2vec_features_for_story(
            wordseqs[story],
            response_len=resp_lengths[story],
            window_trs=args.local_window_trs,
            word_vectors=word_vectors,
            emb_dim=dim,
            target_representation=args.target_representation,
            idf_weights=idf_weights,
        )
    return features


def zscore_train_apply(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return np.nan_to_num((train - mean) / std).astype(np.float32), np.nan_to_num((test - mean) / std).astype(np.float32)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    x = x / denom
    return x @ x.T


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def vector_corr(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "spearman":
        a = rankdata_average(a)
        b = rankdata_average(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def upper_triangle_values(sim: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(sim.shape[0], k=1)
    return sim[idx].astype(np.float32)


def rsa_for_story(brain: np.ndarray, text: np.ndarray, metric: str) -> tuple[float, int]:
    if brain.shape[0] < 3:
        return float("nan"), 0
    brain_vec = upper_triangle_values(cosine_similarity_matrix(brain))
    text_vec = upper_triangle_values(cosine_similarity_matrix(text))
    return vector_corr(brain_vec, text_vec, metric), len(brain_vec)


def subsample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def make_lagged_features(base_by_story: dict[str, np.ndarray], story: str, lag: int) -> np.ndarray:
    return make_delayed(base_by_story[story], [lag]).astype(np.float32)


def circular_shift_text(text: np.ndarray, rng: np.random.Generator, min_shift: int) -> np.ndarray:
    n = text.shape[0]
    if n <= 2 * min_shift:
        shift = int(rng.integers(1, max(2, n)))
    else:
        shift = int(rng.choice(np.arange(min_shift, n - min_shift + 1)))
    return np.roll(text, shift=shift, axis=0)


def roi_rsa_by_lag(
    brain_by_story: dict[str, np.ndarray],
    text_by_story: dict[str, np.ndarray],
    test_stories: list[str],
    lags: list[int],
    metric: str,
    max_trs_per_story: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    scores = []
    details = {}
    story_indices = {
        story: subsample_indices(brain_by_story[story].shape[0], max_trs_per_story, rng)
        for story in test_stories
    }
    for lag in lags:
        story_scores = []
        story_weights = []
        for story in test_stories:
            idx = story_indices[story]
            brain = brain_by_story[story][idx]
            text = make_lagged_features(text_by_story, story, lag)[idx]
            score, n_pairs = rsa_for_story(brain, text, metric)
            if np.isfinite(score) and n_pairs > 0:
                story_scores.append(score)
                story_weights.append(n_pairs)
        if story_weights:
            score = float(np.average(story_scores, weights=story_weights))
        else:
            score = float("nan")
        scores.append(score)
        details[lag] = {
            "mean_story_rsa": float(np.nanmean(story_scores)) if story_scores else float("nan"),
            "weighted_rsa": score,
            "n_pairs": int(np.sum(story_weights)) if story_weights else 0,
        }
    return np.asarray(scores, dtype=np.float32), details


def null_rsa_by_lag(
    brain_by_story: dict[str, np.ndarray],
    text_by_story: dict[str, np.ndarray],
    test_stories: list[str],
    lags: list[int],
    metric: str,
    max_trs_per_story: int,
    min_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scores = []
    for lag in lags:
        story_scores = []
        story_weights = []
        for story in test_stories:
            idx = subsample_indices(brain_by_story[story].shape[0], max_trs_per_story, rng)
            brain = brain_by_story[story][idx]
            text = make_lagged_features(text_by_story, story, lag)
            text = circular_shift_text(text, rng, min_shift)[idx]
            score, n_pairs = rsa_for_story(brain, text, metric)
            if np.isfinite(score) and n_pairs > 0:
                story_scores.append(score)
                story_weights.append(n_pairs)
        scores.append(float(np.average(story_scores, weights=story_weights)) if story_weights else float("nan"))
    return np.asarray(scores, dtype=np.float32)


def lag_stats(scores: np.ndarray, lags: list[int]) -> dict[str, float | int]:
    order = np.argsort(scores)
    best_idx = int(order[-1])
    second_idx = int(order[-2]) if len(order) > 1 else best_idx
    return {
        "preferred_lag_tr": int(lags[best_idx]),
        "best_rsa": float(scores[best_idx]),
        "second_best_rsa": float(scores[second_idx]),
        "lag_selectivity": float(scores[best_idx] - scores[second_idx]),
    }


def empirical_p(observed: float, null_values: np.ndarray) -> float:
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0:
        return float("nan")
    return float(((null_values >= observed).sum() + 1.0) / (null_values.size + 1.0))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    stories = rse.load_story_list(args)
    train_stories, test_stories = rse.split_story_list(stories, args)
    if not test_stories:
        raise ValueError("Preferred-lag RSA requires held-out test stories.")
    resp_lengths, total_voxels = rse.load_resp_info(args.subject, stories)
    text_by_story = build_base_features(args, stories, train_stories, resp_lengths)
    train_text = np.vstack([text_by_story[story] for story in train_stories])
    for story in stories:
        _, text_by_story[story] = zscore_train_apply(train_text, text_by_story[story])

    rows = []
    for roi in args.rois:
        vox = load_roi_indices(args.subject, roi, args.ba_dir, total_voxels)
        log.info("Loading %s (%d voxels)", roi, len(vox))
        resp_by_story = get_resp(args.subject, stories, stack=False, vox=vox)
        train_resp = np.vstack([resp_by_story[story] for story in train_stories])
        for story in stories:
            _, resp_by_story[story] = zscore_train_apply(train_resp, resp_by_story[story])

        scores, details = roi_rsa_by_lag(
            resp_by_story,
            text_by_story,
            test_stories,
            args.lags,
            args.metric,
            args.max_trs_per_story,
            rng,
        )
        observed = lag_stats(scores, args.lags)
        null_best = np.zeros(args.null_iters, dtype=np.float32)
        null_selectivity = np.zeros(args.null_iters, dtype=np.float32)
        for i in range(args.null_iters):
            null_scores = null_rsa_by_lag(
                resp_by_story,
                text_by_story,
                test_stories,
                args.lags,
                args.metric,
                args.max_trs_per_story,
                args.min_shift_trs,
                rng,
            )
            null_observed = lag_stats(null_scores, args.lags)
            null_best[i] = float(null_observed["best_rsa"])
            null_selectivity[i] = float(null_observed["lag_selectivity"])
            if (i + 1) % 25 == 0:
                log.info("%s null %d/%d", roi, i + 1, args.null_iters)

        row = {
            "roi": roi,
            "n_voxels": int(len(vox)),
            "target_representation": args.target_representation,
            "local_window_trs": args.local_window_trs,
            "metric": args.metric,
            "max_trs_per_story": args.max_trs_per_story,
            **observed,
            "p_best_rsa": empirical_p(float(observed["best_rsa"]), null_best),
            "p_lag_selectivity": empirical_p(float(observed["lag_selectivity"]), null_selectivity),
        }
        row["preferred_lag_sec"] = float(row["preferred_lag_tr"] * args.tr_sec)
        for lag_idx, lag in enumerate(args.lags):
            row[f"rsa_lag_{lag}tr"] = float(scores[lag_idx])
            row[f"n_pairs_lag_{lag}tr"] = int(details[lag]["n_pairs"])
        rows.append(row)
        log.info(
            "%s preferred lag=%s TR best_rsa=%.4f selectivity=%.4f",
            roi, row["preferred_lag_tr"], row["best_rsa"], row["lag_selectivity"],
        )

    results_path = out_dir / "preferred_lag_rsa_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "subject": args.subject,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "lags_tr": args.lags,
        "lags_sec": [lag * args.tr_sec for lag in args.lags],
        "target_representation": args.target_representation,
        "embedding_model": args.embedding_model if args.target_representation == "sentence_transformer" else None,
        "word_vector_path": args.word_vector_path,
        "word_vector_model": args.word_vector_model if args.word_vector_path is None else None,
    }
    with open(out_dir / "preferred_lag_rsa_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Wrote %s", results_path)


if __name__ == "__main__":
    main()
