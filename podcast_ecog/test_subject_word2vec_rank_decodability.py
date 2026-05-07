#!/usr/bin/env python3
"""Decode static word embeddings at different word ranks from each subject.

This is a subject-level alternative to the contextual GPT rank analyses. For
each subject, the neural feature is the vector of all selected ECoG channels in
a peri-word window around the current word. Targets are static word embeddings
for word ranks t, t-1, t-2, ... reduced with PCA.

The main intended source is a local word2vec model loaded with gensim. The
script also supports the dataset's bundled static ``en_core_web_lg`` vectors as
a smoke-test/static-embedding fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_mfg_rank_subspace import fif_path, ridge_fit_predict, segment_means, target_corr  # noqa: E402


DEFAULT_BIDS_ROOT = Path("/Volumes/ellie/language-decoding-expts/podcast_ecog/data/ds005574")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "subject_word2vec_rank_decodability")
    parser.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)])
    parser.add_argument("--task", default="podcast")
    parser.add_argument("--embedding-source", choices=["word2vec", "dataset_static"], default="word2vec")
    parser.add_argument("--word2vec-path", type=Path, default=None, help="Path to word2vec-format file or gensim .kv model.")
    parser.add_argument("--word2vec-binary", action="store_true", help="Use binary word2vec format. Auto-enabled for .bin/.bin.gz if omitted.")
    parser.add_argument("--word2vec-limit", type=int, default=None, help="Optional vocab limit for gensim load_word2vec_format.")
    parser.add_argument("--dataset-feature-space", default="en_core_web_lg")
    parser.add_argument("--transcript-feature-space", default="gpt2-xl", help="Transcript to use for word timings in word2vec mode.")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--target-modes", nargs="+", choices=["raw_pca", "residual_pca"], default=["raw_pca", "residual_pca"])
    parser.add_argument("--window-start", type=float, default=-0.5)
    parser.add_argument("--window-end", type=float, default=0.5)
    parser.add_argument("--picks-regex", default=None, help="Optional channel regex. Default uses all high-gamma channels.")
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--min-shift", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    return parser.parse_args()


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    pv = p[finite]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[finite] = restored
    return out


def transcript_words_from_token_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    tokens = pd.read_csv(path, sep="\t", index_col=0)
    rows = []
    for word_idx, group in tokens.groupby("word_idx", sort=True):
        rows.append(
            {
                "word_idx": int(word_idx),
                "word": str(group["word"].iloc[0]),
                "start": float(group["start"].iloc[0]),
                "end": float(group["end"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values("word_idx").reset_index(drop=True)


def clean_word_candidates(word: str) -> list[str]:
    word = str(word)
    stripped = word.strip()
    lower = stripped.lower()
    punct = string.punctuation + "“”‘’"
    bare = lower.strip(punct)
    candidates = [stripped, lower, bare]
    if bare.endswith("'s"):
        candidates.append(bare[:-2])
    if bare.endswith("s'"):
        candidates.append(bare[:-2])
    if "-" in bare:
        candidates.append(bare.replace("-", "_"))
        candidates.extend([part for part in bare.split("-") if part])
    out = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def load_word2vec_embeddings(root: Path, transcript_feature_space: str, model_path: Path, binary: bool, limit: int | None) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    try:
        from gensim.models import KeyedVectors
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gensim is required for --embedding-source word2vec") from exc

    if model_path is None:
        raise ValueError("--word2vec-path is required when --embedding-source word2vec")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    transcript = transcript_words_from_token_table(root / "stimuli" / transcript_feature_space / "transcript.tsv")

    suffixes = "".join(model_path.suffixes).lower()
    if suffixes.endswith(".kv"):
        model = KeyedVectors.load(str(model_path), mmap="r")
    else:
        auto_binary = suffixes.endswith(".bin") or suffixes.endswith(".bin.gz")
        model = KeyedVectors.load_word2vec_format(str(model_path), binary=(binary or auto_binary), limit=limit)

    vectors = np.full((len(transcript), model.vector_size), np.nan, dtype=np.float32)
    rows = []
    for i, word in enumerate(transcript["word"].astype(str)):
        matched = None
        for candidate in clean_word_candidates(word):
            if candidate in model:
                matched = candidate
                vectors[i] = model[candidate]
                break
        rows.append({"word_idx": int(transcript.loc[i, "word_idx"]), "word": word, "matched_token": matched, "in_vocab": matched is not None})
    coverage = pd.DataFrame(rows)
    return transcript, vectors, coverage


def load_dataset_static_embeddings(root: Path, feature_space: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    stim_dir = root / "stimuli" / feature_space
    transcript = transcript_words_from_token_table(stim_dir / "transcript.tsv")
    with h5py.File(stim_dir / "features.hdf5", "r") as handle:
        token_vectors = handle["vectors"][...].astype(np.float32)
    token_table = pd.read_csv(stim_dir / "transcript.tsv", sep="\t", index_col=0)
    word_vectors = []
    coverage_rows = []
    for word_idx, group in token_table.groupby("word_idx", sort=True):
        idx = group.index.to_numpy()
        vec = token_vectors[idx]
        finite = np.isfinite(vec).all(axis=1) & (np.linalg.norm(vec, axis=1) > 0)
        if finite.any():
            word_vectors.append(vec[finite].mean(axis=0))
            in_vocab = True
        else:
            word_vectors.append(np.full(token_vectors.shape[1], np.nan, dtype=np.float32))
            in_vocab = False
        coverage_rows.append(
            {
                "word_idx": int(word_idx),
                "word": str(group["word"].iloc[0]),
                "matched_token": str(group.loc[finite, "token"].iloc[0]) if finite.any() else None,
                "in_vocab": bool(in_vocab),
            }
        )
    return transcript, np.vstack(word_vectors).astype(np.float32), pd.DataFrame(coverage_rows)


def make_pca_scores(vectors: np.ndarray, n_components: int) -> tuple[np.ndarray, pd.DataFrame]:
    ok = np.isfinite(vectors).all(axis=1)
    scaler = StandardScaler()
    z = scaler.fit_transform(vectors[ok].astype(np.float64))
    pca = PCA(n_components=n_components, random_state=0)
    scores_ok = pca.fit_transform(z).astype(np.float32)
    scores = np.full((len(vectors), n_components), np.nan, dtype=np.float32)
    scores[ok] = scores_ok
    explained = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return scores, explained


def cache_paths(cache_dir: Path, subject: str) -> tuple[Path, Path]:
    return cache_dir / f"sub-{subject}_central_allch.npz", cache_dir / f"sub-{subject}_central_allch_channels.csv"


def load_subject_neural(
    root: Path,
    subject: str,
    task: str,
    words: pd.DataFrame,
    window_start: float,
    window_end: float,
    picks_regex: str | None,
    cache_dir: Path,
    reuse_cache: bool,
    save_cache: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz_path, channel_path = cache_paths(cache_dir, subject)
    if reuse_cache and npz_path.is_file() and channel_path.is_file():
        npz = np.load(npz_path)
        channels = pd.read_csv(channel_path)["channel"].astype(str).tolist()
        return npz["x"].astype(np.float32), npz["valid"].astype(bool), channels

    raw = mne.io.read_raw_fif(fif_path(root, subject, task), preload=False, verbose=False)
    if picks_regex:
        picks = mne.pick_channels_regexp(raw.ch_names, picks_regex)
        if len(picks) == 0:
            raise ValueError(f"sub-{subject}: no channels match {picks_regex!r}")
        raw.pick(picks)
    print(f"sub-{subject}: reading {len(raw.ch_names)} channels at {raw.info['sfreq']:.1f} Hz", flush=True)
    data = raw.get_data().astype(np.float32)
    centers = raw.time_as_index(words["start"].to_numpy(dtype=float), use_rounding=True)
    x, valid = segment_means(data, centers, float(raw.info["sfreq"]), window_start, window_end)
    channels = list(raw.ch_names)
    if save_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, x=x.astype(np.float32), valid=valid.astype(bool))
        pd.DataFrame({"channel": channels}).to_csv(channel_path, index=False)
    return x, valid, channels


def rank_targets(scores: np.ndarray, word_indices: np.ndarray, ranks: list[int]) -> dict[int, np.ndarray]:
    return {rank: scores[word_indices - int(rank)].astype(np.float32) for rank in ranks}


def residualize_targets(targets: dict[int, np.ndarray], alpha: float = 1.0) -> dict[int, np.ndarray]:
    residuals = {}
    ranks = sorted(targets)
    for rank in ranks:
        y = targets[rank]
        other = np.hstack([targets[other_rank] for other_rank in ranks if other_rank != rank])
        other_z = StandardScaler().fit_transform(other)
        y_scaler = StandardScaler()
        y_z = y_scaler.fit_transform(y)
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(other_z, y_z)
        pred = y_scaler.inverse_transform(model.predict(other_z))
        residuals[rank] = (y - pred).astype(np.float32)
    return residuals


def choose_offsets(n_samples: int, n_permutations: int, min_shift: int, rng: np.random.Generator) -> np.ndarray:
    if n_permutations <= 0:
        return np.asarray([], dtype=int)
    min_shift = min(int(min_shift), max(1, n_samples // 3))
    possible = np.arange(min_shift, n_samples - min_shift)
    if len(possible) == 0:
        possible = np.arange(1, n_samples)
    return rng.choice(possible, size=n_permutations, replace=n_permutations > len(possible)).astype(int)


def decode_cv(x: np.ndarray, y: np.ndarray, alpha: float, outer_splits: int) -> dict[str, float]:
    pred = np.zeros_like(y, dtype=np.float32)
    cv = KFold(n_splits=outer_splits, shuffle=False)
    for train_idx, test_idx in cv.split(x):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x[train_idx]).astype(np.float32)
        x_test = x_scaler.transform(x[test_idx]).astype(np.float32)
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)
        y_pred_z, _coef = ridge_fit_predict(x_train, y_train, x_test, alpha)
        pred[test_idx] = y_scaler.inverse_transform(y_pred_z).astype(np.float32)
    pc_corr = target_corr(y, pred)
    y_centered = y - y.mean(axis=0, keepdims=True)
    ss_total = np.sum(y_centered * y_centered, axis=0)
    weights = ss_total / np.maximum(ss_total.sum(), 1e-12)
    return {
        "mean_pc_r": float(np.nanmean(pc_corr)),
        "median_pc_r": float(np.nanmedian(pc_corr)),
        "pc1_r": float(pc_corr[0]),
        "weighted_pc_r": float(np.nansum(pc_corr * weights)),
    }


def permutation_test(x: np.ndarray, y: np.ndarray, alpha: float, outer_splits: int, offsets: np.ndarray) -> dict[str, float]:
    observed = decode_cv(x, y, alpha, outer_splits)
    null_scores = []
    for offset in offsets:
        shifted = np.roll(y, int(offset), axis=0)
        null_scores.append(decode_cv(x, shifted, alpha, outer_splits))
    null = pd.DataFrame(null_scores)
    for metric, value in list(observed.items()):
        if len(null):
            observed[f"{metric}_null_mean"] = float(null[metric].mean())
            observed[f"{metric}_null_sd"] = float(null[metric].std(ddof=1))
            observed[f"{metric}_p"] = float((1 + np.sum(null[metric].to_numpy() >= value)) / (len(null) + 1))
        else:
            observed[f"{metric}_null_mean"] = np.nan
            observed[f"{metric}_null_sd"] = np.nan
            observed[f"{metric}_p"] = np.nan
    return observed


def target_similarity(targets: dict[int, np.ndarray], mode: str) -> pd.DataFrame:
    rows = []
    ranks = sorted(targets)
    for a_i, a in enumerate(ranks):
        for b in ranks[a_i + 1 :]:
            ya = StandardScaler().fit_transform(targets[a])
            yb = StandardScaler().fit_transform(targets[b])
            corr = target_corr(ya, yb)
            rows.append(
                {
                    "target_mode": mode,
                    "rank_a": int(a),
                    "rank_b": int(b),
                    "rank_distance": int(abs(a - b)),
                    "mean_pc_corr": float(np.nanmean(corr)),
                    "pc1_corr": float(corr[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_dir: Path, metric: str = "weighted_pc_r") -> None:
    for mode, group in summary.groupby("target_mode", sort=False):
        matrix = group.pivot(index="subject", columns="rank", values=metric)
        q_matrix = group.pivot(index="subject", columns="rank", values=f"{metric}_q")
        fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
        vmax = max(0.01, float(np.nanpercentile(np.abs(matrix.to_numpy()), 98)))
        sns.heatmap(matrix, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, cbar_kws={"label": metric}, ax=ax)
        for i, subject in enumerate(matrix.index):
            for j, rank in enumerate(matrix.columns):
                val = matrix.loc[subject, rank]
                q = q_matrix.loc[subject, rank]
                if pd.isna(val):
                    continue
                text = f"{val:.3f}"
                if pd.notna(q) and q < 0.05:
                    text += "*"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=8, color="black")
        ax.set_title(f"{mode}: static embedding rank decoding, * q<0.05")
        ax.set_xlabel("word rank back")
        ax.set_ylabel("subject")
        fig.savefig(output_dir / f"{mode}_{metric}_heatmap.png", dpi=220)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ranks = sorted(set(int(rank) for rank in args.ranks))
    cache_dir = args.cache_dir or args.output_dir / "neural_cache"
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    if args.embedding_source == "word2vec":
        words, vectors, coverage = load_word2vec_embeddings(
            args.bids_root,
            args.transcript_feature_space,
            args.word2vec_path,
            args.word2vec_binary,
            args.word2vec_limit,
        )
    else:
        words, vectors, coverage = load_dataset_static_embeddings(args.bids_root, args.dataset_feature_space)
    coverage.to_csv(args.output_dir / "embedding_coverage.csv", index=False)
    print(
        f"Embedding coverage: {int(coverage['in_vocab'].sum())}/{len(coverage)} "
        f"({coverage['in_vocab'].mean() * 100:.1f}%)",
        flush=True,
    )
    scores, explained = make_pca_scores(vectors, args.n_components)
    explained.to_csv(args.output_dir / "embedding_pca_explained_variance.csv", index=False)

    rng = np.random.default_rng(args.random_seed)
    summary_rows = []
    similarity_rows = []
    max_rank = max(args.ranks)
    for subject in args.subjects:
        x_all, neural_valid, channels = load_subject_neural(
            args.bids_root,
            subject,
            args.task,
            words,
            args.window_start,
            args.window_end,
            args.picks_regex,
            cache_dir,
            args.reuse_cache,
            args.save_cache,
        )
        common = neural_valid.copy()
        common[:max_rank] = False
        for rank in args.ranks:
            target_idx = np.arange(len(words)) - int(rank)
            valid_target = (target_idx >= 0) & np.isfinite(scores[np.maximum(target_idx, 0)]).all(axis=1)
            common &= valid_target
        word_indices = np.flatnonzero(common)
        x = x_all[word_indices].astype(np.float32)
        print(f"sub-{subject}: X={x.shape}; common words={len(word_indices)}", flush=True)
        raw_targets = rank_targets(scores, word_indices, args.ranks)
        mode_targets = {"raw_pca": raw_targets}
        if "residual_pca" in args.target_modes:
            mode_targets["residual_pca"] = residualize_targets(raw_targets)
        offsets = choose_offsets(len(word_indices), args.n_permutations, args.min_shift, rng)
        for mode in args.target_modes:
            targets = mode_targets[mode]
            sim = target_similarity(targets, mode)
            sim.insert(0, "subject", f"sub-{subject}")
            similarity_rows.append(sim)
            for rank in args.ranks:
                row = {
                    "subject": f"sub-{subject}",
                    "target_mode": mode,
                    "rank": int(rank),
                    "n_samples": int(len(word_indices)),
                    "n_channels": int(x.shape[1]),
                    "n_components": int(args.n_components),
                    "n_permutations": int(len(offsets)),
                }
                row.update(permutation_test(x, targets[rank], args.ridge_alpha, args.outer_splits, offsets))
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    for metric in ["mean_pc_r", "median_pc_r", "pc1_r", "weighted_pc_r"]:
        p_col = f"{metric}_p"
        if p_col in summary:
            summary[f"{metric}_q"] = fdr_bh(summary[p_col].to_numpy())
    summary.to_csv(args.output_dir / "subject_static_rank_decodability_summary.csv", index=False)
    if similarity_rows:
        pd.concat(similarity_rows, ignore_index=True).to_csv(args.output_dir / "subject_static_target_rank_similarity.csv", index=False)
    plot_summary(summary, args.output_dir, "weighted_pc_r")
    plot_summary(summary, args.output_dir, "mean_pc_r")

    print("\nPCA explained variance", flush=True)
    print(explained.round(4).to_string(index=False), flush=True)
    print("\nWeighted PC r", flush=True)
    table = summary.pivot_table(index=["subject", "target_mode"], columns="rank", values="weighted_pc_r")
    print(table.round(4).to_string(), flush=True)
    print("\nWeighted PC r q-values", flush=True)
    qtable = summary.pivot_table(index=["subject", "target_mode"], columns="rank", values="weighted_pc_r_q")
    print(qtable.round(4).to_string(), flush=True)
    print(f"\nSaved subject static embedding rank decoding outputs to: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
