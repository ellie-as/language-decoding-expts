#!/usr/bin/env python3
"""Train standalone MiniLM combo ridge encoders for decoding comparison.

This adds two GPT-1-comparison conditions:

* ``minilm_summary_combo``: [1TR MiniLM, summary h20, summary h50, summary h200]
* ``minilm_window_combo``: [1TR MiniLM, raw-text window w20, w50, w200]

Both are trained as lagged TR-level ridge encoders and exported directly in the
``decode_and_score.py`` schema. The window variant is decoder-native: the same
candidate-window features can be computed during beam search. The summary
variant is useful for encoding comparison, but decoding uses the same candidate
window proxy because oracle summaries of the held-out story are not available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(REPO_DIR / "lag_preference_analysis"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from _shared import TRIM_START, load_or_build_chunk_embeddings  # noqa: E402
from compare_gpt1_encoding import DEFAULT_SESSIONS  # noqa: E402
from train_lag_encoding import (  # noqa: E402
    configure_data_root,
    load_full_frontal_voxels,
    per_voxel_corr,
    split_stories,
    stack_lag,
)
from train_summary_combo_encoding import build_combo_embeddings, load_or_build_summary_embeddings  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


CONDITION_MINILM_SUMMARY_COMBO = "minilm_summary_combo"
CONDITION_MINILM_WINDOW_COMBO = "minilm_window_combo"
CONDITIONS = [CONDITION_MINILM_SUMMARY_COMBO, CONDITION_MINILM_WINDOW_COMBO]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=["S1"])
    p.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_SESSIONS)
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0])
    p.add_argument("--voxel-chunk-size", type=int, default=5_000)
    p.add_argument("--voxel-count", type=int, default=config.VOXELS)
    p.add_argument("--voxel-set", choices=["full_frontal", "all"], default="full_frontal")
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--summary-horizons", nargs="+", type=int, default=[20, 50, 200])
    p.add_argument("--window-sizes", nargs="+", type=int, default=[20, 50, 200])
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--embedding-cache-dir", default=str(THIS_DIR / "cache"))
    p.add_argument("--one-tr-cache-dir", default=str(REPO_DIR / "27-04-expts" / "cache"))

    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--output-dir", default=str(THIS_DIR / "outputs"))
    return p.parse_args()


def load_story_list(args: argparse.Namespace) -> List[str]:
    if args.stories:
        return list(args.stories)
    with open(Path(config.DATA_TRAIN_DIR) / "sess_to_story.json", encoding="utf-8") as f:
        sess_to_story = json.load(f)
    stories: List[str] = []
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def maybe_stage_response_root(args: argparse.Namespace, subject: str, stories: Sequence[str], mounted_root: Path | None) -> str:
    if args.local_compute_mode and mounted_root is not None:
        return str(
            rse.stage_local_response_cache(
                subject,
                stories,
                Path(config.DATA_TRAIN_DIR),
                Path(args.local_cache_root).expanduser().resolve(),
            )
        )
    return config.DATA_TRAIN_DIR


def window_cache_key(args: argparse.Namespace, subject: str, stories: Sequence[str], resp_lengths: Dict[str, int]) -> str:
    payload = {
        "subject": subject,
        "stories": list(stories),
        "resp_lengths": {s: int(resp_lengths[s]) for s in stories},
        "embedding_model": args.embedding_model,
        "window_sizes": list(map(int, args.window_sizes)),
        "lag": int(args.lag),
        "trim_start": int(TRIM_START),
        "version": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def text_window_for_tr(wordseq, response_start: int, n_words: int) -> str:
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    words = np.asarray(wordseq.data)
    tr = float(np.median(np.diff(tr_times))) if len(tr_times) > 1 else 2.0
    stim_tr = TRIM_START + response_start
    stop_t = tr_times[stim_tr] + tr / 2.0
    upto = np.nonzero(word_times < stop_t)[0]
    if upto.size == 0:
        return ""
    selected = upto[-int(n_words):]
    return " ".join(str(words[i]).strip() for i in selected if str(words[i]).strip())


def load_or_build_window_embeddings(
    args: argparse.Namespace,
    subject: str,
    stories: Sequence[str],
    resp_lengths: Dict[str, int],
) -> Tuple[Dict[int, Dict[str, np.ndarray]], int, str]:
    cache_dir = Path(args.embedding_cache_dir).expanduser().resolve() / subject
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = window_cache_key(args, subject, stories, resp_lengths)
    cache_path = cache_dir / f"minilm_text_windows__w{'-'.join(map(str, args.window_sizes))}__lag{args.lag}__{key}.pkl"
    if cache_path.is_file():
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload["embeddings"], int(payload["embedding_dim"]), str(cache_path)

    from sentence_transformers import SentenceTransformer

    wordseqs = get_story_wordseqs(list(stories))
    model = SentenceTransformer(args.embedding_model, device="cpu")
    emb_dim = int(model.get_sentence_embedding_dimension())
    out: Dict[int, Dict[str, np.ndarray]] = {int(w): {} for w in args.window_sizes}
    try:
        for window in args.window_sizes:
            texts: List[str] = []
            slices: Dict[str, tuple[int, int]] = {}
            cursor = 0
            for story in stories:
                n = int(resp_lengths[story]) - int(args.lag)
                story_texts = [text_window_for_tr(wordseqs[story], i, int(window)) for i in range(n)]
                texts.extend(story_texts)
                slices[story] = (cursor, cursor + n)
                cursor += n

            vecs = np.zeros((len(texts), emb_dim), dtype=np.float32)
            nonempty = [i for i, text in enumerate(texts) if text.strip()]
            if nonempty:
                vecs[nonempty] = model.encode(
                    [texts[i] for i in nonempty],
                    batch_size=int(args.embed_batch_size),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32, copy=False)
            for story, (start, end) in slices.items():
                out[int(window)][story] = vecs[start:end]
    finally:
        del model

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "embeddings": out,
                "embedding_dim": emb_dim,
                "embedding_model": args.embedding_model,
                "stories": list(stories),
                "window_sizes": list(map(int, args.window_sizes)),
            },
            f,
        )
    return out, emb_dim, str(cache_path)


def build_window_combo_embeddings(
    one_tr: Dict[str, np.ndarray],
    window_embs: Dict[int, Dict[str, np.ndarray]],
    stories: Sequence[str],
    windows: Sequence[int],
) -> Dict[str, np.ndarray]:
    combo: Dict[str, np.ndarray] = {}
    for story in stories:
        n = int(one_tr[story].shape[0])
        blocks = [one_tr[story]]
        for window in windows:
            block = window_embs[int(window)][story]
            if block.shape[0] < n:
                raise ValueError(f"{story} w{window}: {block.shape[0]} window rows < {n} 1TR rows")
            blocks.append(block[:n])
        combo[story] = np.concatenate(blocks, axis=1).astype(np.float32)
    return combo


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def normalize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def fit_ridgecv_chunks(
    x_train_z: np.ndarray,
    y_train: np.ndarray,
    x_val_z: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float],
    voxel_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import RidgeCV

    n_voxels = y_train.shape[1]
    corrs = np.zeros(n_voxels, dtype=np.float32)
    best_alphas = np.zeros(n_voxels, dtype=np.float32)
    chunk = max(1, int(voxel_chunk_size))
    n_chunks = (n_voxels + chunk - 1) // chunk
    for ci, start in enumerate(range(0, n_voxels, chunk), start=1):
        end = min(start + chunk, n_voxels)
        t0 = time.time()
        model = RidgeCV(alphas=list(alphas), alpha_per_target=True, fit_intercept=False)
        model.fit(x_train_z, y_train[:, start:end])
        pred = model.predict(x_val_z).astype(np.float32)
        corrs[start:end] = per_voxel_corr(pred, y_val[:, start:end])
        alpha = np.asarray(model.alpha_, dtype=np.float32)
        if alpha.ndim == 0:
            alpha = np.full(end - start, float(alpha), dtype=np.float32)
        best_alphas[start:end] = alpha
        print(f"  alpha/select chunk {ci}/{n_chunks} voxels {start}:{end} in {time.time() - t0:.1f}s")
    return corrs, best_alphas


def fit_selected_weights(x_train_z: np.ndarray, y_train_sel: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import Ridge

    weights = np.zeros((x_train_z.shape[1], y_train_sel.shape[1]), dtype=np.float32)
    for alpha in np.unique(alphas):
        cols = np.nonzero(alphas == alpha)[0]
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(x_train_z, y_train_sel[:, cols])
        coef = np.asarray(model.coef_, dtype=np.float32)
        if coef.ndim == 1:
            coef = coef[None, :]
        weights[:, cols] = coef.T
    return weights


def estimate_noise_model(x_val_z: np.ndarray, y_val_sel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    residual = (y_val_sel - x_val_z @ weights).astype(np.float32)
    sigma = residual.T @ residual
    diag_mean = float(np.diag(sigma).mean())
    if diag_mean > 0:
        sigma = sigma / diag_mean
    return sigma.astype(np.float32, copy=False)


def delayed_weights(combo_weights: np.ndarray, combo_dim: int, lag: int) -> np.ndarray:
    delays = list(config.STIM_DELAYS)
    if int(lag) not in delays:
        raise ValueError(f"--lag {lag} is not in config.STIM_DELAYS={delays}")
    weights = np.zeros((combo_dim * len(delays), combo_weights.shape[1]), dtype=np.float32)
    di = delays.index(int(lag))
    weights[di * combo_dim : (di + 1) * combo_dim] = combo_weights
    return weights


def train_condition(
    subject: str,
    condition: str,
    args: argparse.Namespace,
    stories: Sequence[str],
    train_stories: Sequence[str],
    val_stories: Sequence[str],
    voxels_base: np.ndarray,
    responses_by_story: Dict[str, np.ndarray],
    one_tr: Dict[str, np.ndarray],
    summary_embs: Dict[int, Dict[str, np.ndarray]] | None,
    window_embs: Dict[int, Dict[str, np.ndarray]] | None,
    caches: Dict[str, str],
    out_path: Path,
) -> dict:
    if condition == CONDITION_MINILM_SUMMARY_COMBO:
        if summary_embs is None:
            raise ValueError("summary embeddings are required for minilm_summary_combo")
        features_by_story = build_combo_embeddings(one_tr, summary_embs, stories, args.summary_horizons)
        feature_blocks = ["1TR"] + [f"h{h}" for h in args.summary_horizons]
    elif condition == CONDITION_MINILM_WINDOW_COMBO:
        if window_embs is None:
            raise ValueError("window embeddings are required for minilm_window_combo")
        features_by_story = build_window_combo_embeddings(one_tr, window_embs, stories, args.window_sizes)
        feature_blocks = ["1TR"] + [f"w{w}" for w in args.window_sizes]
    else:
        raise ValueError(condition)

    x_train, y_train = stack_lag(features_by_story, responses_by_story, train_stories, args.lag)
    x_val, y_val = stack_lag(features_by_story, responses_by_story, val_stories, args.lag)
    x_mean, x_std = normalize_fit(x_train)
    x_train_z = normalize_apply(x_train, x_mean, x_std)
    x_val_z = normalize_apply(x_val, x_mean, x_std)

    print(f"[{subject} / {condition}] ridge CV X_train={x_train_z.shape} Y_train={y_train.shape}")
    corrs, best_alphas = fit_ridgecv_chunks(
        x_train_z,
        y_train,
        x_val_z,
        y_val,
        args.ridge_alphas,
        args.voxel_chunk_size,
    )
    voxel_count = min(int(args.voxel_count), corrs.size)
    local_selected = np.sort(np.argsort(corrs)[-voxel_count:])
    selected_voxels = voxels_base[local_selected]
    selected_alphas = best_alphas[local_selected]

    print(f"[{subject} / {condition}] fitting selected weights for {voxel_count} voxels")
    combo_weights = fit_selected_weights(x_train_z, y_train[:, local_selected], selected_alphas)
    noise_model = estimate_noise_model(x_val_z, y_val[:, local_selected], combo_weights)
    weights = delayed_weights(combo_weights, combo_weights.shape[0], args.lag)

    metadata = {
        "condition": condition,
        "subject": subject,
        "feature_model": "minilm_combo",
        "feature_blocks": feature_blocks,
        "embedding_model": args.embedding_model,
        "lag": int(args.lag),
        "voxel_set": args.voxel_set,
        "voxel_count": int(voxel_count),
        "train_stories": list(train_stories),
        "val_stories": list(val_stories),
        "ridge_alphas": list(map(float, args.ridge_alphas)),
        "candidate_decode_features": "window_proxy",
        "caches": caches,
    }
    np.savez(
        out_path,
        weights=weights,
        weights_are_selected=True,
        noise_model=noise_model,
        alphas=selected_alphas.astype(np.float32),
        voxels=selected_voxels.astype(np.int64),
        stories=np.array(stories),
        train_stories=np.array(train_stories),
        val_stories=np.array(val_stories),
        tr_stats=np.array([x_mean, x_std], dtype=object),
        word_stats=np.array([x_mean], dtype=object),
        bootstrap_corrs=corrs.astype(np.float32),
        selected_corrs=corrs[local_selected].astype(np.float32),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "condition": condition,
        "path": str(out_path),
        "mean_encoding_r": float(corrs.mean()),
        "median_encoding_r": float(np.median(corrs)),
        "max_encoding_r": float(corrs.max()),
        "n_r_gt_0_05": int((corrs > 0.05).sum()),
        "n_r_gt_0_1": int((corrs > 0.10).sum()),
        "n_r_gt_0_2": int((corrs > 0.20).sum()),
        "selected_mean_r": float(corrs[local_selected].mean()),
        "selected_median_r": float(np.median(corrs[local_selected])),
    }


def load_existing_summary(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"].reshape(-1)[0]))
    corrs = np.asarray(data["bootstrap_corrs"], dtype=np.float32)
    selected = np.asarray(data["selected_corrs"], dtype=np.float32)
    return {
        "condition": metadata["condition"],
        "path": str(path),
        "mean_encoding_r": float(corrs.mean()),
        "median_encoding_r": float(np.median(corrs)),
        "max_encoding_r": float(corrs.max()),
        "n_r_gt_0_05": int((corrs > 0.05).sum()),
        "n_r_gt_0_1": int((corrs > 0.10).sum()),
        "n_r_gt_0_2": int((corrs > 0.20).sum()),
        "selected_mean_r": float(selected.mean()),
        "selected_median_r": float(np.median(selected)),
    }


def write_encoding_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.summary_horizons = sorted({int(h) for h in args.summary_horizons})
    args.window_sizes = sorted({int(w) for w in args.window_sizes})
    args.lags = [int(args.lag)]
    args.chunk_trs = 1
    args.feature_model = "embedding"

    mounted_root = configure_data_root(args)
    stories = load_story_list(args)
    train_stories, val_stories = split_stories(stories, args)
    print(f"Stories: {len(stories)} total | {len(train_stories)} train | {len(val_stories)} val")
    print(f"Validation stories: {', '.join(val_stories)}")

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        args.subject = subject
        response_root = maybe_stage_response_root(args, subject, stories, mounted_root)
        sample = get_resp(subject, [stories[0]], stack=True, vox=None, response_root=response_root)
        total_voxels = int(sample.shape[1])
        if args.voxel_set == "full_frontal":
            voxels_base = load_full_frontal_voxels(subject, total_voxels, args.ba_dir)
        else:
            voxels_base = np.arange(total_voxels, dtype=np.int64)
        print(f"\n=== {subject}: voxel_set={args.voxel_set} n={len(voxels_base)} / {total_voxels} ===")

        responses = get_resp(subject, stories, stack=False, vox=voxels_base, response_root=response_root)
        responses = {story: arr.astype(np.float32) for story, arr in responses.items()}
        resp_lengths = {story: int(arr.shape[0]) for story, arr in responses.items()}

        one_args = argparse.Namespace(
            subject=subject,
            embedding_cache_dir=args.one_tr_cache_dir,
            feature_model="embedding",
            chunk_trs=1,
            lag_trs=int(args.lag),
            embed_batch_size=int(args.embed_batch_size),
        )
        one_tr, one_dim, one_cache = load_or_build_chunk_embeddings(
            one_args,
            stories,
            resp_lengths,
            response_root=config.DATA_TRAIN_DIR,
        )
        caches = {"one_tr_cache": one_cache}

        summary_embs = None
        if CONDITION_MINILM_SUMMARY_COMBO in args.conditions:
            summary_embs, summary_model, summary_cache = load_or_build_summary_embeddings(args, stories, resp_lengths)
            caches["summary_cache"] = summary_cache
            caches["summary_model"] = summary_model

        window_embs = None
        if CONDITION_MINILM_WINDOW_COMBO in args.conditions:
            window_embs, window_dim, window_cache = load_or_build_window_embeddings(args, subject, stories, resp_lengths)
            if window_dim != one_dim:
                raise ValueError(f"Window embedding dim {window_dim} != 1TR dim {one_dim}")
            caches["window_cache"] = window_cache

        subject_dir = out_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for condition in args.conditions:
            out_path = subject_dir / f"encoding_model_{condition}.npz"
            if args.skip_existing and out_path.exists():
                print(f"[{subject} / {condition}] reusing {out_path}")
                row = load_existing_summary(out_path)
            else:
                row = train_condition(
                    subject,
                    condition,
                    args,
                    stories,
                    train_stories,
                    val_stories,
                    voxels_base,
                    responses,
                    one_tr,
                    summary_embs,
                    window_embs,
                    caches,
                    out_path,
                )
            rows.append(row)

        summary_path = subject_dir / "minilm_combo_encoding_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({row["condition"]: row for row in rows}, f, indent=2, sort_keys=True)
        write_encoding_csv(subject_dir / "minilm_combo_encoding_summary.csv", rows)
        print(f"[{subject}] wrote {summary_path}")


if __name__ == "__main__":
    main()
