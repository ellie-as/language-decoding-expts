#!/usr/bin/env python3
"""Self-contained Huth-style encoding-model voxel ranking.

Trains a simple FIR ridge encoding model on **training stories only** and
saves a per-voxel held-out correlation map you can pass to
``train_mindeye_text.py --voxel-select-encoding-corrs`` for
"language-selective" voxel selection.

Method (per subject, training stories only):

  1. Build chunked text embeddings (or load from cache) at lag 0.
  2. For each TR ``t`` in a training story, build a feature vector by
     concatenating chunk embeddings at offsets ``t - L`` for ``L`` in
     ``--lags`` (default 1..4). Drop TRs without all valid offsets.
  3. Concat across stories. Story-grouped K-fold cross-validation:
     for each fold, fit ridge with multi-alpha SVD path, pick the best
     alpha per voxel by held-out correlation. Average per-voxel held-out
     correlation across folds.
  4. Save ``corrs`` (length n_roi_voxels) and ``voxels`` (full-brain
     indices) to an ``.npz`` file. This is the same format that
     ``run_context_encoding.py`` produces and that
     ``mindeye_text/data.py`` knows how to load.

Example (on the cluster, all three subjects, full_frontal ROI):

    python -m mindeye_text.encoding_rank \\
      --subjects S1 S2 S3 \\
      --roi full_frontal \\
      --feature-model gtr-base \\
      --lags 1 2 3 4 \\
      --n-folds 5 \\
      --alphas 10 100 1000 10000 \\
      --output-dir mindeye_text/cache/encoding_corrs

Then for training:

    VOXEL_SELECT_CORRS="mindeye_text/cache/encoding_corrs/{subject}__gtr-base__full_frontal__lags1-2-3-4.npz" \\
    VOXEL_SELECT_TOPK=2000 \\
    bash mindeye_text/run_train.sh ...

Notes
-----
* Uses *only* the training stories from the current ``--holdout-stories``
  / ``--holdout-count`` setting, so there is no test-set leakage in
  voxel selection. Re-run if you change the held-out story set.
* Designed to be self-contained inside ``mindeye_text/``: imports only
  from ``_shared.py`` (which itself only touches always-present
  top-level project modules).
* CPU is fine — the SVD-based ridge sweep across ~10 alphas costs
  well under a minute per subject for ~25k samples and 25k voxels.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import (  # noqa: E402
    EMBEDDING_MODELS,
    load_or_build_chunk_embeddings,
    load_responses_by_story,
    load_stories,
    resolve_response_root,
    resolve_roi,
    rse,
)


log = logging.getLogger("mindeye_text.encoding_rank")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    p.add_argument("--stories", nargs="+", default=None,
                   help="Optional explicit story list. If unset, derived from --sessions.")
    p.add_argument("--holdout-stories", nargs="+", default=None)
    p.add_argument("--holdout-count", type=int, default=rse.DEFAULT_HOLDOUT_COUNT)
    p.add_argument("--no-story-holdout", action="store_true",
                   help="If set, fit on all stories (NOT recommended for "
                        "voxel selection — would leak into test).")
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    p.add_argument("--local-cache-root", default=str(rse.DEFAULT_LOCAL_CACHE_ROOT))
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--embedding-cache-dir", default="27-04-expts/cache",
                   help="Where chunked text embeddings are cached (matches train script).")
    p.add_argument("--roi", default="full_frontal")
    p.add_argument("--feature-model", default="gtr-base", choices=list(EMBEDDING_MODELS.keys()))
    p.add_argument("--normalize-targets", action="store_true",
                   help="L2-normalize raw embedding targets before z-score (matches train).")
    p.add_argument("--chunk-trs", type=int, default=5,
                   help="Width of the text chunk in TRs. Match the train-time chunk_trs.")

    p.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="FIR encoding lags (TRs). For predicting brain at TR t, use chunk "
                        "embeddings at indices t-L for L in lags. Default 1..4 covers the "
                        "canonical HRF.")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Story-grouped K-fold for held-out correlation.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
                   help="Ridge regularisation parameters; per-voxel best is picked by held-out r.")
    p.add_argument("--voxel-chunk-size", type=int, default=8192,
                   help="Process this many voxels per ridge solve (memory).")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output-dir", default="mindeye_text/cache/encoding_corrs",
                   help="Directory to write per-subject .npz files.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _voxel_zscore_concat(
    resp_by_story: Dict[str, np.ndarray],
    stories: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    concat = np.vstack([resp_by_story[s] for s in stories]).astype(np.float32)
    mean = concat.mean(axis=0).astype(np.float32)
    std = concat.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    out = {
        s: np.nan_to_num((resp_by_story[s] - mean) / std).astype(np.float32)
        for s in stories
    }
    return out, mean, std


def _build_fir_xy(
    chunk_emb_by_story: Dict[str, np.ndarray],
    resp_z_by_story: Dict[str, np.ndarray],
    stories: List[str],
    lags: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, Y, story_ids)``.

    For each TR ``t`` in story ``s``, ``X[row]`` is ``concat(chunk_emb[t-L]
    for L in lags)`` and ``Y[row]`` is ``resp_z[t]`` (z-scored voxels).
    Only TRs ``t`` such that ``t-L`` is a valid chunk index for every ``L``
    are kept. Story id is recorded for grouped CV.
    """
    sorted_lags = sorted(lags)
    max_lag = sorted_lags[-1]
    min_lag = sorted_lags[0]

    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    story_ids: List[int] = []
    n_dropped = 0
    for story_idx, story in enumerate(stories):
        emb = chunk_emb_by_story[story]
        resp = resp_z_by_story[story]
        n_chunks = int(emb.shape[0])
        n_trs = int(resp.shape[0])
        t_lo = max_lag
        t_hi = min(n_trs, n_chunks + min_lag)
        if t_hi <= t_lo:
            log.warning(
                "[%s] no valid TRs after FIR lag (n_TRs=%d, n_chunks=%d, lags=%s)",
                story, n_trs, n_chunks, sorted_lags,
            )
            n_dropped += 1
            continue
        feats = [emb[t_lo - lag : t_hi - lag] for lag in sorted_lags]
        X_block = np.hstack(feats).astype(np.float32)
        Y_block = resp[t_lo:t_hi].astype(np.float32)
        Xs.append(X_block)
        Ys.append(Y_block)
        story_ids.extend([story_idx] * (t_hi - t_lo))
    if not Xs:
        raise RuntimeError("No usable stories produced FIR features.")
    if n_dropped:
        log.warning("Dropped %d stories with insufficient TRs.", n_dropped)
    return (
        np.vstack(Xs).astype(np.float32),
        np.vstack(Ys).astype(np.float32),
        np.asarray(story_ids, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Ridge with multi-alpha SVD (per-voxel best alpha)
# ---------------------------------------------------------------------------


def _pearson_columnwise(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between two (n, V) matrices."""
    Yt = Y_true - Y_true.mean(axis=0)
    Yp = Y_pred - Y_pred.mean(axis=0)
    nt = np.linalg.norm(Yt, axis=0)
    np_ = np.linalg.norm(Yp, axis=0)
    nt[nt == 0] = 1.0
    np_[np_ == 0] = 1.0
    return ((Yt * Yp).sum(axis=0) / (nt * np_)).astype(np.float32)


def _ridge_svd_fold(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    alphas: List[float],
    voxel_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One outer fold. Returns ``(corrs_best, alpha_best)`` per voxel.

    Uses the SVD of ``X_train`` once and sweeps alphas analytically; voxels
    are processed in chunks to keep memory bounded.
    """
    # Standardize features (per-feature) using training stats only.
    mu = X_train.mean(axis=0).astype(np.float32)
    sigma = X_train.std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
    # Project X_val into the SVD basis once: X_val @ V = U_val_eff
    X_val_V = X_val @ Vt.T  # (n_val, k)
    s2 = s * s

    n_voxels = Y_train.shape[1]
    best_corr = np.full(n_voxels, -np.inf, dtype=np.float32)
    best_alpha = np.zeros(n_voxels, dtype=np.float32)

    for v0 in range(0, n_voxels, voxel_chunk_size):
        v1 = min(v0 + voxel_chunk_size, n_voxels)
        Y_tr = Y_train[:, v0:v1]
        Y_va = Y_val[:, v0:v1]
        UtY = U.T @ Y_tr  # (k, V_chunk)
        for alpha in alphas:
            # ridge solution in SVD basis: w_basis = (s / (s^2 + alpha)) * UtY
            d = (s / (s2 + alpha)).astype(np.float32)  # (k,)
            preds = X_val_V @ (d[:, None] * UtY)        # (n_val, V_chunk)
            r = _pearson_columnwise(Y_va, preds)
            better = r > best_corr[v0:v1]
            best_corr[v0:v1][better] = r[better]
            best_alpha[v0:v1][better] = float(alpha)
    return best_corr, best_alpha


def _kfold_story_groups(
    story_ids: np.ndarray, n_folds: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(story_ids)
    perm = rng.permutation(unique)
    folds = np.array_split(perm, n_folds)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in folds:
        val_mask = np.isin(story_ids, f)
        train_mask = ~val_mask
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        out.append((train_mask, val_mask))
    return out


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------


def encode_subject(args: argparse.Namespace, subject: str, out_dir: Path) -> Path:
    args = argparse.Namespace(**vars(args))
    args.subject = subject

    response_root, mounted_root = resolve_response_root(args)
    if args.local_compute_mode and mounted_root is not None:
        stories_for_cache = load_stories(args, response_root)
        response_root = rse.stage_local_response_cache(
            subject,
            stories_for_cache,
            mounted_data_train_dir=str(mounted_root / "data_train"),
            cache_root=Path(args.local_cache_root).expanduser().resolve(),
        )

    stories = load_stories(args, response_root)
    train_stories, test_stories = rse.split_story_list(stories, args)
    if args.no_story_holdout:
        log.warning(
            "[%s] --no-story-holdout: fitting encoding model on ALL stories. "
            "This is OK for exploration but will leak into decoder evaluation.",
            subject,
        )
        train_stories = list(stories)

    train_resp_lengths, total_voxels = rse.load_resp_info(
        subject, train_stories, data_train_dir=response_root
    )
    test_resp_lengths, _ = rse.load_resp_info(
        subject, test_stories, data_train_dir=response_root
    )
    resp_lengths = {**train_resp_lengths, **test_resp_lengths}

    roi_name, vox = resolve_roi(args, total_voxels)
    log.info("[%s] ROI %s: %d voxels", subject, roi_name, len(vox))

    raw = load_responses_by_story(train_stories, subject, vox, response_root)
    resp_z, _vox_mean, _vox_std = _voxel_zscore_concat(raw, train_stories)

    # Use lag=0 chunk embeddings (FIR lag is applied below); chunk_trs from CLI.
    args_for_emb = argparse.Namespace(
        **{
            **vars(args),
            "subject": subject,
            "lag_trs": 0,
            "chunk_trs": int(args.chunk_trs),
        }
    )
    embeddings_by_story, emb_dim, _cache_path = load_or_build_chunk_embeddings(
        args_for_emb, stories, resp_lengths, response_root=response_root
    )
    train_emb = {s: embeddings_by_story[s] for s in train_stories}

    if args.normalize_targets:
        for s in list(train_emb.keys()):
            arr = train_emb[s]
            n = np.linalg.norm(arr, axis=1, keepdims=True)
            n[n == 0] = 1.0
            train_emb[s] = (arr / n).astype(np.float32)

    log.info(
        "[%s] %d train stories, lags=%s, emb_dim=%d → FIR feat dim=%d",
        subject, len(train_stories), sorted(args.lags), emb_dim,
        emb_dim * len(args.lags),
    )

    X, Y, story_ids = _build_fir_xy(train_emb, resp_z, train_stories, list(args.lags))
    log.info("[%s] X=%s, Y=%s, %d unique stories",
             subject, X.shape, Y.shape, np.unique(story_ids).size)

    # K-fold story-grouped CV; average per-voxel best held-out r.
    folds = _kfold_story_groups(story_ids, int(args.n_folds), seed=int(args.seed))
    per_fold_corrs: List[np.ndarray] = []
    for fold_i, (tr_mask, va_mask) in enumerate(folds, 1):
        t0 = time.time()
        corrs, alphas_best = _ridge_svd_fold(
            X[tr_mask], Y[tr_mask], X[va_mask], Y[va_mask],
            alphas=list(args.alphas),
            voxel_chunk_size=int(args.voxel_chunk_size),
        )
        elapsed = time.time() - t0
        log.info(
            "[%s] fold %d/%d: train=%d, val=%d | r mean=%.3f, max=%.3f, "
            ">0.1 frac=%.3f | %.1fs",
            subject, fold_i, len(folds), int(tr_mask.sum()), int(va_mask.sum()),
            float(corrs.mean()), float(corrs.max()),
            float((corrs > 0.1).mean()), elapsed,
        )
        per_fold_corrs.append(corrs)

    corrs_mean = np.stack(per_fold_corrs, axis=0).mean(axis=0).astype(np.float32)
    log.info(
        "[%s] per-voxel held-out r (mean across %d folds): "
        "mean=%.3f, max=%.3f, n(r>0.1)=%d, n(r>0.2)=%d",
        subject, len(folds),
        float(corrs_mean.mean()), float(corrs_mean.max()),
        int((corrs_mean > 0.1).sum()), int((corrs_mean > 0.2).sum()),
    )

    # Save in the .npz format mindeye_text/data.py knows how to load.
    out_dir.mkdir(parents=True, exist_ok=True)
    lag_tag = "-".join(str(int(L)) for L in sorted(args.lags))
    out_path = out_dir / (
        f"{subject}__{args.feature_model}__{roi_name}__lags{lag_tag}.npz"
    )
    np.savez(
        out_path,
        corrs=corrs_mean,
        voxels=np.asarray(vox, dtype=np.int64),
        roi_name=np.array(roi_name),
        feature_model=np.array(args.feature_model),
        lags=np.asarray(sorted(args.lags), dtype=np.int32),
        alphas=np.asarray(args.alphas, dtype=np.float32),
        n_folds=np.int32(int(args.n_folds)),
        n_train_samples=np.int32(int(X.shape[0])),
        n_train_stories=np.int32(int(np.unique(story_ids).size)),
    )
    log.info("[%s] wrote: %s", subject, out_path)
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    log.info("Output directory: %s", out_dir)
    log.info(
        "Lags=%s | n-folds=%d | alphas=%s | feature_model=%s",
        sorted(args.lags), args.n_folds, args.alphas, args.feature_model,
    )

    for subject in args.subjects:
        log.info("=" * 60)
        log.info("Subject %s", subject)
        encode_subject(args, subject, out_dir)


if __name__ == "__main__":
    main()
