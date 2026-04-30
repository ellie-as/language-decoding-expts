#!/usr/bin/env python3
"""Per-lag MiniLM encoding model on full_frontal voxels.

For each lag L in --lags (default 1..10):
    X = MiniLM embedding of the words spoken inside a single 1-TR window at chunk
        index i.
    Y = brain response at TR index (i + L) for every voxel in BA_full_frontal.

A separate ridge regression is fit per lag (per-voxel alpha selected via LOO
``RidgeCV``). Models are trained on a story-grouped train split and scored on a
held-out story split, giving a per-voxel Pearson r at each lag. The full
``corrs`` matrix (n_lags x n_voxels) plus per-lag npz files are written under
``--output-dir/<tag>/``.

Run ``analyze_lag_preference.py --results-dir <out>`` afterwards for a per-ROI
breakdown of preferred lags.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from _shared import load_or_build_chunk_embeddings  # noqa: E402
from utils_resp import get_resp  # noqa: E402

SUBJECT_TO_UTS = rse.SUBJECT_TO_UTS
LOCAL_DEFAULT_BA_DIR = rse.LOCAL_DEFAULT_BA_DIR
LOCAL_DEFAULT_SUMMARIES_DIR = rse.LOCAL_DEFAULT_SUMMARIES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lag_pref_encoding")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="S1", choices=sorted(SUBJECT_TO_UTS.keys()))
    p.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
        help="Training sessions to pull stories from (default: same as run_summaries_encoding).",
    )
    p.add_argument("--stories", nargs="+", default=None,
                   help="Explicit list of stories. Overrides --sessions.")
    p.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Lags (in TRs) to evaluate. Default: 1..10.",
    )
    p.add_argument("--feature-model", default="embedding", choices=["embedding", "gtr-base"])
    p.add_argument("--chunk-trs", type=int, default=1, help="Text window length in TRs (default 1).")
    p.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0],
        help="Alpha grid for per-voxel RidgeCV (LOO).",
    )
    p.add_argument("--voxel-chunk-size", type=int, default=5_000,
                   help="How many voxels to fit at once inside RidgeCV (memory control).")
    p.add_argument("--val-story-count", type=int, default=8,
                   help="If --val-stories not given, hold this many random stories.")
    p.add_argument("--val-stories", nargs="+", default=None,
                   help="Explicit list of held-out stories.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--ba-dir",
        default=str(LOCAL_DEFAULT_BA_DIR),
        help="Brodmann ROI dir (expects <UTSxx>/BA_full_frontal.json).",
    )
    p.add_argument(
        "--embedding-cache-dir",
        default=str(REPO_DIR / "27-04-expts" / "cache"),
        help="Where to cache MiniLM chunk embeddings (shares layout with 27-04-expts).",
    )
    p.add_argument("--embed-batch-size", type=int, default=256)

    p.add_argument(
        "--data-root",
        default=None,
        help="Override the data root (must contain data_train, data_lm, models). "
             "If --local-compute-mode is set without this, --mounted-project-root is used.",
    )
    p.add_argument(
        "--local-compute-mode",
        action="store_true",
        help="Read data from --mounted-project-root while writing outputs locally.",
    )
    p.add_argument(
        "--mounted-project-root",
        default="/Volumes/ellie/language-decoding-expts",
        help="Path where the cluster project tree is mounted (default: macOS Volumes mount).",
    )
    p.add_argument(
        "--summaries-dir",
        default=str(LOCAL_DEFAULT_SUMMARIES_DIR),
        help="Unused but required by configure_local_compute_mode.",
    )

    p.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "results"),
        help="Where to write per-lag npz files and summary CSVs.",
    )
    p.add_argument("--tag", default=None,
                   help="Subdirectory name (default derived from feature/lags/seed).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------


def configure_data_root(args: argparse.Namespace) -> Path | None:
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        for sub in ("data_train", "data_lm", "models"):
            target = root / sub
            if not target.is_dir():
                raise FileNotFoundError(f"--data-root missing required dir: {target}")
        config.DATA_TRAIN_DIR = str(root / "data_train")
        config.DATA_LM_DIR = str(root / "data_lm")
        config.MODEL_DIR = str(root / "models")
        config.DATA_TEST_DIR = str(root / "data_test")
        if not Path(args.ba_dir).is_dir():
            cand = root / "ba_indices"
            if cand.is_dir():
                args.ba_dir = str(cand)
        log.info("Using --data-root: %s", root)
        return root
    if args.local_compute_mode:
        return rse.configure_local_compute_mode(args)
    return None


def load_stories(args: argparse.Namespace) -> List[str]:
    if args.stories:
        return list(args.stories)
    sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
    with open(sess_to_story_path, encoding="utf-8") as f:
        sess_to_story = json.load(f)
    stories: List[str] = []
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def split_stories(stories: List[str], args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    if args.val_stories:
        val = [s for s in args.val_stories if s in stories]
        missing = sorted(set(args.val_stories) - set(val))
        if missing:
            raise ValueError(f"--val-stories not found in story pool: {missing}")
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = list(stories)
        rng.shuffle(shuffled)
        val = sorted(shuffled[: max(1, int(args.val_story_count))])
    val_set = set(val)
    train = [s for s in stories if s not in val_set]
    if not train or not val:
        raise ValueError(f"Empty train ({len(train)}) or val ({len(val)}) split.")
    return train, val


def load_full_frontal_voxels(subject: str, total_voxels: int, ba_dir: str) -> np.ndarray:
    uts_id = SUBJECT_TO_UTS.get(subject)
    if not uts_id:
        raise ValueError(f"Unknown subject: {subject!r}")
    path = Path(ba_dir).expanduser().resolve() / uts_id / "BA_full_frontal.json"
    if not path.is_file():
        raise FileNotFoundError(f"Full-frontal ROI file not found: {path}")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    vox = np.asarray(next(iter(payload.values())), dtype=np.int64)
    vox = np.sort(np.unique(vox))
    vox = vox[(vox >= 0) & (vox < int(total_voxels))]
    if vox.size == 0:
        raise ValueError(f"Full-frontal ROI is empty for subject {subject!r}.")
    return vox


def stack_lag(
    embeddings: Dict[str, np.ndarray],
    responses: Dict[str, np.ndarray],
    stories: Sequence[str],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for story in stories:
        emb = embeddings[story]
        resp = responses[story]
        n_chunks = int(emb.shape[0])
        if resp.shape[0] < n_chunks + lag:
            raise ValueError(
                f"{story}: response length {resp.shape[0]} < n_chunks+lag = {n_chunks + lag}"
            )
        xs.append(emb)
        ys.append(resp[lag : lag + n_chunks])
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32)


# ---------------------------------------------------------------------------
# Ridge fit / score
# ---------------------------------------------------------------------------


def per_voxel_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64) - pred.mean(axis=0, keepdims=True)
    true = true.astype(np.float64) - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred * pred).sum(axis=0) * (true * true).sum(axis=0))
    out = np.divide(
        (pred * true).sum(axis=0),
        denom,
        out=np.zeros(pred.shape[1], dtype=np.float64),
        where=denom > 0,
    )
    return out.astype(np.float32)


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    alphas: Sequence[float],
    voxel_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """LOO RidgeCV with per-voxel alpha. Returns (pred_val, best_alphas) on the
    original target scale."""
    from sklearn.linear_model import RidgeCV

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    x_train_z = ((x_train - x_mean) / x_std).astype(np.float32)
    x_val_z = ((x_val - x_mean) / x_std).astype(np.float32)
    y_train_z = ((y_train - y_mean) / y_std).astype(np.float32)

    n_voxels = y_train_z.shape[1]
    pred = np.zeros((x_val_z.shape[0], n_voxels), dtype=np.float32)
    best_alphas = np.zeros(n_voxels, dtype=np.float32)

    chunk = max(1, int(voxel_chunk_size))
    n_chunks = (n_voxels + chunk - 1) // chunk
    for ci, start in enumerate(range(0, n_voxels, chunk), start=1):
        end = min(start + chunk, n_voxels)
        t0 = time.time()
        model = RidgeCV(alphas=list(alphas), alpha_per_target=True)
        model.fit(x_train_z, y_train_z[:, start:end])
        pred_z = model.predict(x_val_z).astype(np.float32)
        pred[:, start:end] = (pred_z * y_std[start:end] + y_mean[start:end]).astype(np.float32)
        chunk_alpha = np.asarray(model.alpha_, dtype=np.float32)
        if chunk_alpha.ndim == 0:
            chunk_alpha = np.full(end - start, float(chunk_alpha), dtype=np.float32)
        best_alphas[start:end] = chunk_alpha
        log.info(
            "  voxel chunk %d/%d (%d:%d) done in %.1fs",
            ci, n_chunks, start, end, time.time() - t0,
        )
    return pred, best_alphas


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def build_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    return (
        f"{args.subject}__{args.feature_model}"
        f"__lags{args.lags[0]}-{args.lags[-1]}__chunk{args.chunk_trs}tr"
        f"__seed{args.seed}"
    )


def write_summary_csv(
    csv_path: Path,
    lags: Sequence[int],
    corrs_by_lag: np.ndarray,
) -> None:
    rows = []
    for li, lag in enumerate(lags):
        c = corrs_by_lag[li]
        rows.append({
            "lag": int(lag),
            "n_voxels": int(c.size),
            "mean_r": float(c.mean()),
            "median_r": float(np.median(c)),
            "p95_r": float(np.quantile(c, 0.95)),
            "max_r": float(c.max()),
            "n_r_gt_0_05": int((c > 0.05).sum()),
            "n_r_gt_0_1": int((c > 0.10).sum()),
            "n_r_gt_0_2": int((c > 0.20).sum()),
        })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.lags = sorted({int(l) for l in args.lags})
    if args.lags[0] < 0:
        raise ValueError("--lags must be non-negative.")
    np.random.seed(args.seed)

    mounted_root = configure_data_root(args)
    if mounted_root is not None:
        log.info("Mounted project root: %s", mounted_root)

    stories = load_stories(args)
    train_stories, val_stories = split_stories(stories, args)
    log.info(
        "Stories: %d total | %d train | %d val", len(stories), len(train_stories), len(val_stories)
    )
    log.info("Validation stories: %s", ", ".join(val_stories))

    log.info("Loading sample response to determine voxel count")
    sample_resp = get_resp(args.subject, [stories[0]], stack=True, vox=None)
    total_voxels = int(sample_resp.shape[1])
    voxels = load_full_frontal_voxels(args.subject, total_voxels, args.ba_dir)
    log.info("Full-frontal voxels: %d / %d", len(voxels), total_voxels)

    log.info("Loading full_frontal responses for %d stories", len(stories))
    responses_by_story = get_resp(args.subject, stories, stack=False, vox=voxels)
    responses_by_story = {s: arr.astype(np.float32) for s, arr in responses_by_story.items()}
    resp_lengths = {s: int(r.shape[0]) for s, r in responses_by_story.items()}

    max_lag = max(args.lags)
    emb_args = argparse.Namespace(
        subject=args.subject,
        embedding_cache_dir=args.embedding_cache_dir,
        feature_model=args.feature_model,
        chunk_trs=int(args.chunk_trs),
        lag_trs=int(max_lag),
        embed_batch_size=int(args.embed_batch_size),
    )
    embeddings, emb_dim, cache_path = load_or_build_chunk_embeddings(
        emb_args,
        stories,
        resp_lengths,
        response_root=config.DATA_TRAIN_DIR,
    )
    log.info(
        "%s embeddings: dim=%d, %d stories, cache=%s",
        args.feature_model, emb_dim, len(embeddings), cache_path,
    )

    out_dir = Path(args.output_dir).expanduser().resolve() / build_tag(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_lag_dir = out_dir / "per_lag"
    per_lag_dir.mkdir(parents=True, exist_ok=True)

    n_voxels = int(voxels.size)
    n_lags = len(args.lags)
    corrs_by_lag = np.zeros((n_lags, n_voxels), dtype=np.float32)
    best_alphas_by_lag = np.zeros((n_lags, n_voxels), dtype=np.float32)

    # Cache identical X arrays across lags (text features do not depend on lag).
    x_train_full, _ = stack_lag(embeddings, responses_by_story, train_stories, args.lags[0])
    x_val_full, _ = stack_lag(embeddings, responses_by_story, val_stories, args.lags[0])
    log.info(
        "Inputs: X_train=%s X_val=%s (emb_dim=%d)",
        x_train_full.shape, x_val_full.shape, emb_dim,
    )

    for li, lag in enumerate(args.lags):
        log.info("==== lag=%d ====", lag)
        t0 = time.time()
        _, y_train = stack_lag(embeddings, responses_by_story, train_stories, lag)
        _, y_val = stack_lag(embeddings, responses_by_story, val_stories, lag)
        log.info("  Y_train=%s  Y_val=%s", y_train.shape, y_val.shape)

        pred_val, best_alphas = fit_ridge_predict(
            x_train_full, y_train, x_val_full,
            alphas=args.ridge_alphas,
            voxel_chunk_size=args.voxel_chunk_size,
        )
        corrs = per_voxel_corr(pred_val, y_val)
        elapsed = time.time() - t0

        log.info(
            "  lag=%d mean_r=%.4f  median_r=%.4f  p95_r=%.4f  max_r=%.4f  n(r>0.1)=%d  n(r>0.2)=%d  (%.1fs)",
            lag,
            float(corrs.mean()),
            float(np.median(corrs)),
            float(np.quantile(corrs, 0.95)),
            float(corrs.max()),
            int((corrs > 0.10).sum()),
            int((corrs > 0.20).sum()),
            elapsed,
        )

        np.savez(
            per_lag_dir / f"lag{lag:02d}.npz",
            corrs=corrs,
            best_alphas=best_alphas,
            voxels=voxels,
            lag=int(lag),
            elapsed_sec=float(elapsed),
            train_stories=np.array(train_stories),
            val_stories=np.array(val_stories),
            feature_model=args.feature_model,
            chunk_trs=int(args.chunk_trs),
        )

        corrs_by_lag[li] = corrs
        best_alphas_by_lag[li] = best_alphas

    np.savez(
        out_dir / "lag_corrs.npz",
        corrs=corrs_by_lag,
        best_alphas=best_alphas_by_lag,
        voxels=voxels,
        lags=np.array(args.lags, dtype=int),
        feature_model=args.feature_model,
        chunk_trs=int(args.chunk_trs),
        train_stories=np.array(train_stories),
        val_stories=np.array(val_stories),
        ridge_alphas=np.array(args.ridge_alphas, dtype=float),
        subject=args.subject,
    )

    write_summary_csv(out_dir / "lag_summary.csv", args.lags, corrs_by_lag)

    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject": args.subject,
                "feature_model": args.feature_model,
                "chunk_trs": int(args.chunk_trs),
                "lags": list(args.lags),
                "ridge_alphas": list(args.ridge_alphas),
                "voxel_chunk_size": int(args.voxel_chunk_size),
                "train_stories": train_stories,
                "val_stories": val_stories,
                "n_voxels": n_voxels,
                "embedding_dim": emb_dim,
                "embedding_cache": cache_path,
                "ba_dir": str(Path(args.ba_dir).expanduser().resolve()),
                "data_train_dir": config.DATA_TRAIN_DIR,
                "seed": int(args.seed),
            },
            f,
            indent=2,
        )

    log.info("Wrote per-lag results: %s", per_lag_dir)
    log.info("Wrote combined npz: %s", out_dir / "lag_corrs.npz")
    log.info("Wrote summary CSV : %s", out_dir / "lag_summary.csv")
    log.info("Wrote config      : %s", config_path)
    log.info("Run analyze_lag_preference.py --results-dir %s for the breakdown.", out_dir)


if __name__ == "__main__":
    main()
