#!/usr/bin/env python3
"""Frontal hierarchy decoding sweep.

Primary question:
    Which frontal ROIs can recover fine- vs coarse-grained language
    representations from their voxel population?

This is the decoding complement to `run_frontal_hierarchy_encoding.py`.
It fits ROI activity -> text representation models for local transcript windows
and long-context summaries, then evaluates held-out-story retrieval.

Because decoding can favor larger ROIs, the default analysis matches voxel
counts across ROIs by repeated subsampling.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM
DEFAULT_ROIS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]
ROI_COMPONENTS = ["BA_10", "BA_9_46", "BA_8", "BA_6", "BROCA"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frontal_hierarchy_decoding")


@dataclass(frozen=True)
class TargetFamily:
    name: str
    timescale: str
    train: np.ndarray
    test: np.ndarray
    dim: int


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
    parser.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    parser.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    parser.add_argument("--output-dir", default="frontal_hierarchy_decoding_results")

    parser.add_argument("--rois", nargs="+", default=DEFAULT_ROIS)
    parser.add_argument("--include-full-frontal", action="store_true")
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-horizons", nargs="+", type=int, default=[20, 50, 200, 500])
    parser.add_argument("--include-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-window-trs", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--local-target-lags", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--models", nargs="+", default=["pca_ridge", "pls"])
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[10.0, 100.0, 1000.0])
    parser.add_argument("--brain-pca", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--target-pca", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--pls-components", nargs="+", type=int, default=[16, 32, 64])

    parser.add_argument("--match-voxel-count", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-voxels-per-roi", type=int, default=0)
    parser.add_argument("--subsample-iters", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--save-best-predictions", action="store_true")
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


def stack_by_story(mats_by_story: dict[str, np.ndarray], stories: list[str]) -> np.ndarray:
    return np.vstack([mats_by_story[story] for story in stories]).astype(np.float32)


def zscore_train_apply(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return np.nan_to_num((train - mean) / std).astype(np.float32), np.nan_to_num((test - mean) / std).astype(np.float32)


def build_recent_texts_for_story(wordseq, response_len: int, window_trs: int, target_lag: int) -> list[str]:
    words = np.asarray(wordseq.data)
    word_times = np.asarray(wordseq.data_times, dtype=np.float64)
    tr_times = np.asarray(wordseq.tr_times, dtype=np.float64)
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
        texts.append(" ".join(str(w).strip() for w in words[mask] if str(w).strip()))
    return texts


def embed_texts(model, texts: list[str], dim: int, batch_size: int) -> np.ndarray:
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    idx = [i for i, t in enumerate(texts) if t.strip()]
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


def trim_summary_vectors(story: str, vectors: np.ndarray, response_len: int) -> np.ndarray:
    if vectors.shape[0] <= TRIM_START + TRIM_END:
        raise ValueError(f"Story {story} has too few summary rows: {vectors.shape[0]}")
    trimmed = vectors[TRIM_START:-TRIM_END]
    if trimmed.shape[0] != response_len:
        raise ValueError(
            f"Story {story}: summaries trim to {trimmed.shape[0]} rows, response has {response_len}"
        )
    return trimmed


def build_target_families(
    args: argparse.Namespace,
    stories: list[str],
    train_stories: list[str],
    test_stories: list[str],
    resp_lengths: dict[str, int],
) -> list[TargetFamily]:
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model %s", args.embedding_model)
    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    emb_dim = int(encoder.get_sentence_embedding_dimension())
    wordseqs = get_story_wordseqs(stories)
    families: list[TargetFamily] = []

    if args.include_local:
        for window_trs, target_lag in product(args.local_window_trs, args.local_target_lags):
            per_story = {}
            for story in stories:
                texts = build_recent_texts_for_story(wordseqs[story], resp_lengths[story], window_trs, target_lag)
                per_story[story] = embed_texts(encoder, texts, emb_dim, args.embedding_batch_size)
            train = stack_by_story(per_story, train_stories)
            test = stack_by_story(per_story, test_stories)
            train, test = zscore_train_apply(train, test)
            name = f"local_w{window_trs}tr_lag{target_lag}tr"
            families.append(TargetFamily(name=name, timescale=f"{window_trs * 2}s_local", train=train, test=test, dim=train.shape[1]))
            log.info("Built target %s: train=%s test=%s", name, train.shape, test.shape)

    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    index = rse.build_summary_index(summaries_dir)
    summary_model = rse.resolve_summary_model(index, stories, args.summary_model)
    horizons = rse.resolve_summary_horizons(index, stories, summary_model, args.summary_horizons)
    for horizon in horizons:
        per_story = {}
        for story in stories:
            path = index[(story, summary_model, horizon)]
            texts = rse.load_summary_texts(path, story, summary_model, horizon)["texts"]
            vecs = embed_texts(encoder, texts, emb_dim, args.embedding_batch_size)
            per_story[story] = trim_summary_vectors(story, vecs, resp_lengths[story])
        train = stack_by_story(per_story, train_stories)
        test = stack_by_story(per_story, test_stories)
        train, test = zscore_train_apply(train, test)
        name = f"summary_h{horizon}"
        families.append(TargetFamily(name=name, timescale=f"{horizon}word_summary", train=train, test=test, dim=train.shape[1]))
        log.info("Built target %s: train=%s test=%s", name, train.shape, test.shape)

    return families


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def retrieval_metrics(true_emb: np.ndarray, pred_emb: np.ndarray) -> dict[str, float]:
    true_unit = l2_normalize(true_emb.astype(np.float32, copy=False))
    pred_unit = l2_normalize(pred_emb.astype(np.float32, copy=False))
    sim = pred_unit @ true_unit.T
    diag = np.diag(sim)
    ranks = 1 + (sim > diag[:, None]).sum(axis=1)
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "embedding_cosine": float(np.mean(diag)),
    }


def dim_corr(true_emb: np.ndarray, pred_emb: np.ndarray) -> float:
    yt = true_emb - true_emb.mean(axis=0, keepdims=True)
    yp = pred_emb - pred_emb.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(yt, axis=0) * np.linalg.norm(yp, axis=0)
    denom[denom == 0] = 1.0
    corrs = np.nan_to_num((yt * yp).sum(axis=0) / denom)
    return float(corrs.mean())


def fit_pca(train: np.ndarray, test: np.ndarray, n_components: int, seed: int):
    if n_components <= 0:
        return train, test, None, "none"
    from sklearn.decomposition import PCA

    n_comp = min(int(n_components), train.shape[0] - 1, train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized")
    train_p = pca.fit_transform(train).astype(np.float32)
    test_p = pca.transform(test).astype(np.float32)
    train_p, test_p = zscore_train_apply(train_p, test_p)
    return train_p, test_p, pca, f"{n_comp}:{pca.explained_variance_ratio_.sum():.4f}"


def target_pca_train(family: TargetFamily, n_components: int, seed: int):
    if n_components <= 0:
        return family.train, family.test, None, "none"
    from sklearn.decomposition import PCA

    n_comp = min(int(n_components), family.train.shape[0] - 1, family.train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    train = pca.fit_transform(family.train).astype(np.float32)
    test = pca.transform(family.test).astype(np.float32)
    return train, test, pca, f"{n_comp}:{pca.explained_variance_ratio_.sum():.4f}"


def fit_predict(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, params: dict) -> np.ndarray:
    if kind in {"ridge", "pca_ridge"}:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=float(params["alpha"]))
        model.fit(x_train, y_train)
        return model.predict(x_test).astype(np.float32)
    if kind == "pls":
        from sklearn.cross_decomposition import PLSRegression

        n_comp = min(int(params["n_components"]), x_train.shape[1], y_train.shape[1], x_train.shape[0] - 1)
        model = PLSRegression(n_components=n_comp, scale=False)
        model.fit(x_train, y_train)
        return model.predict(x_test).astype(np.float32)
    raise ValueError(f"Unknown model: {kind}")


def model_grid(args: argparse.Namespace, kind: str):
    if kind == "ridge":
        for alpha, target_pca in product(args.ridge_alphas, args.target_pca):
            yield {"alpha": alpha, "brain_pca": 0, "target_pca": target_pca}
    elif kind == "pca_ridge":
        for brain_pca, alpha, target_pca in product(args.brain_pca, args.ridge_alphas, args.target_pca):
            if brain_pca > 0:
                yield {"brain_pca": brain_pca, "alpha": alpha, "target_pca": target_pca}
    elif kind == "pls":
        for brain_pca, n_components, target_pca in product(args.brain_pca, args.pls_components, args.target_pca):
            if brain_pca > 0:
                yield {"brain_pca": brain_pca, "n_components": n_components, "target_pca": target_pca}
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def choose_voxel_subsets(
    roi_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, list[np.ndarray]]:
    if not args.match_voxel_count:
        return {roi: [idx] for roi, idx in roi_indices.items()}
    if args.n_voxels_per_roi > 0:
        n_vox = args.n_voxels_per_roi
    else:
        n_vox = min(len(idx) for idx in roi_indices.values())
    if n_vox <= 0:
        raise ValueError("Matched voxel count must be positive.")
    out = {}
    for roi, idx in roi_indices.items():
        if len(idx) < n_vox:
            raise ValueError(f"ROI {roi} has {len(idx)} voxels, cannot sample {n_vox}")
        out[roi] = [
            np.sort(rng.choice(idx, size=n_vox, replace=False)).astype(np.int64)
            for _ in range(max(1, args.subsample_iters))
        ]
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    stories = rse.load_story_list(args)
    train_stories, test_stories = rse.split_story_list(stories, args)
    if not test_stories:
        raise ValueError("This analysis requires held-out test stories.")
    resp_lengths, total_voxels = rse.load_resp_info(args.subject, stories)
    rois = list(args.rois)
    if args.include_full_frontal and "BA_full_frontal" not in rois:
        rois.append("BA_full_frontal")
    roi_indices = {
        roi: load_roi_indices(args.subject, roi, args.ba_dir, total_voxels)
        for roi in rois
    }
    voxel_subsets = choose_voxel_subsets(roi_indices, args, rng)
    targets = build_target_families(args, stories, train_stories, test_stories, resp_lengths)

    metadata = {
        "subject": args.subject,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "roi_voxel_counts": {roi: int(len(idx)) for roi, idx in roi_indices.items()},
        "matched_voxel_count": int(len(next(iter(next(iter(voxel_subsets.values())))))) if args.match_voxel_count else None,
        "embedding_model": args.embedding_model,
        "targets": [{"name": t.name, "timescale": t.timescale, "dim": t.dim} for t in targets],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    rows: list[dict] = []
    best = None
    results_path = out_dir / "frontal_hierarchy_decoding_results.csv"
    for roi, subsets in voxel_subsets.items():
        for sample_idx, vox in enumerate(subsets):
            log.info("Loading %s sample %d/%d (%d voxels)", roi, sample_idx + 1, len(subsets), len(vox))
            resp_by_story = get_resp(args.subject, stories, stack=False, vox=vox)
            x_train_raw = stack_by_story(resp_by_story, train_stories)
            x_test_raw = stack_by_story(resp_by_story, test_stories)
            x_train_raw, x_test_raw = zscore_train_apply(x_train_raw, x_test_raw)

            for target in targets:
                for kind in args.models:
                    for params in model_grid(args, kind):
                        x_train, x_test, _brain_pca, brain_pca_info = fit_pca(
                            x_train_raw, x_test_raw, int(params.get("brain_pca", 0)), args.random_seed
                        )
                        y_train, _y_test_pca, y_pca, target_pca_info = target_pca_train(
                            target, int(params.get("target_pca", 0)), args.random_seed
                        )
                        pred = fit_predict(kind, x_train, y_train, x_test, params)
                        if y_pca is not None:
                            pred_full = y_pca.inverse_transform(pred).astype(np.float32)
                        else:
                            pred_full = pred
                        metrics = retrieval_metrics(target.test, pred_full)
                        row = {
                            "roi": roi,
                            "sample_idx": sample_idx,
                            "n_voxels": int(len(vox)),
                            "target": target.name,
                            "timescale": target.timescale,
                            "target_dim": target.dim,
                            "model": kind,
                            "alpha": params.get("alpha", ""),
                            "brain_pca": params.get("brain_pca", 0),
                            "brain_pca_info": brain_pca_info,
                            "target_pca": params.get("target_pca", 0),
                            "target_pca_info": target_pca_info,
                            "n_components": params.get("n_components", ""),
                            "dim_corr": dim_corr(target.test, pred_full),
                            **metrics,
                        }
                        rows.append(row)
                        log.info(
                            "%s %s %s sample=%d top5=%.4f mrr=%.4f cos=%.4f",
                            roi, target.name, kind, sample_idx, row["top5"], row["mrr"], row["embedding_cosine"],
                        )
                        if best is None or row["mrr"] > best["row"]["mrr"]:
                            best = {"row": row, "pred": pred_full, "true": target.test}

                        with open(results_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
                            writer.writeheader()
                            writer.writerows(rows)

    if best is not None:
        with open(out_dir / "best_result.json", "w", encoding="utf-8") as f:
            json.dump(best["row"], f, indent=2)
        if args.save_best_predictions:
            np.savez_compressed(out_dir / "best_predictions.npz", pred=best["pred"], true=best["true"])
    log.info("Wrote %s", results_path)


if __name__ == "__main__":
    main()
