#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "fmri_text_mae" / "src"
DECODING = ROOT / "decoding"
for path in (ROOT, SRC, DECODING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from evaluate_retrieval import retrieval_metrics  # noqa: E402
from hrf_alignment import response_tr_times, words_in_lagged_window  # noqa: E402
from utils import ensure_dir, save_json, set_seed  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_stim import get_story_wordseqs  # noqa: E402


TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM
ROI_FILES = ["BA_10.json", "BA_6.json", "BA_8.json", "BA_9_46.json", "BROCA.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep sklearn fMRI->text-embedding retrieval baselines using BA_full_frontal voxels."
    )
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--holdout-stories", nargs="+", default=None)
    parser.add_argument("--holdout-count", type=int, default=5)
    parser.add_argument("--no-story-holdout", action="store_true")
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--include-val-in-train", action="store_true")
    parser.add_argument("--roi", default="BA_full_frontal", help="ROI JSON stem, e.g. BA_full_frontal, BA_10, BROCA, or full_frontal.")
    parser.add_argument("--ba-dir", default=str(ROOT / "ba_indices"))
    parser.add_argument("--output-dir", default="fmri_text_mae/outputs/retrieval_sweeps/S1_full_frontal")
    parser.add_argument("--response-root", default=None)
    parser.add_argument("--local-compute-mode", action="store_true")
    parser.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    parser.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))

    parser.add_argument("--tr-sec", type=float, default=2.0)
    parser.add_argument("--window-lens-tr", nargs="+", type=int, default=[4, 8, 10])
    parser.add_argument("--stride-tr", type=int, default=2)
    parser.add_argument("--hrf-lags-sec", nargs="+", type=float, default=[2.0, 4.0, 6.0])

    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)

    parser.add_argument("--models", nargs="+", default=["pca_ridge", "pls"])
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[10.0, 100.0, 1000.0])
    parser.add_argument("--brain-pca", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--target-pca", nargs="+", type=int, default=[0, 64])
    parser.add_argument("--pls-components", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--elasticnet-alphas", nargs="+", type=float, default=[0.001, 0.01])
    parser.add_argument("--elasticnet-l1-ratios", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--save-best-predictions", action="store_true")
    return parser.parse_args()


def load_story_split(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    if args.local_compute_mode:
        rse.configure_local_compute_mode(args)
    stories = rse.load_story_list(args)
    train_plus_val, test = rse.split_story_list(stories, args)
    if args.include_val_in_train or args.val_count <= 0:
        return train_plus_val, [], test
    if len(train_plus_val) <= args.val_count:
        raise ValueError("Not enough training stories to reserve validation stories.")
    return train_plus_val[:-args.val_count], train_plus_val[-args.val_count:], test


def subject_to_uts(subject: str) -> str:
    if subject in rse.SUBJECT_TO_UTS:
        return rse.SUBJECT_TO_UTS[subject]
    if subject.startswith("UTS"):
        return subject
    raise ValueError(f"Cannot map subject {subject!r} to UTS ID.")


def load_roi_indices(subject: str, roi: str, ba_dir: str | Path, total_voxels: int | None = None) -> np.ndarray:
    roi = "BA_full_frontal" if roi == "full_frontal" else roi
    subj_dir = Path(ba_dir) / subject_to_uts(subject)
    roi_path = subj_dir / f"{roi}.json"
    if roi_path.exists():
        with open(roi_path, encoding="utf-8") as f:
            data = json.load(f)
        values = np.asarray(next(iter(data.values())), dtype=np.int64)
    elif roi == "BA_full_frontal":
        full: set[int] = set()
        for name in ROI_FILES:
            path = subj_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing ROI file needed for full frontal union: {path}")
            with open(path, encoding="utf-8") as f:
                full.update(int(v) for v in next(iter(json.load(f).values())))
        values = np.asarray(sorted(full), dtype=np.int64)
    else:
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    if total_voxels is not None:
        values = values[values < total_voxels]
    if values.size == 0:
        raise ValueError(f"ROI {roi} for {subject} has no valid voxels.")
    return np.sort(values)


def fit_train_zscore(resp_by_story: dict[str, np.ndarray], train_stories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train = np.vstack([resp_by_story[s] for s in train_stories]).astype(np.float32)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_windows(
    stories: list[str],
    resp_by_story: dict[str, np.ndarray],
    wordseqs: dict,
    mean: np.ndarray,
    std: np.ndarray,
    tr_sec: float,
    hrf_lag_sec: float,
    window_len_tr: int,
    stride_tr: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    bold, texts, story_ids = [], [], []
    for story in stories:
        resp = np.nan_to_num((resp_by_story[story].astype(np.float32) - mean) / std)
        tr_times = response_tr_times(wordseqs[story], resp.shape[0], tr_sec, TRIM_START, TRIM_END)
        max_start = resp.shape[0] - window_len_tr
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride_tr):
            end = start + window_len_tr
            start_sec = float(tr_times[start])
            end_sec = float(tr_times[end - 1] + tr_sec)
            text = words_in_lagged_window(wordseqs[story], start_sec, end_sec, hrf_lag_sec)
            if not text:
                continue
            bold.append(resp[start:end].reshape(-1))
            texts.append(text)
            story_ids.append(story)
    return np.asarray(bold, dtype=np.float32), texts, np.asarray(story_ids, dtype=object)


def zscore_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return np.nan_to_num((x_train - mean) / std).astype(np.float32), np.nan_to_num((x_test - mean) / std).astype(np.float32)


def embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def maybe_pca_x(x_train: np.ndarray, x_test: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if n_components <= 0:
        return x_train, x_test, "none"
    from sklearn.decomposition import PCA

    n_comp = min(int(n_components), x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized")
    xtr = pca.fit_transform(x_train).astype(np.float32)
    xte = pca.transform(x_test).astype(np.float32)
    xtr, xte = zscore_train_test(xtr, xte)
    return xtr, xte, f"{n_comp}:{pca.explained_variance_ratio_.sum():.4f}"


def maybe_pca_y(y_train_z: np.ndarray, y_test: np.ndarray, n_components: int, seed: int):
    if n_components <= 0:
        return y_train_z, None, "none"
    from sklearn.decomposition import PCA

    n_comp = min(int(n_components), y_train_z.shape[0] - 1, y_train_z.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    ytr = pca.fit_transform(y_train_z).astype(np.float32)
    return ytr, pca, f"{n_comp}:{pca.explained_variance_ratio_.sum():.4f}"


def restore_embedding_prediction(pred_z, target_pca, emb_mean, emb_std) -> np.ndarray:
    if target_pca is not None:
        pred_z = target_pca.inverse_transform(pred_z)
    return (pred_z.astype(np.float32) * emb_std + emb_mean).astype(np.float32)


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
    if kind == "elasticnet":
        from sklearn.linear_model import MultiTaskElasticNet

        model = MultiTaskElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
            random_state=int(params["seed"]),
        )
        model.fit(x_train, y_train)
        return model.predict(x_test).astype(np.float32)
    raise ValueError(f"Unknown model kind: {kind}")


def model_param_grid(args: argparse.Namespace, kind: str):
    if kind == "ridge":
        for alpha, target_pca in product(args.ridge_alphas, args.target_pca):
            yield {"alpha": alpha, "brain_pca": 0, "target_pca": target_pca}
    elif kind == "pca_ridge":
        for brain_pca, alpha, target_pca in product(args.brain_pca, args.ridge_alphas, args.target_pca):
            if brain_pca > 0:
                yield {"alpha": alpha, "brain_pca": brain_pca, "target_pca": target_pca}
    elif kind == "pls":
        for brain_pca, n_components, target_pca in product(args.brain_pca, args.pls_components, args.target_pca):
            if brain_pca > 0:
                yield {"brain_pca": brain_pca, "n_components": n_components, "target_pca": target_pca}
    elif kind == "elasticnet":
        for brain_pca, alpha, l1_ratio, target_pca in product(
            args.brain_pca, args.elasticnet_alphas, args.elasticnet_l1_ratios, args.target_pca
        ):
            if brain_pca > 0:
                yield {
                    "brain_pca": brain_pca,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "target_pca": target_pca,
                    "seed": args.random_seed,
                }
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)
    out_dir = ensure_dir(args.output_dir)

    train_stories, val_stories, test_stories = load_story_split(args)
    all_stories = train_stories + val_stories + test_stories
    total_resp = get_resp(args.subject, [all_stories[0]], stack=True, vox=None, response_root=args.response_root).shape[1]
    vox = load_roi_indices(args.subject, args.roi, args.ba_dir, total_voxels=total_resp)
    print(f"Using {len(vox)} voxels from {args.roi} for {args.subject}")

    resp_by_story = get_resp(args.subject, all_stories, stack=False, vox=vox, response_root=args.response_root)
    wordseqs = get_story_wordseqs(all_stories)
    mean, std = fit_train_zscore(resp_by_story, train_stories)

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.embedding_model)
    rows = []
    best = None
    results_path = out_dir / "sweep_results.csv"
    metadata = {
        "subject": args.subject,
        "roi": args.roi,
        "n_voxels": int(len(vox)),
        "train_stories": train_stories,
        "val_stories": val_stories,
        "test_stories": test_stories,
        "embedding_model": args.embedding_model,
    }
    save_json(metadata, out_dir / "metadata.json")
    np.save(out_dir / "voxel_indices.npy", vox)

    for window_len_tr, hrf_lag_sec in product(args.window_lens_tr, args.hrf_lags_sec):
        print(f"\nWindow={window_len_tr} TR, lag={hrf_lag_sec}s")
        x_train, train_texts, _ = make_windows(
            train_stories, resp_by_story, wordseqs, mean, std, args.tr_sec, hrf_lag_sec, window_len_tr, args.stride_tr
        )
        x_test, test_texts, _ = make_windows(
            test_stories, resp_by_story, wordseqs, mean, std, args.tr_sec, hrf_lag_sec, window_len_tr, args.stride_tr
        )
        x_train, x_test = zscore_train_test(x_train, x_test)
        y_train = embed_texts(encoder, train_texts, args.embedding_batch_size)
        y_test = embed_texts(encoder, test_texts, args.embedding_batch_size)
        emb_mean = y_train.mean(axis=0)
        emb_std = y_train.std(axis=0)
        emb_std[emb_std == 0] = 1.0
        y_train_z_full = np.nan_to_num((y_train - emb_mean) / emb_std).astype(np.float32)

        for kind in args.models:
            for params in model_param_grid(args, kind):
                xtr, xte, brain_pca_info = maybe_pca_x(x_train, x_test, int(params.get("brain_pca", 0)), args.random_seed)
                ytr, target_pca, target_pca_info = maybe_pca_y(
                    y_train_z_full, y_test, int(params.get("target_pca", 0)), args.random_seed
                )
                pred_z = fit_predict(kind, xtr, ytr, xte, params)
                pred = restore_embedding_prediction(pred_z, target_pca, emb_mean, emb_std)
                metrics = retrieval_metrics(y_test, pred)
                row = {
                    "model": kind,
                    "window_len_tr": window_len_tr,
                    "window_len_sec": window_len_tr * args.tr_sec,
                    "hrf_lag_sec": hrf_lag_sec,
                    "stride_tr": args.stride_tr,
                    "n_train": len(train_texts),
                    "n_test": len(test_texts),
                    "brain_pca": params.get("brain_pca", 0),
                    "brain_pca_info": brain_pca_info,
                    "target_pca": params.get("target_pca", 0),
                    "target_pca_info": target_pca_info,
                    "alpha": params.get("alpha", ""),
                    "l1_ratio": params.get("l1_ratio", ""),
                    "n_components": params.get("n_components", ""),
                    **metrics,
                }
                rows.append(row)
                print(row)
                if best is None or row["mrr"] > best["row"]["mrr"]:
                    best = {"row": row, "pred": pred, "true": y_test, "texts": np.asarray(test_texts, dtype=object)}

                with open(results_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    if best is not None:
        save_json(best["row"], out_dir / "best_result.json")
        if args.save_best_predictions:
            np.savez_compressed(
                out_dir / "best_predictions.npz",
                pred=best["pred"],
                true=best["true"],
                texts=best["texts"],
            )


if __name__ == "__main__":
    main()
