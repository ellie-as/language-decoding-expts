#!/usr/bin/env python3
"""
Sweep decoders for h20 embedding decoding (S1 full_frontal).

Goal: pick a model that maximizes test-set dim_r on held-out stories without
using test stories for hyperparameter selection.

Models (small, scalable set):
- Ridge (expanded alpha grid)
- Ridge + rank-k truncation (low-rank mapping)
- PLSRegression
- MultiTaskElasticNet

Metrics (held-out stories only):
- dim_r: mean Pearson r per embedding dimension across time
- retrieval_top1 / retrieval_mrr: does pred[i] retrieve true[i] among all test TRs
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config
from utils_resp import get_resp

import run_summaries_encoding as rse
from run_summary_decoding import (
    EMBEDDING_MODELS,
    build_embedding_targets,
    load_encoder,
    retrieval_metrics,
    TRIM_END,
    TRIM_START,
    zscore_embeddings,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("h20_sweep")


def dim_r(true_emb: np.ndarray, pred_emb: np.ndarray) -> float:
    """Mean Pearson r per embedding dimension across time."""
    A = true_emb.astype(np.float64, copy=False)
    B = pred_emb.astype(np.float64, copy=False)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = np.sum(A * B, axis=0)
    den = np.sqrt(np.sum(A * A, axis=0) * np.sum(B * B, axis=0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    r = np.nan_to_num(r)
    return float(r.mean())


def build_brain_and_groups(subject: str, stories, vox, response_root, resp_lengths_by_story):
    """Load and z-score brain responses; return X and per-row group labels (story)."""
    Xs = []
    groups = []
    for s in stories:
        x = get_resp(subject, [s], stack=True, vox=vox, response_root=response_root).astype(np.float32)
        expected = resp_lengths_by_story[s]
        if x.shape[0] != expected:
            raise ValueError(f"{s}: brain TRs {x.shape[0]} != expected {expected}")
        Xs.append(x)
        groups.extend([s] * x.shape[0])
    X = np.vstack(Xs).astype(np.float32)
    # Z-score across time (train stats later for test transform)
    return X, np.array(groups)


def zscore_X_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1
    Xtr = np.nan_to_num((X_train - mu) / sd).astype(np.float32)
    Xte = np.nan_to_num((X_test - mu) / sd).astype(np.float32)
    return Xtr, Xte


def group_kfold_splits(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Create group-based folds over unique stories."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for fi in range(n_splits):
        val_groups = set(folds[fi].tolist())
        val_mask = np.array([g in val_groups for g in groups])
        tr_idx = np.nonzero(~val_mask)[0]
        va_idx = np.nonzero(val_mask)[0]
        yield tr_idx, va_idx


def ridge_fit_predict(Xtr, Ytr, Xte, alpha):
    """Fit multioutput ridge using sklearn; return predictions on Xte."""
    from sklearn.linear_model import Ridge

    # Use iterative solver to avoid ill-conditioned direct solves.
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr", random_state=0)
    model.fit(Xtr, Ytr)
    return model, model.predict(Xte).astype(np.float32)


def ridge_rankk_predict_from_model(model, Xte, rank_k: int):
    """Apply rank-k truncation to a fitted Ridge model's weight matrix."""
    # model.coef_ shape: (D, P) for sklearn Ridge
    W = model.coef_.T.astype(np.float32)  # (P, D)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(rank_k, U.shape[1])
    Wk = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return (Xte @ Wk).astype(np.float32)


def pls_fit_predict(Xtr, Ytr, Xte, n_components: int):
    from sklearn.cross_decomposition import PLSRegression

    n_comp = int(min(n_components, Xtr.shape[1], Xtr.shape[0] - 1, Ytr.shape[1]))
    model = PLSRegression(n_components=n_comp, scale=False)
    model.fit(Xtr, Ytr)
    pred = model.predict(Xte).astype(np.float32)
    return model, pred


def multitask_elasticnet_fit_predict(Xtr, Ytr, Xte, alpha: float, l1_ratio: float):
    from sklearn.linear_model import MultiTaskElasticNet

    model = MultiTaskElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=False,
        max_iter=2000,
        random_state=0,
    )
    model.fit(Xtr, Ytr)
    pred = model.predict(Xte).astype(np.float32)
    return model, pred


# -----------------------------
# PyTorch decoders
# -----------------------------

def _resolve_torch_device(pref: str = "auto"):
    import torch

    if pref == "cuda":
        return torch.device("cuda")
    if pref == "mps":
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_torch_model(arch: str, in_dim: int, out_dim: int, **kwargs):
    """Return an nn.Module for the requested architecture."""
    import torch.nn as nn

    if arch == "linear":
        return nn.Linear(in_dim, out_dim, bias=True)
    if arch == "lowrank":
        k = int(kwargs["rank"])
        return nn.Sequential(
            nn.Linear(in_dim, k, bias=False),
            nn.Linear(k, out_dim, bias=True),
        )
    if arch == "mlp":
        hidden = int(kwargs["hidden"])
        dropout = float(kwargs.get("dropout", 0.2))
        return nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=True),
        )
    raise ValueError(f"Unknown arch: {arch}")


def torch_fit_predict(
    Xtr, Ytr, Xte,
    arch: str,
    device,
    *,
    weight_decay: float = 1e-3,
    lr: float = 1e-3,
    max_epochs: int = 300,
    patience: int = 25,
    batch_size: int = 256,
    val_frac: float = 0.1,
    seed: int = 0,
    **arch_kwargs,
):
    """Train an nn.Module with AdamW + early stopping on an internal val split.

    Returns (trained_model, predictions_on_Xte_as_np).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Ytr = np.asarray(Ytr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    n = Xtr.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr_t = torch.from_numpy(Xtr[tr_idx]).to(device)
    Ytr_t = torch.from_numpy(Ytr[tr_idx]).to(device)
    Xva_t = torch.from_numpy(Xtr[val_idx]).to(device)
    Yva_t = torch.from_numpy(Ytr[val_idx]).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    model = _build_torch_model(arch, Xtr.shape[1], Ytr.shape[1], **arch_kwargs).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    n_train = Xtr_t.shape[0]
    for _ in range(max_epochs):
        model.train()
        idx = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            b = idx[start:start + batch_size]
            optim.zero_grad(set_to_none=True)
            pred = model(Xtr_t[b])
            loss = loss_fn(pred, Ytr_t[b])
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva_t), Yva_t).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = model(Xte_t).detach().cpu().numpy().astype(np.float32)
    return model, pred_te


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="S1")
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    p.add_argument("--holdout-stories", nargs="+", default=None)
    p.add_argument("--holdout-count", type=int, default=rse.DEFAULT_HOLDOUT_COUNT)
    p.add_argument("--no-story-holdout", action="store_true")
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default=str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    p.add_argument("--local-cache-root", default=str(rse.DEFAULT_LOCAL_CACHE_ROOT))
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--summary-model", default="gpt-4o-mini")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--feature-model", default="embedding", choices=list(EMBEDDING_MODELS.keys()))
    p.add_argument(
        "--roi",
        default="full_frontal",
        help="ROI to decode from. Either a per-ROI name (e.g. BA_10, BA_45) or 'full_frontal' "
             "to use all frontal voxels. Default: full_frontal.",
    )
    p.add_argument("--nfolds", type=int, default=5)
    p.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip PyTorch model sweep (linear/low-rank/MLP).",
    )
    p.add_argument(
        "--skip-sklearn",
        action="store_true",
        help="Skip the slow sklearn sweeps (ridge, ridge+rankk, PLS, ElasticNet).",
    )
    p.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation. Fit each config on all training stories and "
             "evaluate directly on the held-out test stories. Every config is reported.",
    )
    p.add_argument(
        "--torch-device",
        default="auto",
        help="Device for PyTorch models: 'auto' (cuda > mps > cpu), 'cuda', 'mps', or 'cpu'.",
    )
    p.add_argument("--output-dir", default="summary_decoding_results")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    mounted_root = None
    if args.local_compute_mode:
        try:
            mounted_root = rse.configure_local_compute_mode(args)
        except FileNotFoundError as e:
            # Allow running purely from the local cache if already present.
            log.warning("%s", e)
            log.warning("Falling back to local cache only (no mounted volume).")

    # Resolve response root early (needed for story-list fallback).
    local_cache_root = Path(args.local_cache_root).expanduser().resolve()
    cached_base = local_cache_root / "data_train"
    fallback_response_root = str(cached_base) if cached_base.exists() else config.DATA_TRAIN_DIR

    try:
        stories = rse.load_story_list(args)
    except FileNotFoundError as e:
        # When running from local cache only, sess_to_story.json may be unavailable.
        log.warning("%s", e)
        subj_dir = Path(fallback_response_root) / "train_response" / args.subject
        if not subj_dir.exists():
            raise
        stories = sorted([p.stem for p in subj_dir.glob("*.hf5")])
        log.warning("Falling back to story list from cached responses (%d stories).", len(stories))

    train_stories, test_stories = rse.split_story_list(stories, args)
    if not test_stories:
        raise ValueError("Need held-out stories for evaluation.")
    log.info("Train stories: %d, test stories: %d", len(train_stories), len(test_stories))

    # Stage response cache
    if args.local_compute_mode and mounted_root is not None:
        response_root = rse.stage_local_response_cache(
            args.subject, stories,
            mounted_data_train_dir=config.DATA_TRAIN_DIR,
            cache_root=local_cache_root,
        )
    else:
        # Use local cache if available, otherwise fall back to config.DATA_TRAIN_DIR.
        response_root = fallback_response_root

    train_resp_lengths, total_voxels = rse.load_resp_info(args.subject, train_stories, data_train_dir=response_root)
    test_resp_lengths, _ = rse.load_resp_info(args.subject, test_stories, data_train_dir=response_root)

    # Load ROI voxels (per-ROI BA or full_frontal)
    uts_id = rse.SUBJECT_TO_UTS.get(args.subject)
    if not uts_id:
        raise ValueError(f"Unknown subject {args.subject}")
    roi_name = args.roi
    if roi_name == "full_frontal":
        roi_json = Path(args.ba_dir) / uts_id / "BA_full_frontal.json"
    else:
        # Allow either "BA_10" or bare "10".
        base = roi_name if roi_name.startswith("BA_") else f"BA_{roi_name}"
        roi_json = Path(args.ba_dir) / uts_id / f"{base}.json"
        roi_name = base
    if not roi_json.is_file():
        raise FileNotFoundError(f"ROI file not found: {roi_json}")
    with open(roi_json, encoding="utf-8") as f:
        roi_data = json.load(f)
    vox = np.sort(np.array(list(roi_data.values())[0], dtype=int))
    vox = vox[vox < total_voxels]
    log.info("Using %s voxels: %d", roi_name, len(vox))

    # Load summaries and embeddings
    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    try:
        summary_index = rse.build_summary_index(summaries_dir)
    except FileNotFoundError as e:
        # Fallback to mounted summaries dir if local outputs aren't present.
        mounted_outputs = Path(args.mounted_project_root) / "generate_summaries" / "outputs"
        if mounted_outputs.is_dir():
            log.warning("%s", e)
            log.warning("Falling back to mounted summaries at %s", mounted_outputs)
            summaries_dir = mounted_outputs
            summary_index = rse.build_summary_index(summaries_dir)
        else:
            raise FileNotFoundError(
                f"{e}\n\n"
                f"Local summaries directory is empty: {summaries_dir}\n"
                f"Mounted summaries directory not found: {mounted_outputs}\n"
                f"Mount the project root and re-run with:\n"
                f"  --local-compute-mode --mounted-project-root /Volumes/<...>/language-decoding-expts\n"
            ) from e
    summary_model = args.summary_model
    horizon = args.horizon

    texts_by_story = {}
    for s in stories:
        path = summary_index[(s, summary_model, horizon)]
        loaded = rse.load_summary_texts(
            path=path, expected_story=s,
            expected_model=summary_model, expected_horizon=horizon,
        )
        texts_by_story[s] = loaded["texts"]

    emb_model_name, emb_dim = EMBEDDING_MODELS[args.feature_model]
    encoder, emb_dim = load_encoder(emb_model_name, device="cpu")

    Y_train = build_embedding_targets(train_stories, texts_by_story, train_resp_lengths, encoder, emb_dim)
    Y_test = build_embedding_targets(test_stories, texts_by_story, test_resp_lengths, encoder, emb_dim)
    Y_train_z, Y_test_z, y_mu, y_sd = zscore_embeddings(Y_train, Y_test)
    del Y_train

    # Load brain X (unscaled; we'll z-score with train stats)
    X_train_raw, g_train = build_brain_and_groups(args.subject, train_stories, vox, response_root, train_resp_lengths)
    X_test_raw, g_test = build_brain_and_groups(args.subject, test_stories, vox, response_root, test_resp_lengths)
    X_train, X_test = zscore_X_train_test(X_train_raw, X_test_raw)
    del X_train_raw, X_test_raw

    # CV folds over training stories (only needed when CV is enabled)
    if args.skip_cv:
        folds = []
        log.info("Skipping CV fold construction (--skip-cv).")
    else:
        folds = list(group_kfold_splits(g_train, n_splits=args.nfolds, seed=args.seed))
        log.info("Prepared %d group folds over training stories", len(folds))

    results = []

    def eval_on_test(model_name, params, pred_test):
        # un-zscore target space for eval
        pred_test_unz = (pred_test * y_sd + y_mu).astype(np.float32)
        dimr = dim_r(Y_test_z, pred_test)  # dim_r in z-scored space is fine (scale-invariant)
        top1, mrr, mean_rank = retrieval_metrics(Y_test, pred_test_unz)
        return {
            "model": model_name,
            **params,
            "dim_r_test": float(dimr),
            "retrieval_top1_test": float(top1),
            "retrieval_mrr_test": float(mrr),
            "retrieval_mean_rank_test": float(mean_rank),
        }

    # ---- sklearn sweeps (ridge / ridge+rankk / PLS / ElasticNet) ----
    if args.skip_sklearn:
        log.info("Skipping sklearn sweeps (--skip-sklearn).")
    else:
        # ---- Ridge sweep (alpha) ----
        ridge_alphas = np.logspace(0, 8, 25)
        if args.skip_cv:
            log.info("Ridge: --skip-cv, fitting each alpha on full train and testing on test.")
            best = None
            for ai, alpha in enumerate(ridge_alphas):
                log.info("  Ridge alpha %d/%d = %.3g", ai + 1, len(ridge_alphas), float(alpha))
                _, pred_test = ridge_fit_predict(X_train, Y_train_z, X_test, alpha)
                results.append(eval_on_test("ridge", {"alpha": float(alpha)}, pred_test))
                if best is None or results[-1]["dim_r_test"] > best["dim_r_test"]:
                    best = {"alpha": float(alpha), "dim_r_test": results[-1]["dim_r_test"]}
            log.info("Best ridge on test: alpha=%.3g dim_r=%.4f", best["alpha"], best["dim_r_test"])
        else:
            best = None
            for ai, alpha in enumerate(ridge_alphas):
                log.info("Ridge CV: alpha %d/%d = %.3g", ai + 1, len(ridge_alphas), float(alpha))
                cv_scores = []
                for fi, (tr_idx, va_idx) in enumerate(folds):
                    log.info("  fold %d/%d (train=%d, val=%d)", fi + 1, len(folds), len(tr_idx), len(va_idx))
                    _, pred_va = ridge_fit_predict(X_train[tr_idx], Y_train_z[tr_idx], X_train[va_idx], alpha)
                    cv_scores.append(dim_r(Y_train_z[va_idx], pred_va))
                mean_cv = float(np.mean(cv_scores))
                log.info("  alpha=%.3g mean_cv_dim_r=%.4f", float(alpha), mean_cv)
                if best is None or mean_cv > best["cv"]:
                    best = {"alpha": float(alpha), "cv": mean_cv}
            log.info("Best ridge alpha by train-story CV: %.3g (cv dim_r=%.4f)", best["alpha"], best["cv"])
            ridge_model, pred_test = ridge_fit_predict(X_train, Y_train_z, X_test, best["alpha"])
            results.append(eval_on_test("ridge", {"alpha": best["alpha"], "cv_dim_r": best["cv"]}, pred_test))

        # ---- Ridge + rank-k truncation ----
        ranks = [8, 16, 32, 64, 128]
        if args.skip_cv:
            log.info("Ridge+rankk: --skip-cv, using best ridge alpha=%.3g and testing each rank.", best["alpha"])
            ridge_model_full, _ = ridge_fit_predict(X_train, Y_train_z, X_test, best["alpha"])
            for k in ranks:
                pred_test_rr = ridge_rankk_predict_from_model(ridge_model_full, X_test, k)
                results.append(
                    eval_on_test(
                        "ridge_rankk",
                        {"alpha": best["alpha"], "rank": int(k)},
                        pred_test_rr,
                    )
                )
        else:
            best_rr = None
            for k in ranks:
                cv_scores = []
                for tr_idx, va_idx in folds:
                    mdl, _ = ridge_fit_predict(X_train[tr_idx], Y_train_z[tr_idx], X_train[va_idx], best["alpha"])
                    pred_va = ridge_rankk_predict_from_model(mdl, X_train[va_idx], k)
                    cv_scores.append(dim_r(Y_train_z[va_idx], pred_va))
                mean_cv = float(np.mean(cv_scores))
                if best_rr is None or mean_cv > best_rr["cv"]:
                    best_rr = {"rank": int(k), "cv": mean_cv}
            log.info("Best ridge+rankk by train-story CV: k=%d (cv dim_r=%.4f)", best_rr["rank"], best_rr["cv"])
            ridge_model_full, _ = ridge_fit_predict(X_train, Y_train_z, X_test, best["alpha"])
            pred_test_rr = ridge_rankk_predict_from_model(ridge_model_full, X_test, best_rr["rank"])
            results.append(
                eval_on_test(
                    "ridge_rankk",
                    {"alpha": best["alpha"], "rank": best_rr["rank"], "cv_dim_r": best_rr["cv"]},
                    pred_test_rr,
                )
            )

        # ---- PLS sweep ----
        pls_comps = [8, 16, 32, 64, 128]
        if args.skip_cv:
            log.info("PLS: --skip-cv, testing each n_components on held-out test.")
            for nc in pls_comps:
                _, pred_test_pls = pls_fit_predict(X_train, Y_train_z, X_test, nc)
                results.append(eval_on_test("pls", {"n_components": int(nc)}, pred_test_pls))
        else:
            best_pls = None
            for nc in pls_comps:
                cv_scores = []
                for tr_idx, va_idx in folds:
                    _, pred_va = pls_fit_predict(X_train[tr_idx], Y_train_z[tr_idx], X_train[va_idx], nc)
                    cv_scores.append(dim_r(Y_train_z[va_idx], pred_va))
                mean_cv = float(np.mean(cv_scores))
                if best_pls is None or mean_cv > best_pls["cv"]:
                    best_pls = {"n_components": int(nc), "cv": mean_cv}
            log.info("Best PLS by train-story CV: n=%d (cv dim_r=%.4f)", best_pls["n_components"], best_pls["cv"])
            _, pred_test_pls = pls_fit_predict(X_train, Y_train_z, X_test, best_pls["n_components"])
            results.append(eval_on_test("pls", {"n_components": best_pls["n_components"], "cv_dim_r": best_pls["cv"]}, pred_test_pls))

        # ---- MultiTaskElasticNet sweep (small grid) ----
        en_alphas = [1e-4, 1e-3, 1e-2, 1e-1]
        en_l1 = [0.05, 0.1, 0.2, 0.5]
        if args.skip_cv:
            log.info("ElasticNet: --skip-cv, testing each (alpha, l1_ratio) on held-out test.")
            for a in en_alphas:
                for l1 in en_l1:
                    try:
                        _, pred_test_en = multitask_elasticnet_fit_predict(
                            X_train, Y_train_z, X_test, a, l1,
                        )
                    except Exception as e:
                        log.warning("  ElasticNet (a=%.1e, l1=%.2f) failed: %s", a, l1, e)
                        continue
                    results.append(
                        eval_on_test(
                            "elasticnet",
                            {"alpha": float(a), "l1_ratio": float(l1)},
                            pred_test_en,
                        )
                    )
        else:
            best_en = None
            for a in en_alphas:
                for l1 in en_l1:
                    cv_scores = []
                    for tr_idx, va_idx in folds:
                        try:
                            _, pred_va = multitask_elasticnet_fit_predict(X_train[tr_idx], Y_train_z[tr_idx], X_train[va_idx], a, l1)
                        except Exception:
                            cv_scores = None
                            break
                        cv_scores.append(dim_r(Y_train_z[va_idx], pred_va))
                    if not cv_scores:
                        continue
                    mean_cv = float(np.mean(cv_scores))
                    if best_en is None or mean_cv > best_en["cv"]:
                        best_en = {"alpha": float(a), "l1_ratio": float(l1), "cv": mean_cv}
            if best_en is not None:
                log.info("Best ElasticNet by train-story CV: a=%.1e l1=%.2f (cv dim_r=%.4f)", best_en["alpha"], best_en["l1_ratio"], best_en["cv"])
                _, pred_test_en = multitask_elasticnet_fit_predict(X_train, Y_train_z, X_test, best_en["alpha"], best_en["l1_ratio"])
                results.append(
                    eval_on_test(
                        "elasticnet",
                        {"alpha": best_en["alpha"], "l1_ratio": best_en["l1_ratio"], "cv_dim_r": best_en["cv"]},
                        pred_test_en,
                    )
                )
            else:
                log.warning("ElasticNet sweep failed for all settings.")

    # ---- PyTorch sweep (linear / low-rank / MLP) ----
    if not args.skip_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("PyTorch not available; skipping torch sweep.")
        else:
            device = _resolve_torch_device(args.torch_device)
            log.info("Torch device: %s", device)

            # Configs chosen with limited training data in mind: small bottlenecks,
            # strong weight decay, early stopping. Each config runs CV across folds
            # for hyperparameter selection (on train-story dim_r), then a final fit
            # on all train stories for test eval.
            torch_configs = []
            # Linear baseline (weight-decay sweep)
            for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
                torch_configs.append({"arch": "linear", "weight_decay": wd})
            # Low-rank linear
            for k in [8, 16, 32, 64]:
                for wd in [1e-3, 1e-2]:
                    torch_configs.append({"arch": "lowrank", "rank": k, "weight_decay": wd})
            # MLP (small, regularized)
            for hidden in [32, 64, 128]:
                for dropout in [0.1, 0.3]:
                    for wd in [1e-3, 1e-2]:
                        torch_configs.append(
                            {"arch": "mlp", "hidden": hidden, "dropout": dropout, "weight_decay": wd}
                        )

            if args.skip_cv:
                log.info("Torch: --skip-cv, fitting each config on full train and testing on test.")
                for ci, cfg in enumerate(torch_configs):
                    log.info("Torch %d/%d: %s", ci + 1, len(torch_configs), cfg)
                    try:
                        _, pred_te = torch_fit_predict(
                            X_train, Y_train_z, X_test,
                            device=device, seed=args.seed, **cfg,
                        )
                    except Exception as e:  # pragma: no cover
                        log.warning("  torch config failed (%s): %s", cfg, e)
                        continue
                    arch = cfg["arch"]
                    params = {k: v for k, v in cfg.items() if k != "arch"}
                    results.append(eval_on_test(f"torch_{arch}", params, pred_te))
            else:
                def _run_torch_cv(cfg):
                    cv_scores = []
                    for fi, (tr_idx, va_idx) in enumerate(folds):
                        _, pred_va = torch_fit_predict(
                            X_train[tr_idx], Y_train_z[tr_idx], X_train[va_idx],
                            device=device, seed=args.seed, **cfg,
                        )
                        cv_scores.append(dim_r(Y_train_z[va_idx], pred_va))
                    return float(np.mean(cv_scores))

                best_by_arch: dict = {}
                for ci, cfg in enumerate(torch_configs):
                    log.info("Torch CV %d/%d: %s", ci + 1, len(torch_configs), cfg)
                    try:
                        cv = _run_torch_cv(cfg)
                    except Exception as e:  # pragma: no cover
                        log.warning("  torch config failed (%s): %s", cfg, e)
                        continue
                    log.info("  cv dim_r=%.4f", cv)
                    arch = cfg["arch"]
                    if arch not in best_by_arch or cv > best_by_arch[arch]["cv"]:
                        best_by_arch[arch] = {"cfg": cfg, "cv": cv}

                for arch, best_cfg in best_by_arch.items():
                    cfg = best_cfg["cfg"]
                    log.info("Best torch-%s cv dim_r=%.4f (%s)", arch, best_cfg["cv"], cfg)
                    try:
                        _, pred_te = torch_fit_predict(
                            X_train, Y_train_z, X_test,
                            device=device, seed=args.seed, **cfg,
                        )
                    except Exception as e:  # pragma: no cover
                        log.warning("  torch final fit failed: %s", e)
                        continue
                    params = {k: v for k, v in cfg.items() if k != "arch"}
                    params["cv_dim_r"] = float(best_cfg["cv"])
                    results.append(eval_on_test(f"torch_{arch}", params, pred_te))

    # Sort and save
    results_sorted = sorted(results, key=lambda r: r["dim_r_test"], reverse=True)
    out_dir = Path(args.output_dir) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"h20_sweep__{roi_name}__{args.feature_model}__{summary_model}__holdout-5.csv"
    # Union all keys so rows with extra arch-specific params (rank, hidden, ...) are preserved.
    all_keys: list = []
    seen = set()
    for r in results_sorted:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results_sorted)

    log.info("Saved sweep results to %s", out_path)
    log.info("Top models:")
    for r in results_sorted[:5]:
        log.info(
            "  %-12s dim_r=%.4f top1=%.3f mrr=%.3f (params=%s)",
            r["model"],
            r["dim_r_test"],
            r["retrieval_top1_test"],
            r["retrieval_mrr_test"],
            {k: v for k, v in r.items() if k not in {"model", "dim_r_test", "retrieval_top1_test", "retrieval_mrr_test", "retrieval_mean_rank_test"}},
        )


if __name__ == "__main__":
    main()

