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


def story_retrieval_metrics(true_emb: np.ndarray, pred_emb: np.ndarray, groups: np.ndarray):
    """Story-level retrieval.

    Two complementary scores (chance = 1/n_stories for both top1 variants):

    - argmax_in_correct:   for each test TR i, take argmax over j of
                           cos(pred[i], true[j]). Return fraction where
                           groups[argmax] == groups[i]. (Easy case: nearest
                           TR tends to be within-story due to temporal
                           smoothness, so this is a liberal metric.)
    - mean_sim_top1 / mrr: for each test TR i, compute mean cos(pred[i], true[j])
                           restricted to each story c (averaging over j in c),
                           rank stories by that score, and return top1 / MRR
                           over stories.
    """
    t = true_emb.astype(np.float32, copy=False)
    p = pred_emb.astype(np.float32, copy=False)
    t_n = t / np.clip(np.linalg.norm(t, axis=1, keepdims=True), 1e-12, None)
    p_n = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)

    sim = p_n @ t_n.T  # (T, T)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n_stories = len(uniq)

    argmax_j = np.argmax(sim, axis=1)
    argmax_in_correct = float((groups[argmax_j] == groups).mean())

    story_cols = {s: np.nonzero(groups == s)[0] for s in uniq}
    mean_sim_per_story = np.stack(
        [sim[:, cols].mean(axis=1) for s, cols in story_cols.items()],
        axis=1,
    )  # (T, n_stories)
    ordered = list(story_cols.keys())
    correct_col = np.array([ordered.index(g) for g in groups])
    correct_score = mean_sim_per_story[np.arange(sim.shape[0]), correct_col]
    ranks = 1 + (mean_sim_per_story > correct_score[:, None]).sum(axis=1)
    mean_sim_top1 = float((ranks == 1).mean())
    mean_sim_mrr = float((1.0 / ranks).mean())
    mean_sim_mean_rank = float(ranks.mean())

    return {
        "n_stories": int(n_stories),
        "argmax_in_correct_story": argmax_in_correct,
        "story_top1": mean_sim_top1,
        "story_mrr": mean_sim_mrr,
        "story_mean_rank": mean_sim_mean_rank,
    }


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
    return X, np.array(groups)


def build_features_and_groups(features_dir: Path, subject: str, stories, resp_lengths_by_story):
    """Load pre-extracted features from brain_encoder_pretrain/extract_features.py.

    Each story is read from <features_dir>/<subject>/<story>.npz under key 'X'.
    Returns the concatenated feature array [T_total, d_feat] and a per-row group
    label array. TR counts are validated against resp_lengths_by_story.
    """
    subj_dir = Path(features_dir) / subject
    if not subj_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {subj_dir}")
    Xs = []
    groups = []
    d_feat = None
    for s in stories:
        path = subj_dir / f"{s}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")
        with np.load(path) as npz:
            if "X" not in npz.files:
                raise ValueError(f"{path}: expected key 'X' in .npz, found {npz.files}")
            x = npz["X"].astype(np.float32)
        expected = resp_lengths_by_story[s]
        if x.shape[0] != expected:
            raise ValueError(
                f"{s}: feature TRs {x.shape[0]} != expected {expected}. Did the checkpoint "
                f"use the same story / z-score as the response files?"
            )
        if d_feat is None:
            d_feat = int(x.shape[1])
        elif int(x.shape[1]) != d_feat:
            raise ValueError(f"{s}: feature dim {x.shape[1]} != expected {d_feat}")
        Xs.append(x)
        groups.extend([s] * x.shape[0])
    X = np.vstack(Xs).astype(np.float32)
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


def _loss_from_kind(kind: str, temperature: float):
    """Return a callable loss(pred, target) for the named kind."""
    import torch
    import torch.nn.functional as F

    if kind == "mse":
        return lambda pred, target: F.mse_loss(pred, target)
    if kind == "cosine":
        def _cos(pred, target):
            p = F.normalize(pred, dim=-1)
            t = F.normalize(target, dim=-1)
            return (1.0 - (p * t).sum(dim=-1)).mean()
        return _cos
    if kind == "infonce":
        def _nce(pred, target):
            p = F.normalize(pred, dim=-1)
            t = F.normalize(target, dim=-1)
            logits = (p @ t.T) / float(temperature)
            labels = torch.arange(logits.shape[0], device=logits.device)
            # Symmetric InfoNCE (pred->target and target->pred)
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return _nce
    raise ValueError(f"Unknown loss: {kind}")


def _group_val_split(n: int, groups, val_frac: float, seed: int):
    """Return (tr_idx, val_idx). If groups is not None, hold out whole groups for val."""
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_val_g = max(1, int(round(val_frac * len(uniq))))
        val_g = set(uniq[:n_val_g].tolist())
        val_mask = np.array([g in val_g for g in groups])
        val_idx = np.nonzero(val_mask)[0]
        tr_idx = np.nonzero(~val_mask)[0]
        if len(tr_idx) == 0:
            # Fallback: random TR split
            perm = rng.permutation(n)
            n_val = max(1, int(round(val_frac * n)))
            return perm[n_val:], perm[:n_val]
        return tr_idx, val_idx
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    return perm[n_val:], perm[:n_val]


def torch_fit_predict(
    Xtr, Ytr, Xte,
    arch: str,
    device,
    *,
    groups=None,
    loss: str = "infonce",
    temperature: float = 0.07,
    weight_decay: float = 1e-3,
    lr: float = 1e-3,
    max_epochs: int = 1000,
    patience: int = 50,
    batch_size: int = 256,
    val_frac: float = 0.1,
    seed: int = 0,
    **arch_kwargs,
):
    """Train an nn.Module with AdamW + early stopping on an internal val split.

    If `groups` is provided (length == Xtr.shape[0]), the early-stopping val set
    holds out whole groups (stories), not random TRs, to avoid leakage from
    temporal autocorrelation.

    Returns (trained_model, predictions_on_Xte_as_np).
    """
    import torch

    torch.manual_seed(seed)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Ytr = np.asarray(Ytr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    tr_idx, val_idx = _group_val_split(Xtr.shape[0], groups, val_frac, seed)

    Xtr_t = torch.from_numpy(Xtr[tr_idx]).to(device)
    Ytr_t = torch.from_numpy(Ytr[tr_idx]).to(device)
    Xva_t = torch.from_numpy(Xtr[val_idx]).to(device)
    Yva_t = torch.from_numpy(Ytr[val_idx]).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    model = _build_torch_model(arch, Xtr.shape[1], Ytr.shape[1], **arch_kwargs).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = _loss_from_kind(loss, temperature)

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
            loss_val = loss_fn(pred, Ytr_t[b])
            loss_val.backward()
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


def _unit_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def diagnose_targets(
    Y_test: np.ndarray,
    g_test: np.ndarray,
    *,
    log_fn=print,
    retrieval_fn=None,
    lags=(1, 2, 5, 10, 20, 50),
    windows=(1, 3, 10, 20),
    noise_scales=(0.0, 0.1, 0.3, 1.0, 3.0),
    shift_ks=(1, 2, 5, 10),
    seed: int = 0,
):
    """Print diagnostics about target structure and retrieval ceilings.

    Computes:
    1. Target autocorrelation: mean cos(Y[t], Y[t+lag]) for each lag, within-story.
    2. Nearest-other-TR distance distribution: for each TR, how far (in TRs) is
       the most-similar OTHER TR within the same story? (Tells us: when an
       oracle picks a non-self TR, how close is it?)
    3. Within-window retrieval using oracle pred=Y_test: fraction of TRs where
       the argmax-over-non-self lies within ±w TRs.
    4. Noise-sensitivity: pred = Y + sigma * Gaussian; top1/mrr as sigma varies.
    5. Shifted-oracle retrieval: pred[i] = Y[i+k]; top1/mrr as k varies.
    """
    if retrieval_fn is None:
        retrieval_fn = retrieval_metrics

    rng = np.random.default_rng(seed)
    Y = Y_test.astype(np.float32, copy=False)
    Y_unit = _unit_normalize(Y)
    groups = np.asarray(g_test)
    T = Y.shape[0]

    # --- 1. Autocorrelation within story ---
    log_fn("  [1] Target autocorrelation (mean cos_sim within-story):")
    log_fn("       lag      mean_cos    n_pairs")
    per_lag_rows = []
    for lag in lags:
        sims = []
        for s in np.unique(groups):
            idx = np.nonzero(groups == s)[0]
            if len(idx) <= lag:
                continue
            a = Y_unit[idx[:-lag]]
            b = Y_unit[idx[lag:]]
            sims.append((a * b).sum(axis=1))
        if not sims:
            continue
        flat = np.concatenate(sims)
        per_lag_rows.append((lag, float(flat.mean()), int(flat.size)))
        log_fn(f"       {lag:>3d}       {flat.mean():+.4f}     {flat.size}")

    # --- 2. Nearest-other-TR (within story) distance distribution ---
    log_fn("  [2] Distance (|Δt|) to most-similar OTHER TR, within same story:")
    lags_found = []
    for s in np.unique(groups):
        idx = np.nonzero(groups == s)[0]
        if len(idx) < 3:
            continue
        Ys = Y_unit[idx]
        sim = Ys @ Ys.T
        np.fill_diagonal(sim, -np.inf)
        nn = np.argmax(sim, axis=1)
        lags_found.append(np.abs(nn - np.arange(len(idx))))
    if lags_found:
        flat = np.concatenate(lags_found)
        log_fn(
            f"       median={np.median(flat):.1f}  p75={np.percentile(flat,75):.1f}  "
            f"p90={np.percentile(flat,90):.1f}  p95={np.percentile(flat,95):.1f}  "
            f"max={flat.max()}"
        )
        for w in windows:
            frac = float((flat <= w).mean())
            log_fn(f"       frac(|Δt| ≤ {w:>2d}): {frac:.3f}")

    # --- 3. Noise-sensitivity of retrieval when predictions ARE the true targets ---
    log_fn("  [3] Retrieval under additive noise on oracle predictions:")
    log_fn("       sigma        top1     mrr      mean_rank")
    Y_std_per_dim = float(Y.std(axis=0).mean())
    for sigma_mult in noise_scales:
        noise = rng.standard_normal(Y.shape).astype(np.float32) * (sigma_mult * Y_std_per_dim)
        pred = Y + noise
        top1, mrr, mean_rank = retrieval_fn(Y, pred)
        log_fn(f"       {sigma_mult:>4.1f}        {top1:.3f}    {mrr:.3f}    {mean_rank:.1f}")

    # --- 4. Shifted-oracle retrieval ---
    log_fn("  [4] Shifted oracle (pred[i] = Y[i+k]) retrieval on full test set:")
    log_fn("       k            top1     mrr      mean_rank")
    for k in shift_ks:
        if k <= 0 or k >= T:
            continue
        pred = np.roll(Y, -k, axis=0)
        top1, mrr, mean_rank = retrieval_fn(Y, pred)
        log_fn(f"       {k:>3d}          {top1:.3f}    {mrr:.3f}    {mean_rank:.1f}")


def select_voxels_by_encoding(
    X_train: np.ndarray,
    Y_train_z: np.ndarray,
    g_train: np.ndarray,
    n_select: int,
    alpha: float = 10.0,
    seed: int = 0,
):
    """Rank voxels by 2-fold (story-grouped) CV r of an encoding ridge (Y -> X_v).

    Returns (sorted_voxel_indices, per_voxel_r). Sorted by decreasing r.
    """
    def _fit_ridge_closed(Y, X, alpha):
        # Y: (T, D), X: (T, V) -> W: (D, V) = (YtY + aI)^-1 Yt X
        D = Y.shape[1]
        YtY = (Y.T @ Y).astype(np.float64)
        YtY[np.diag_indices_from(YtY)] += float(alpha)
        W = np.linalg.solve(YtY, (Y.T @ X).astype(np.float64))
        return W.astype(np.float32)

    uniq = np.unique(g_train)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    half = max(1, len(uniq) // 2)
    g_a = set(uniq[:half].tolist())
    mask_a = np.array([g in g_a for g in g_train])
    idx_a = np.nonzero(mask_a)[0]
    idx_b = np.nonzero(~mask_a)[0]

    log.info("Voxel selection: encoding ridge on %d train stories (alpha=%.2g)", len(uniq), alpha)
    W_ab = _fit_ridge_closed(Y_train_z[idx_a], X_train[idx_a], alpha)
    pred_b = (Y_train_z[idx_b] @ W_ab).astype(np.float32)
    W_ba = _fit_ridge_closed(Y_train_z[idx_b], X_train[idx_b], alpha)
    pred_a = (Y_train_z[idx_a] @ W_ba).astype(np.float32)

    pred_full = np.empty_like(X_train, dtype=np.float32)
    pred_full[idx_a] = pred_a
    pred_full[idx_b] = pred_b

    A = X_train - X_train.mean(axis=0, keepdims=True)
    B = pred_full - pred_full.mean(axis=0, keepdims=True)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0))
    r = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den != 0)
    r = np.nan_to_num(r)

    n_keep = min(int(n_select), r.shape[0])
    top_idx = np.argsort(-r)[:n_keep]
    log.info(
        "Voxel selection: kept %d/%d voxels; top r=%.3f, median r=%.3f",
        n_keep, r.shape[0], float(r[top_idx[0]]) if n_keep else float("nan"),
        float(np.median(r[top_idx])) if n_keep else float("nan"),
    )
    return np.sort(top_idx), r


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
        help="Voxel set: 'all' (every voxel the subject has), 'full_frontal' (all frontal), "
             "or a single BA ROI name (e.g. BA_10, BA_45, or just '10'). Default: full_frontal.",
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
    p.add_argument(
        "--target-pca",
        type=int,
        default=0,
        help="If >0, fit PCA on Y_train and train models to predict K components. "
             "Predictions are inverted before evaluation. Default: 0 (no PCA).",
    )
    p.add_argument(
        "--loss",
        default="infonce",
        choices=["mse", "cosine", "infonce"],
        help="Loss for PyTorch models. 'infonce' is in-batch contrastive (recommended for "
             "retrieval metrics). 'cosine' is 1-cos_sim. 'mse' is standard regression.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for InfoNCE loss.",
    )
    p.add_argument(
        "--select-voxels",
        type=int,
        default=0,
        help="If >0, pre-select top N voxels by encoding-model CV r on training stories, "
             "before decoding. Default: 0 (use all ROI voxels).",
    )
    p.add_argument(
        "--brain-pca",
        type=int,
        default=0,
        help="If >0, fit PCA on X_train and project both X_train and X_test to K components "
             "before decoding. Applied after --select-voxels if both are set. Default: 0 (no PCA).",
    )
    p.add_argument(
        "--features-dir",
        default=None,
        help="If set, use pre-extracted per-TR features from this directory instead of raw "
             "voxel responses. Expects <features-dir>/<subject>/<story>.npz with key 'X'. "
             "Produced by brain_encoder_pretrain/extract_features.py. When set, --roi and "
             "--select-voxels are ignored (features are already a fixed-dim representation).",
    )
    p.add_argument(
        "--features-tag",
        default=None,
        help="Short tag added to output filename when --features-dir is set (defaults to "
             "the features-dir's last path component).",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="Before any model training, print target-structure diagnostics "
             "(autocorrelation, nearest-neighbor distances, noise-sensitivity of "
             "retrieval, shifted-oracle retrieval).",
    )
    p.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run diagnostics and exit without training any model. Implies --diagnose.",
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

    use_features = args.features_dir is not None
    if use_features:
        # Pre-extracted features from brain_encoder_pretrain; --roi / --select-voxels don't apply.
        if args.select_voxels > 0:
            log.warning("--select-voxels is ignored when --features-dir is set.")
        roi_name = "features"
        vox = None
    else:
        # Load voxels: all, full_frontal, or a single BA ROI
        uts_id = rse.SUBJECT_TO_UTS.get(args.subject)
        if not uts_id:
            raise ValueError(f"Unknown subject {args.subject}")
        roi_name = args.roi
        if roi_name in ("all", "whole", "all_voxels"):
            roi_name = "all"
            vox = np.arange(total_voxels, dtype=int)
        else:
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
    if use_features:
        features_dir = Path(args.features_dir).expanduser().resolve()
        log.info("Loading pre-extracted features from %s", features_dir)
        X_train_raw, g_train = build_features_and_groups(
            features_dir, args.subject, train_stories, train_resp_lengths,
        )
        X_test_raw, g_test = build_features_and_groups(
            features_dir, args.subject, test_stories, test_resp_lengths,
        )
        log.info("Feature dim: %d", X_train_raw.shape[1])
    else:
        X_train_raw, g_train = build_brain_and_groups(args.subject, train_stories, vox, response_root, train_resp_lengths)
        X_test_raw, g_test = build_brain_and_groups(args.subject, test_stories, vox, response_root, test_resp_lengths)
    X_train, X_test = zscore_X_train_test(X_train_raw, X_test_raw)
    del X_train_raw, X_test_raw

    # Target-structure diagnostics (no model training required)
    if args.diagnose or args.diagnose_only:
        log.info("=== Target diagnostics (horizon=%d, %d test TRs, %d stories) ===",
                 args.horizon, Y_test.shape[0], len(np.unique(g_test)))
        diagnose_targets(
            Y_test, g_test,
            log_fn=log.info,
            retrieval_fn=retrieval_metrics,
            seed=args.seed,
        )
        log.info("=== end diagnostics ===")
        if args.diagnose_only:
            log.info("--diagnose-only set; exiting without training.")
            return

    # Optional voxel selection using a cheap encoding model
    if args.select_voxels and args.select_voxels > 0:
        sel_idx, _ = select_voxels_by_encoding(
            X_train, Y_train_z, g_train, n_select=args.select_voxels, seed=args.seed,
        )
        X_train = X_train[:, sel_idx]
        X_test = X_test[:, sel_idx]
        log.info("After voxel selection: X_train shape %s", X_train.shape)

    # Optional input-side PCA (applied after voxel selection if both are set)
    if args.brain_pca and args.brain_pca > 0:
        from sklearn.decomposition import PCA as _PCA

        K_x = int(min(args.brain_pca, X_train.shape[0] - 1, X_train.shape[1]))
        log.info("Brain PCA: %d components (of %d voxels)", K_x, X_train.shape[1])
        x_pca = _PCA(n_components=K_x, random_state=args.seed)
        X_train = x_pca.fit_transform(X_train).astype(np.float32)
        X_test = x_pca.transform(X_test).astype(np.float32)
        explained_x = float(np.sum(x_pca.explained_variance_ratio_))
        log.info("  brain PCA explained variance: %.3f", explained_x)

    # Optional target-side PCA (model learns to predict PCs; we invert for eval)
    if args.target_pca and args.target_pca > 0:
        from sklearn.decomposition import PCA

        K = int(min(args.target_pca, Y_train_z.shape[0], Y_train_z.shape[1]))
        log.info("Target PCA: %d components (of %d)", K, Y_train_z.shape[1])
        y_pca = PCA(n_components=K, random_state=args.seed)
        Y_train_model = y_pca.fit_transform(Y_train_z).astype(np.float32)
        Y_test_model = y_pca.transform(Y_test_z).astype(np.float32)
        explained = float(np.sum(y_pca.explained_variance_ratio_))
        log.info("  explained variance: %.3f", explained)

        def _invert_target(pred_model_space: np.ndarray) -> np.ndarray:
            return y_pca.inverse_transform(pred_model_space).astype(np.float32)
    else:
        Y_train_model = Y_train_z
        Y_test_model = Y_test_z

        def _invert_target(pred_model_space: np.ndarray) -> np.ndarray:
            return pred_model_space

    # CV folds over training stories (only needed when CV is enabled)
    if args.skip_cv:
        folds = []
        log.info("Skipping CV fold construction (--skip-cv).")
    else:
        folds = list(group_kfold_splits(g_train, n_splits=args.nfolds, seed=args.seed))
        log.info("Prepared %d group folds over training stories", len(folds))

    results = []

    def eval_on_test(model_name, params, pred_test):
        # Invert PCA (if enabled) back to full-dim z-scored embedding space, then
        # un-zscore for retrieval metrics against the original Y_test.
        pred_test_z = _invert_target(pred_test)
        pred_test_unz = (pred_test_z * y_sd + y_mu).astype(np.float32)
        dimr = dim_r(Y_test_z, pred_test_z)
        top1, mrr, mean_rank = retrieval_metrics(Y_test, pred_test_unz)
        story = story_retrieval_metrics(Y_test, pred_test_unz, g_test)
        return {
            "model": model_name,
            **params,
            "dim_r_test": float(dimr),
            "retrieval_top1_test": float(top1),
            "retrieval_mrr_test": float(mrr),
            "retrieval_mean_rank_test": float(mean_rank),
            "story_argmax_in_correct": float(story["argmax_in_correct_story"]),
            "story_top1": float(story["story_top1"]),
            "story_mrr": float(story["story_mrr"]),
            "story_mean_rank": float(story["story_mean_rank"]),
            "n_test_stories": int(story["n_stories"]),
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
                _, pred_test = ridge_fit_predict(X_train, Y_train_model, X_test, alpha)
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
                    _, pred_va = ridge_fit_predict(X_train[tr_idx], Y_train_model[tr_idx], X_train[va_idx], alpha)
                    cv_scores.append(dim_r(Y_train_model[va_idx], pred_va))
                mean_cv = float(np.mean(cv_scores))
                log.info("  alpha=%.3g mean_cv_dim_r=%.4f", float(alpha), mean_cv)
                if best is None or mean_cv > best["cv"]:
                    best = {"alpha": float(alpha), "cv": mean_cv}
            log.info("Best ridge alpha by train-story CV: %.3g (cv dim_r=%.4f)", best["alpha"], best["cv"])
            ridge_model, pred_test = ridge_fit_predict(X_train, Y_train_model, X_test, best["alpha"])
            results.append(eval_on_test("ridge", {"alpha": best["alpha"], "cv_dim_r": best["cv"]}, pred_test))

        # ---- Ridge + rank-k truncation ----
        ranks = [8, 16, 32, 64, 128]
        if args.skip_cv:
            log.info("Ridge+rankk: --skip-cv, using best ridge alpha=%.3g and testing each rank.", best["alpha"])
            ridge_model_full, _ = ridge_fit_predict(X_train, Y_train_model, X_test, best["alpha"])
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
                    mdl, _ = ridge_fit_predict(X_train[tr_idx], Y_train_model[tr_idx], X_train[va_idx], best["alpha"])
                    pred_va = ridge_rankk_predict_from_model(mdl, X_train[va_idx], k)
                    cv_scores.append(dim_r(Y_train_model[va_idx], pred_va))
                mean_cv = float(np.mean(cv_scores))
                if best_rr is None or mean_cv > best_rr["cv"]:
                    best_rr = {"rank": int(k), "cv": mean_cv}
            log.info("Best ridge+rankk by train-story CV: k=%d (cv dim_r=%.4f)", best_rr["rank"], best_rr["cv"])
            ridge_model_full, _ = ridge_fit_predict(X_train, Y_train_model, X_test, best["alpha"])
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
                _, pred_test_pls = pls_fit_predict(X_train, Y_train_model, X_test, nc)
                results.append(eval_on_test("pls", {"n_components": int(nc)}, pred_test_pls))
        else:
            best_pls = None
            for nc in pls_comps:
                cv_scores = []
                for tr_idx, va_idx in folds:
                    _, pred_va = pls_fit_predict(X_train[tr_idx], Y_train_model[tr_idx], X_train[va_idx], nc)
                    cv_scores.append(dim_r(Y_train_model[va_idx], pred_va))
                mean_cv = float(np.mean(cv_scores))
                if best_pls is None or mean_cv > best_pls["cv"]:
                    best_pls = {"n_components": int(nc), "cv": mean_cv}
            log.info("Best PLS by train-story CV: n=%d (cv dim_r=%.4f)", best_pls["n_components"], best_pls["cv"])
            _, pred_test_pls = pls_fit_predict(X_train, Y_train_model, X_test, best_pls["n_components"])
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
                            X_train, Y_train_model, X_test, a, l1,
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
                            _, pred_va = multitask_elasticnet_fit_predict(X_train[tr_idx], Y_train_model[tr_idx], X_train[va_idx], a, l1)
                        except Exception:
                            cv_scores = None
                            break
                        cv_scores.append(dim_r(Y_train_model[va_idx], pred_va))
                    if not cv_scores:
                        continue
                    mean_cv = float(np.mean(cv_scores))
                    if best_en is None or mean_cv > best_en["cv"]:
                        best_en = {"alpha": float(a), "l1_ratio": float(l1), "cv": mean_cv}
            if best_en is not None:
                log.info("Best ElasticNet by train-story CV: a=%.1e l1=%.2f (cv dim_r=%.4f)", best_en["alpha"], best_en["l1_ratio"], best_en["cv"])
                _, pred_test_en = multitask_elasticnet_fit_predict(X_train, Y_train_model, X_test, best_en["alpha"], best_en["l1_ratio"])
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

            # Small set of configs. With limited data, prefer low-rank mappings and
            # small MLPs with strong regularization. Early stopping uses a
            # story-grouped val split within the training data.
            torch_configs = []
            for wd in [1e-3, 1e-2, 1e-1]:
                torch_configs.append({"arch": "linear", "weight_decay": wd})
            for k in [16, 32, 64]:
                for wd in [1e-3, 1e-2]:
                    torch_configs.append({"arch": "lowrank", "rank": k, "weight_decay": wd})
            for hidden in [64, 128]:
                for wd in [1e-3, 1e-2]:
                    torch_configs.append(
                        {"arch": "mlp", "hidden": hidden, "dropout": 0.3, "weight_decay": wd}
                    )

            shared_torch_kwargs = {
                "loss": args.loss,
                "temperature": args.temperature,
            }
            log.info("Torch loss=%s temperature=%.3f", args.loss, args.temperature)

            if args.skip_cv:
                log.info("Torch: --skip-cv, fitting each config on full train and testing on test.")
                for ci, cfg in enumerate(torch_configs):
                    log.info("Torch %d/%d: %s", ci + 1, len(torch_configs), cfg)
                    try:
                        _, pred_te = torch_fit_predict(
                            X_train, Y_train_model, X_test,
                            device=device, seed=args.seed,
                            groups=g_train,
                            **shared_torch_kwargs, **cfg,
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
                            X_train[tr_idx], Y_train_model[tr_idx], X_train[va_idx],
                            device=device, seed=args.seed,
                            groups=g_train[tr_idx],
                            **shared_torch_kwargs, **cfg,
                        )
                        cv_scores.append(dim_r(Y_train_model[va_idx], pred_va))
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
                            X_train, Y_train_model, X_test,
                            device=device, seed=args.seed,
                            groups=g_train,
                            **shared_torch_kwargs, **cfg,
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
    tag_parts = [
        f"h20_sweep__{roi_name}",
        args.feature_model,
        summary_model,
        f"loss-{args.loss}",
    ]
    if use_features:
        feat_tag = args.features_tag or Path(args.features_dir).resolve().name
        tag_parts.append(f"feat-{feat_tag}")
    if args.target_pca and args.target_pca > 0:
        tag_parts.append(f"pca-{args.target_pca}")
    if args.select_voxels and args.select_voxels > 0 and not use_features:
        tag_parts.append(f"vox-{args.select_voxels}")
    if args.brain_pca and args.brain_pca > 0:
        tag_parts.append(f"brainpca-{args.brain_pca}")
    if args.skip_cv:
        tag_parts.append("no-cv")
    out_path = out_dir / ("__".join(tag_parts) + ".csv")
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
    n_test_stories = results_sorted[0].get("n_test_stories", 0)
    log.info(
        "Top models (story-level metrics: chance top1=1/%d=%.3f):",
        max(1, n_test_stories), 1.0 / max(1, n_test_stories),
    )
    hidden = {
        "model", "dim_r_test", "retrieval_top1_test", "retrieval_mrr_test",
        "retrieval_mean_rank_test", "story_argmax_in_correct", "story_top1",
        "story_mrr", "story_mean_rank", "n_test_stories",
    }
    for r in results_sorted[:5]:
        log.info(
            "  %-12s dim_r=%.4f TR_top1=%.3f TR_mrr=%.3f | story_top1=%.3f story_mrr=%.3f story_argmax=%.3f (params=%s)",
            r["model"],
            r["dim_r_test"],
            r["retrieval_top1_test"],
            r["retrieval_mrr_test"],
            r.get("story_top1", float("nan")),
            r.get("story_mrr", float("nan")),
            r.get("story_argmax_in_correct", float("nan")),
            {k: v for k, v in r.items() if k not in hidden},
        )


if __name__ == "__main__":
    main()

