#!/usr/bin/env python3
"""
Summary-horizon encoding analysis.

Uses precomputed TR-aligned story summaries as the stimulus representation.
For each summary horizon, this script:

1. Loads one summary per TR for every training story
2. Extracts one feature vector per summary using a chosen text model
3. Applies the same TR trimming and FIR delays as the context-length pipeline
4. Trains an encoding model (summary features -> voxels)
5. Saves per-voxel prediction correlations for each feature-model / horizon pair

This is analogous to ``run_context_encoding.py``, except the experimental
manipulation is the summary horizon used to generate the summaries rather than
the raw word-context length used to extract token features.

Usage
-----
  python run_summaries_encoding.py \
      --subject S1 \
      --summaries-dir /path/to/summaries \
      --summary-model gpt-4o-mini \
      --models gpt1 gpt2 gpt2-pool embedding \
      --encoding-model ridge \
      --voxels-from-rois

  python run_summaries_encoding.py \
      --subject S1 \
      --stories wildwomenanddancingqueens \
      --summary-horizons 20 50 200 500 \
      --summaries-dir /path/to/summaries \
      --encoding-model lgbm \
      --voxels-from-rois

By default, the last few selected stories are held out as a story-level test
set. Use ``--no-story-holdout`` to revert to within-training bootstrap CV only.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
from GPT import GPT  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_ridge.ridge import bootstrap_ridge  # noqa: E402
from utils_ridge.util import make_delayed  # noqa: E402

SUBJECT_TO_UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
MODEL_CHOICES = ["gpt1", "gpt2", "gpt2-pool", "embedding"]
ENCODING_MODEL_CHOICES = ["ridge", "pls-ridge", "mlp", "lgbm"]
DEFAULT_HOLDOUT_COUNT = 5
SUMMARY_FILE_RE = re.compile(
    r"^(?P<story>[^.]+)\.(?P<model>.+)\.ctx(?P<horizon>\d+)\.jsonl$"
)
TRIM_START = 5 + config.TRIM
TRIM_END = config.TRIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summaries_encoding")
np.random.seed(42)


def _configure_huggingface_downloads():
    """Raise HF Hub timeouts for clusters / remote servers."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


def _load_gpt2_with_retry(model_name_or_path, device, n_retries=4):
    """Load GPT-2 tokenizer + model with retries for transient network issues."""
    import time
    from transformers import GPT2Model, GPT2Tokenizer

    last_err = None
    for attempt in range(n_retries):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = GPT2Model.from_pretrained(model_name_or_path).eval().to(device)
            return tokenizer, model
        except Exception as err:
            last_err = err
            wait = 30 * (attempt + 1)
            log.warning(
                "GPT-2 load failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                n_retries,
                err,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        "Could not download/load GPT-2 from Hugging Face. "
        "Either warm the Hugging Face cache on a machine with internet access, "
        "copy the cache to the cluster, or pass --gpt2-model /path/to/local/gpt2.\n"
        f"Original error: {last_err}"
    ) from last_err


def load_ba_rois(ba_subject_dir):
    """Load subject-specific Brodmann area ROI definitions."""
    import glob as globmod

    rois = {}
    for path in sorted(globmod.glob(os.path.join(ba_subject_dir, "*.json"))):
        fname = os.path.basename(path)
        if fname == "BA_full_frontal.json":
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for key, indices in d.items():
            rois[key] = indices
    return rois


def load_voxel_set(subject, all_voxels):
    """Load pretrained language-responsive voxels when available."""
    em_path = os.path.join(config.MODEL_DIR, subject, "encoding_model_perceived.npz")
    if os.path.exists(em_path):
        em = np.load(em_path)
        vox = em["voxels"]
        log.info("Loaded %d language-responsive voxels from pretrained model", len(vox))
        return vox, True
    log.warning(
        "No pretrained model found at %s — using all %d voxels (may require a lot of RAM)",
        em_path,
        all_voxels,
    )
    return np.arange(all_voxels), False


def select_pls_supervision_columns(subject, selected_voxels):
    """Choose a smaller voxel subset to supervise the PLS stage."""
    selected_voxels = np.asarray(selected_voxels, dtype=int)
    responsive_voxels, found = load_voxel_set(subject, len(selected_voxels))
    if not found:
        log.warning(
            "PLS supervision fallback: pretrained language-responsive voxel set is unavailable; "
            "using all %d selected voxels as PLS targets.",
            len(selected_voxels),
        )
        return np.arange(len(selected_voxels), dtype=int), "selected_all"

    selected_lookup = {int(vox): idx for idx, vox in enumerate(selected_voxels)}
    local_idx = np.array(
        [selected_lookup[int(vox)] for vox in responsive_voxels if int(vox) in selected_lookup],
        dtype=int,
    )
    if local_idx.size == 0:
        log.warning(
            "PLS supervision fallback: pretrained language-responsive voxels do not overlap the "
            "selected voxel set; using all %d selected voxels as PLS targets.",
            len(selected_voxels),
        )
        return np.arange(len(selected_voxels), dtype=int), "selected_all"

    log.info(
        "Using %d language-responsive voxels to supervise PLS within %d selected output voxels",
        len(local_idx),
        len(selected_voxels),
    )
    return np.sort(local_idx), "responsive_intersection"


def zscore_columns(mat):
    """Z-score each column, guarding against constant vectors."""
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat[:, None]
    mean = mat.mean(0)
    std = mat.std(0)
    std[std == 0] = 1
    return np.nan_to_num((mat - mean) / std)


def score_prediction_matrix(resp, pred):
    """Column-wise correlation between predicted and observed responses."""
    corrs = (zscore_columns(resp) * zscore_columns(pred)).mean(0)
    corrs[np.isnan(corrs)] = 0
    return corrs


def score_encoding_predictions(stim, resp, wt):
    """Column-wise correlation for a linear model weight matrix."""
    return score_prediction_matrix(resp, np.dot(stim, wt))


def standardize_train_test(train_mat, test_mat=None, eps=1e-6, drop_constant=True):
    """Standardize columns using training statistics and drop near-constant columns."""
    train_mat = np.asarray(train_mat, dtype=np.float64)
    test_mat = None if test_mat is None else np.asarray(test_mat, dtype=np.float64)

    mean = train_mat.mean(0)
    std = train_mat.std(0)
    finite = np.isfinite(std)
    keep = finite & (std > eps) if drop_constant else finite
    if not np.any(keep):
        raise ValueError(
            "No finite columns remained after transformation "
            f"(eps={eps}, drop_constant={drop_constant})."
        )

    mean = mean[keep]
    std = std[keep]
    std[~np.isfinite(std) | (std <= eps)] = 1.0
    train_z = np.nan_to_num((train_mat[:, keep] - mean) / std)
    if test_mat is None:
        test_z = None
    else:
        test_z = np.nan_to_num((test_mat[:, keep] - mean) / std)
    return train_z.astype(np.float32, copy=False), (
        None if test_z is None else test_z.astype(np.float32, copy=False)
    ), keep


def chunked_bootstrap_ridge(
    rstim,
    rresp,
    chunk_size=10000,
    eval_stim=None,
    eval_resp=None,
    return_weights=False,
    **kwargs,
):
    """Run bootstrap_ridge in voxel chunks to reduce memory pressure."""
    n_voxels = rresp.shape[1]
    if n_voxels <= chunk_size:
        wt, valphas, bscorrs = bootstrap_ridge(rstim, rresp, **kwargs)
        cv_corrs = bscorrs.mean(2).max(0)
        corrs = (
            score_encoding_predictions(eval_stim, eval_resp, wt)
            if eval_stim is not None and eval_resp is not None
            else cv_corrs
        )
        del valphas, bscorrs
        return corrs, (wt if return_weights else None), cv_corrs

    all_corrs = np.zeros(n_voxels)
    all_cv_corrs = np.zeros(n_voxels)
    all_wt = (
        np.zeros((rstim.shape[1], n_voxels), dtype=np.float32)
        if return_weights
        else None
    )
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        log.info("  Voxel chunk [%d:%d] / %d", start, end, n_voxels)
        wt, valphas, bscorrs = bootstrap_ridge(rstim, rresp[:, start:end], **kwargs)
        chunk_cv_corrs = bscorrs.mean(2).max(0)
        all_cv_corrs[start:end] = chunk_cv_corrs
        if eval_stim is not None and eval_resp is not None:
            all_corrs[start:end] = score_encoding_predictions(
                eval_stim,
                eval_resp[:, start:end],
                wt,
            )
        else:
            all_corrs[start:end] = chunk_cv_corrs
        if return_weights:
            all_wt[:, start:end] = wt.astype(np.float32, copy=False)
        del wt, valphas, bscorrs
    return all_corrs, all_wt, all_cv_corrs


def fit_chunked_lgbm(
    rstim,
    rresp,
    eval_stim,
    eval_resp,
    chunk_size=10000,
    params=None,
):
    """Fit one LightGBM regressor per voxel and score on held-out stories."""
    if eval_stim is None or eval_resp is None:
        raise ValueError(
            "LightGBM encoding requires held-out test stories. "
            "Do not use --no-story-holdout with --encoding-model lgbm."
        )

    try:
        from lightgbm import LGBMRegressor
    except ImportError as err:
        raise ImportError(
            "lightgbm is not installed. Install dependencies from requirements.txt "
            "or run `pip install lightgbm`."
        ) from err

    params = dict(params or {})
    n_voxels = rresp.shape[1]
    all_corrs = np.zeros(n_voxels)
    all_train_fit_corrs = np.zeros(n_voxels)

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        log.info("  Voxel chunk [%d:%d] / %d", start, end, n_voxels)

        chunk_eval_pred = np.zeros((eval_stim.shape[0], end - start), dtype=np.float32)
        chunk_train_pred = np.zeros((rstim.shape[0], end - start), dtype=np.float32)

        for local_idx, voxel_idx in enumerate(range(start, end)):
            model = LGBMRegressor(**params)
            model.fit(rstim, rresp[:, voxel_idx])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                    category=UserWarning,
                )
                chunk_eval_pred[:, local_idx] = model.predict(
                    eval_stim,
                    validate_features=False,
                ).astype(np.float32, copy=False)
                chunk_train_pred[:, local_idx] = model.predict(
                    rstim,
                    validate_features=False,
                ).astype(np.float32, copy=False)

            if (local_idx + 1) % 250 == 0 or voxel_idx == end - 1:
                log.info(
                    "    fitted %d / %d voxels in current chunk",
                    local_idx + 1,
                    end - start,
                )

        all_corrs[start:end] = score_prediction_matrix(eval_resp[:, start:end], chunk_eval_pred)
        all_train_fit_corrs[start:end] = score_prediction_matrix(
            rresp[:, start:end],
            chunk_train_pred,
        )

    return all_corrs, None, np.full(n_voxels, np.nan), all_train_fit_corrs


def fit_chunked_mlp(
    rstim,
    rresp,
    eval_stim,
    eval_resp,
    output_chunk_size=512,
    hidden_dim=128,
    activation="relu",
    alpha=1e-4,
    learning_rate_init=1e-3,
    batch_size=64,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
):
    """Fit one-hidden-layer PyTorch MLP regressors over chunks of output voxels."""
    if eval_stim is None or eval_resp is None:
        raise ValueError(
            "MLP encoding requires held-out test stories. "
            "Do not use --no-story-holdout with --encoding-model mlp."
        )

    device = torch.device(config.EM_DEVICE)
    log.info("  Torch MLP device: %s", device)

    def make_activation(name):
        mapping = {
            "identity": torch.nn.Identity,
            "logistic": torch.nn.Sigmoid,
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
        }
        return mapping[name]()

    class ChunkMLP(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                make_activation(activation),
                torch.nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    n_voxels = rresp.shape[1]
    all_corrs = np.zeros(n_voxels)
    all_train_fit_corrs = np.zeros(n_voxels)
    rstim_t = torch.tensor(rstim, dtype=torch.float32, device=device)
    eval_stim_t = torch.tensor(eval_stim, dtype=torch.float32, device=device)

    for start in range(0, n_voxels, output_chunk_size):
        end = min(start + output_chunk_size, n_voxels)
        log.info("  Output chunk [%d:%d] / %d", start, end, n_voxels)

        train_chunk = rresp[:, start:end]
        y_mean = train_chunk.mean(0)
        y_std = train_chunk.std(0)
        y_std[y_std == 0] = 1
        train_chunk_z = np.nan_to_num((train_chunk - y_mean) / y_std)
        train_chunk_t = torch.tensor(train_chunk_z, dtype=torch.float32, device=device)
        model = ChunkMLP(rstim.shape[1], end - start).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate_init,
            weight_decay=alpha,
        )
        loss_fn = torch.nn.MSELoss()

        n_train = rstim.shape[0]
        use_early_stopping = early_stopping and n_train >= 10
        if use_early_stopping:
            n_val = max(1, int(round(n_train * validation_fraction)))
            n_val = min(n_val, n_train - 1)
            rng = np.random.default_rng(42)
            perm = rng.permutation(n_train)
            val_idx_np = perm[:n_val]
            fit_idx_np = perm[n_val:]
            fit_idx = torch.tensor(fit_idx_np, dtype=torch.long, device=device)
            val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=device)
            best_val = float("inf")
            best_state = None
            patience = 0
        else:
            fit_idx = torch.arange(n_train, device=device)
            val_idx = None
            best_val = None
            best_state = None
            patience = 0

        for epoch in range(max_iter):
            model.train()
            shuffled = fit_idx[torch.randperm(len(fit_idx), device=device)]
            for batch_start in range(0, len(shuffled), batch_size):
                batch_idx = shuffled[batch_start:batch_start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = model(rstim_t[batch_idx])
                loss = loss_fn(pred, train_chunk_t[batch_idx])
                loss.backward()
                optimizer.step()

            if use_early_stopping:
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(rstim_t[val_idx]), train_chunk_t[val_idx]).item()
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= n_iter_no_change:
                        log.info(
                            "    early stop at epoch %d with val_loss=%.6f",
                            epoch + 1,
                            best_val,
                        )
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        model.eval()
        with torch.no_grad():
            eval_pred = model(eval_stim_t).detach().cpu().numpy().astype(np.float32, copy=False)
            train_pred = model(rstim_t).detach().cpu().numpy().astype(np.float32, copy=False)

        all_corrs[start:end] = score_prediction_matrix(eval_resp[:, start:end], eval_pred)
        all_train_fit_corrs[start:end] = score_prediction_matrix(
            train_chunk,
            train_pred,
        )

    return all_corrs, None, np.full(n_voxels, np.nan), all_train_fit_corrs


def fit_encoding_model(
    args,
    train_rstim,
    train_rresp,
    test_rstim,
    test_rresp,
    pls_fit_rresp=None,
):
    """Dispatch to the requested encoding model backend."""
    alphas = np.array([args.single_alpha]) if args.single_alpha else config.ALPHAS
    nchunks = int(np.ceil(train_rresp.shape[0] / 5 / config.CHUNKLEN))

    if args.encoding_model == "ridge":
        log.info(
            "Bootstrap ridge regression (%d boots, chunklen=%d, nchunks=%d, alphas=%s)...",
            args.nboots,
            config.CHUNKLEN,
            nchunks,
            alphas,
        )
        corrs, wt, cv_corrs = chunked_bootstrap_ridge(
            train_rstim,
            train_rresp,
            chunk_size=args.voxel_chunk_size,
            eval_stim=test_rstim,
            eval_resp=test_rresp,
            return_weights=args.save_weights,
            alphas=alphas,
            nboots=args.nboots,
            chunklen=config.CHUNKLEN,
            nchunks=nchunks,
            use_corr=True,
        )
        return corrs, wt, cv_corrs, None

    if args.encoding_model == "pls-ridge":
        if args.save_weights:
            raise ValueError("--save-weights is not yet supported for --encoding-model pls-ridge.")
        if test_rstim is None:
            log.warning(
                "pls-ridge without story holdout fits the PLS transform before bootstrap CV, "
                "so the reported CV score may be optimistic. Prefer the default held-out-story evaluation."
            )
        if args.single_alpha is not None:
            log.warning(
                "Using --single-alpha=%s for pls-ridge. If this value came from plain ridge, "
                "it may not transfer well to the PLS latent space; consider omitting it first.",
                args.single_alpha,
            )

        from sklearn.cross_decomposition import PLSRegression

        max_components = min(
            args.pls_n_components,
            train_rstim.shape[0] - 1,
            train_rstim.shape[1],
        )
        if max_components < 1:
            raise ValueError(
                "PLS needs at least one component. Check the number of training TRs "
                "and delayed features."
            )
        if max_components < args.pls_n_components:
            log.info(
                "Reducing PLS components from requested %d to %d based on data shape",
                args.pls_n_components,
                max_components,
            )

        pls = PLSRegression(
            n_components=max_components,
            scale=args.pls_scale,
            max_iter=args.pls_max_iter,
            tol=args.pls_tol,
            copy=False,
        )
        log.info(
            "Fitting PLS transform (n_components=%d, scale=%s) on training stories...",
            max_components,
            args.pls_scale,
        )
        pls_targets = train_rresp if pls_fit_rresp is None else pls_fit_rresp
        pls_targets = zscore_columns(pls_targets).astype(np.float32, copy=False)
        log.info(
            "PLS supervision matrix: %s (TRs x target voxels)",
            pls_targets.shape,
        )
        train_scores = pls.fit_transform(train_rstim, pls_targets)[0].astype(np.float32, copy=False)
        if test_rstim is not None:
            test_scores = pls.transform(test_rstim).astype(np.float32, copy=False)
        else:
            test_scores = None

        raw_score_std = train_scores.std(0)
        log.info(
            "PLS raw score std: min=%.3e median=%.3e max=%.3e",
            float(raw_score_std.min()),
            float(np.median(raw_score_std)),
            float(raw_score_std.max()),
        )
        log.info(
            "PLS latent diagnostics: finite=%d/%d nonzero=%d/%d",
            int(np.isfinite(raw_score_std).sum()),
            len(raw_score_std),
            int((raw_score_std > 0).sum()),
            len(raw_score_std),
        )
        if float(raw_score_std.max()) < 1e-6:
            log.warning(
                "PLS scores are extremely small in absolute scale; keeping any non-constant "
                "latent dimensions and standardizing them before ridge."
            )
        train_scores, test_scores, keep = standardize_train_test(
            train_scores,
            test_scores,
            eps=0.0,
            drop_constant=False,
        )
        log.info(
            "Retained %d / %d PLS components after latent standardization",
            int(keep.sum()),
            len(keep),
        )

        log.info(
            "Bootstrap ridge regression on PLS scores (%d boots, chunklen=%d, nchunks=%d, alphas=%s)...",
            args.nboots,
            config.CHUNKLEN,
            nchunks,
            alphas,
        )
        corrs, wt, cv_corrs = chunked_bootstrap_ridge(
            train_scores,
            train_rresp,
            chunk_size=args.voxel_chunk_size,
            eval_stim=test_scores,
            eval_resp=test_rresp,
            return_weights=False,
            alphas=alphas,
            nboots=args.nboots,
            chunklen=config.CHUNKLEN,
            nchunks=nchunks,
            use_corr=True,
        )
        del train_scores, test_scores, pls, wt
        return corrs, None, cv_corrs, None

    if args.encoding_model == "mlp":
        if args.save_weights:
            raise ValueError("--save-weights is not supported for --encoding-model mlp.")
        if args.single_alpha is not None:
            raise ValueError("--single-alpha only applies to --encoding-model ridge or pls-ridge.")

        mlp_params = {
            "output_chunk_size": args.mlp_output_chunk_size,
            "hidden_dim": args.mlp_hidden_dim,
            "activation": args.mlp_activation,
            "alpha": args.mlp_alpha,
            "learning_rate_init": args.mlp_learning_rate_init,
            "batch_size": args.mlp_batch_size,
            "max_iter": args.mlp_max_iter,
            "early_stopping": args.mlp_early_stopping,
            "validation_fraction": args.mlp_validation_fraction,
            "n_iter_no_change": args.mlp_n_iter_no_change,
        }
        log.info("MLP regression with params: %s", mlp_params)
        return fit_chunked_mlp(
            train_rstim,
            train_rresp,
            test_rstim,
            test_rresp,
            **mlp_params,
        )

    if args.save_weights:
        raise ValueError("--save-weights is only supported for --encoding-model ridge.")
    if args.single_alpha is not None:
        raise ValueError("--single-alpha only applies to --encoding-model ridge or pls-ridge.")

    lgbm_params = {
        "n_estimators": args.lgbm_n_estimators,
        "learning_rate": args.lgbm_learning_rate,
        "num_leaves": args.lgbm_num_leaves,
        "max_depth": args.lgbm_max_depth,
        "min_child_samples": args.lgbm_min_child_samples,
        "subsample": args.lgbm_subsample,
        "colsample_bytree": args.lgbm_colsample_bytree,
        "reg_alpha": args.lgbm_reg_alpha,
        "reg_lambda": args.lgbm_reg_lambda,
        "n_jobs": args.lgbm_n_jobs,
        "random_state": 42,
        "verbosity": -1,
        "force_col_wise": True,
    }
    log.info("LightGBM regression with params: %s", lgbm_params)
    return fit_chunked_lgbm(
        train_rstim,
        train_rresp,
        test_rstim,
        test_rresp,
        chunk_size=args.voxel_chunk_size,
        params=lgbm_params,
    )


def sanitize_name(value):
    """Make strings safe for filenames while keeping them readable."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def make_story_split_tag(train_stories, test_stories):
    """Compact filename-safe tag describing the train/test story split."""
    if not test_stories:
        return "bootstrap-cv"
    digest = hashlib.md5("\n".join(test_stories).encode("utf-8")).hexdigest()[:8]
    return f"holdout-{len(test_stories)}-{digest}"


def load_story_list(args):
    """Resolve the ordered story list from --stories or --sessions."""
    if args.stories:
        return list(args.stories)

    sess_to_story_path = Path(config.DATA_TRAIN_DIR) / "sess_to_story.json"
    with open(sess_to_story_path, encoding="utf-8") as f:
        sess_to_story = json.load(f)

    stories = []
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def split_story_list(stories, args):
    """Split selected stories into training and held-out test stories."""
    if args.no_story_holdout:
        return list(stories), []

    if args.holdout_stories:
        requested = list(dict.fromkeys(args.holdout_stories))
        missing = [story for story in requested if story not in stories]
        if missing:
            raise ValueError(
                f"--holdout-stories contains stories not in the selected set: {missing}"
            )
        holdout_set = set(requested)
        test_stories = [story for story in stories if story in holdout_set]
    else:
        if args.holdout_count <= 0:
            return list(stories), []
        if args.holdout_count >= len(stories):
            raise ValueError(
                f"--holdout-count ({args.holdout_count}) must be smaller than the "
                f"number of selected stories ({len(stories)})."
            )
        test_stories = list(stories[-args.holdout_count :])

    train_stories = [story for story in stories if story not in set(test_stories)]
    if not train_stories:
        raise ValueError("Story holdout left zero training stories.")
    return train_stories, test_stories


def build_summary_index(summaries_dir):
    """Scan summary JSONL files and index them by (story, model, horizon)."""
    index = {}
    for path in sorted(summaries_dir.glob("*.jsonl")):
        match = SUMMARY_FILE_RE.match(path.name)
        if not match:
            continue
        key = (
            match.group("story"),
            match.group("model"),
            int(match.group("horizon")),
        )
        if key in index:
            raise ValueError(
                f"Duplicate summary file for story/model/horizon {key}: "
                f"{index[key]} and {path}"
            )
        index[key] = path

    if not index:
        raise FileNotFoundError(
            f"No summary JSONL files matching '<story>.<model>.ctx<h>.jsonl' found in "
            f"{summaries_dir}"
        )
    return index


def resolve_summary_model(index, stories, requested_model=None):
    """Choose the summary model to analyze."""
    models_by_story = {}
    for story in stories:
        models = {model for (s, model, _h) in index if s == story}
        if not models:
            raise FileNotFoundError(f"No summary files found for story '{story}'")
        models_by_story[story] = models

    if requested_model:
        missing = [story for story, models in models_by_story.items() if requested_model not in models]
        if missing:
            raise FileNotFoundError(
                f"Requested --summary-model '{requested_model}' is missing for stories: "
                + ", ".join(missing)
            )
        return requested_model

    common_models = set.intersection(*(models for models in models_by_story.values()))
    if len(common_models) == 1:
        return next(iter(common_models))

    details = ", ".join(
        f"{story}: {sorted(models)}" for story, models in sorted(models_by_story.items())
    )
    raise ValueError(
        "Could not infer a unique summary model across the selected stories. "
        "Pass --summary-model explicitly. Available models by story: "
        f"{details}"
    )


def resolve_summary_horizons(index, stories, summary_model, requested_horizons=None):
    """Choose which horizons to analyze, requiring availability for all stories."""
    horizons_by_story = {}
    for story in stories:
        horizons = {h for (s, model, h) in index if s == story and model == summary_model}
        if not horizons:
            raise FileNotFoundError(
                f"No summary files found for story '{story}' and model '{summary_model}'"
            )
        horizons_by_story[story] = horizons

    common_horizons = sorted(set.intersection(*(horizons for horizons in horizons_by_story.values())))
    if not common_horizons:
        details = ", ".join(
            f"{story}: {sorted(horizons)}"
            for story, horizons in sorted(horizons_by_story.items())
        )
        raise FileNotFoundError(
            f"No summary horizons are shared across all stories for model '{summary_model}'. "
            f"Found: {details}"
        )

    if requested_horizons:
        requested = sorted(set(requested_horizons))
        missing = [h for h in requested if h not in common_horizons]
        if missing:
            details = ", ".join(
                f"{story}: {sorted(horizons)}"
                for story, horizons in sorted(horizons_by_story.items())
            )
            raise FileNotFoundError(
                "Some requested --summary-horizons are not available for every selected story. "
                f"Missing horizons: {missing}. Available by story: {details}"
            )
        return requested

    return common_horizons


def load_resp_info(subject, stories):
    """Read response shapes without loading the full response matrices."""
    resp_dir = Path(config.DATA_TRAIN_DIR) / "train_response" / subject
    lengths = {}
    n_voxels = None
    for story in stories:
        resp_path = resp_dir / f"{story}.hf5"
        if not resp_path.exists():
            raise FileNotFoundError(f"Missing response file: {resp_path}")
        with h5py.File(resp_path, "r") as hf:
            shape = hf["data"].shape
            lengths[story] = int(shape[0])
            if n_voxels is None:
                n_voxels = int(shape[1])
            elif int(shape[1]) != n_voxels:
                raise ValueError(
                    f"Inconsistent voxel count across response files: "
                    f"{resp_path} has {int(shape[1])}, expected {n_voxels}"
                )
    return lengths, n_voxels


def load_summary_texts(path, expected_story, expected_model, expected_horizon):
    """Load and validate one summary JSONL file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Summary file is empty: {path}")

    rows.sort(key=lambda row: int(row["tr_index"]))
    tr_indices = [int(row["tr_index"]) for row in rows]
    expected_indices = list(range(len(rows)))
    if tr_indices != expected_indices:
        raise ValueError(
            f"Summary file has non-consecutive tr_index values: {path} "
            f"(found {tr_indices[:10]}...)"
        )

    stories_in_file = {str(row.get("story", "")) for row in rows}
    if stories_in_file != {expected_story}:
        raise ValueError(
            f"Summary file story mismatch in {path}: expected '{expected_story}', "
            f"found {sorted(stories_in_file)}"
        )

    models_in_file = {str(row.get("model", "")) for row in rows}
    if models_in_file != {expected_model}:
        raise ValueError(
            f"Summary file model mismatch in {path}: expected '{expected_model}', "
            f"found {sorted(models_in_file)}"
        )

    horizons_in_file = {int(row.get("context_window_words", -1)) for row in rows}
    if horizons_in_file != {expected_horizon}:
        raise ValueError(
            f"Summary file horizon mismatch in {path}: expected {expected_horizon}, "
            f"found {sorted(horizons_in_file)}"
        )

    summary_word_values = {
        int(row["summary_words"])
        for row in rows
        if row.get("summary_words") is not None
    }
    if len(summary_word_values) > 1:
        raise ValueError(
            f"Inconsistent summary_words values in {path}: {sorted(summary_word_values)}"
        )

    return {
        "texts": [str(row.get("summary", "")) for row in rows],
        "summary_words": next(iter(summary_word_values)) if summary_word_values else None,
    }


class SummaryEmbeddingEncoder:
    """Sentence-transformer wrapper with stable handling of empty summaries."""

    def __init__(self, model_name, device, batch_size):
        _configure_huggingface_downloads()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        log.info("Loading summary embedding model %r on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        nonempty_idx = [i for i, text in enumerate(texts) if text.strip()]
        if not nonempty_idx:
            return vecs

        enc = self.model.encode(
            [texts[i] for i in nonempty_idx],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        vecs[nonempty_idx] = enc
        return vecs

    def close(self):
        del self.model
        torch.cuda.empty_cache()


class GPT1SummaryEncoder:
    """Encode each summary with GPT-1 layer-9 hidden state of the final word."""

    def __init__(self, device, batch_size):
        vocab_path = os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)

        self.device = device
        self.batch_size = batch_size
        self.backend = "perceived"
        self.gpt = GPT(
            path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
            vocab=vocab,
            device=device,
        )
        hidden_dim = getattr(self.gpt.model.config, "n_embd", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.gpt.model.config, "hidden_size")
        self.dim = int(hidden_dim)

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        token_lists = [
            [token for token in text.split() if token.strip()]
            for text in texts
        ]
        nonempty = [(i, tokens) for i, tokens in enumerate(token_lists) if tokens]
        if not nonempty:
            return vecs

        for start in range(0, len(nonempty), self.batch_size):
            batch = nonempty[start:start + self.batch_size]
            lengths = [len(tokens) for _idx, tokens in batch]
            max_len = max(lengths)

            ids = np.full((len(batch), max_len), self.gpt.UNK_ID, dtype=np.int64)
            mask = np.zeros((len(batch), max_len), dtype=np.int64)
            row_to_original = []

            for row_index, (original_index, tokens) in enumerate(batch):
                token_ids = self.gpt.encode(tokens)
                ids[row_index, :len(token_ids)] = token_ids
                mask[row_index, :len(token_ids)] = 1
                row_to_original.append(original_index)

            ids_t = torch.tensor(ids, device=self.device)
            mask_t = torch.tensor(mask, device=self.device)
            with torch.no_grad():
                outputs = self.gpt.model(
                    input_ids=ids_t,
                    attention_mask=mask_t,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[config.GPT_LAYER]
            last_idx = mask_t.sum(dim=1) - 1
            batch_vecs = hidden[
                torch.arange(hidden.shape[0], device=self.device),
                last_idx,
            ].detach().cpu().numpy().astype(np.float32)
            vecs[row_to_original] = batch_vecs

        return vecs

    def close(self):
        del self.gpt
        torch.cuda.empty_cache()


class GPT2SummaryEncoder:
    """Encode each summary with GPT-2 layer-9 last-token or mean-pooled state."""

    def __init__(self, model_name_or_path, device, batch_size, pool):
        _configure_huggingface_downloads()
        self.device = device
        self.batch_size = batch_size
        self.pool = pool
        self.backend = model_name_or_path
        pool_tag = " (mean-pool)" if pool else " (last-token)"
        log.info("Loading GPT-2 from %r%s", model_name_or_path, pool_tag)
        self.tokenizer, self.model = _load_gpt2_with_retry(model_name_or_path, device)
        self.dim = int(self.model.config.n_embd)

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        nonempty_idx = [i for i, text in enumerate(texts) if text.strip()]
        if not nonempty_idx:
            return vecs

        for start in range(0, len(nonempty_idx), self.batch_size):
            batch_idx = nonempty_idx[start:start + self.batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            tok = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            tok = {key: value.to(self.device) for key, value in tok.items()}
            with torch.no_grad():
                outputs = self.model(**tok, output_hidden_states=True)
            hidden = outputs.hidden_states[config.GPT_LAYER]
            attention_mask = tok["attention_mask"]
            if self.pool:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batch_vecs = pooled.detach().cpu().numpy().astype(np.float32)
            else:
                last_idx = attention_mask.sum(dim=1) - 1
                batch_vecs = hidden[
                    torch.arange(hidden.shape[0], device=self.device),
                    last_idx,
                ].detach().cpu().numpy().astype(np.float32)
            vecs[batch_idx] = batch_vecs

        return vecs

    def close(self):
        del self.model
        torch.cuda.empty_cache()


def make_summary_encoder(model_type, args, device):
    """Construct the requested summary feature encoder."""
    if model_type == "gpt1":
        encoder = GPT1SummaryEncoder(device=device, batch_size=args.embed_batch_size)
    elif model_type == "gpt2":
        encoder = GPT2SummaryEncoder(
            model_name_or_path=args.gpt2_model,
            device=device,
            batch_size=args.embed_batch_size,
            pool=False,
        )
    elif model_type == "gpt2-pool":
        encoder = GPT2SummaryEncoder(
            model_name_or_path=args.gpt2_model,
            device=device,
            batch_size=args.embed_batch_size,
            pool=True,
        )
    elif model_type == "embedding":
        encoder = SummaryEmbeddingEncoder(
            model_name=args.embedding_model,
            device=device,
            batch_size=args.embed_batch_size,
        )
        encoder.backend = args.embedding_model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    encoder.feature_model = model_type
    return encoder


def build_design_matrix(stories, texts_by_story, resp_lengths, encoder):
    """Embed per-TR summaries, trim to response-aligned TRs, z-score, add delays."""
    trimmed_story_vecs = []

    for story in stories:
        texts = texts_by_story[story]
        if len(texts) <= TRIM_START + TRIM_END:
            raise ValueError(
                f"Story '{story}' has only {len(texts)} summary TRs; "
                f"need more than {TRIM_START + TRIM_END} to apply trimming."
            )

        story_vecs = encoder.encode(texts)
        trimmed = story_vecs[TRIM_START:-TRIM_END]
        expected_resp_trs = resp_lengths[story]
        if trimmed.shape[0] != expected_resp_trs:
            raise ValueError(
                f"Story '{story}' has {len(texts)} summary TRs, which trims to "
                f"{trimmed.shape[0]} TRs, but the response file has {expected_resp_trs} TRs. "
                "This usually means the summary JSONL is incomplete or was generated against "
                "different story timing."
            )

        trimmed_story_vecs.append(trimmed)
        log.info(
            "  %s: %d summaries -> %s after trim -> response %d TRs",
            story,
            len(texts),
            trimmed.shape,
            expected_resp_trs,
        )

    ds_mat = np.vstack(trimmed_story_vecs)
    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    ds_mat = np.nan_to_num((ds_mat - r_mean) / r_std)
    return make_delayed(ds_mat, config.STIM_DELAYS)


def format_condition_row(meta):
    """Compact human-readable label for summaries ROI tables."""
    return f"{meta['encoding_model']}_{meta['feature_model']}_h{meta['summary_horizon']}"


def print_roi_summary(all_corrs, ba_subject_dir, vox, label_to_meta):
    """Print mean encoding correlation per ROI for each summary horizon."""
    rois = load_ba_rois(ba_subject_dir)
    region_names = sorted(rois.keys())
    global_to_local = {int(g): i for i, g in enumerate(vox)}

    local_rois = {}
    for region_name in region_names:
        local_rois[region_name] = np.array(
            [global_to_local[v] for v in rois[region_name] if v in global_to_local],
            dtype=int,
        )

    hdr = f"  {'condition':<22s}"
    for region_name in region_names:
        hdr += f"  {region_name + f' ({len(local_rois[region_name])})':>25s}"
    hdr += f"  {'all':>10s}"

    print("\n" + "=" * len(hdr))
    print("  Per-ROI mean encoding correlation")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label in sorted(
        all_corrs,
        key=lambda item: (
            label_to_meta[item]["encoding_model"],
            label_to_meta[item]["feature_model"],
            label_to_meta[item]["summary_horizon"],
        ),
    ):
        corrs = all_corrs[label]
        display_label = format_condition_row(label_to_meta[label])
        row = f"  {display_label:<22s}"
        for region_name in region_names:
            idx = local_rois[region_name]
            mean_r = corrs[idx].mean() if len(idx) > 0 else float("nan")
            row += f"  {mean_r:25.4f}"
        row += f"  {corrs.mean():10.4f}"
        print(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--stories",
        nargs="+",
        default=None,
        help="Story pool to analyze. If omitted, stories are derived from --sessions.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
        help="Training sessions to include when --stories is not provided.",
    )
    parser.add_argument(
        "--holdout-stories",
        nargs="+",
        default=None,
        help="Explicit held-out test stories. Must be a subset of the selected story pool.",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=DEFAULT_HOLDOUT_COUNT,
        help=(
            "If --holdout-stories is omitted, hold out the last N selected stories as "
            "the story-level test set."
        ),
    )
    parser.add_argument(
        "--no-story-holdout",
        action="store_true",
        help="Disable story-level holdout and report only within-training bootstrap CV.",
    )
    parser.add_argument(
        "--summaries-dir",
        default=str(REPO_DIR / "generate_summaries" / "outputs"),
        help="Directory containing summary JSONL files.",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Summary source model to use, e.g. gpt-4o-mini. If omitted, inferred.",
    )
    parser.add_argument(
        "--summary-horizons",
        nargs="+",
        type=int,
        default=None,
        help="Summary horizons to analyze. If omitted, uses all horizons shared across stories.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_CHOICES,
        choices=MODEL_CHOICES,
        help="Feature models to use for encoding the summary text.",
    )
    parser.add_argument(
        "--encoding-model",
        default="ridge",
        choices=ENCODING_MODEL_CHOICES,
        help="Encoding model backend used to predict voxel responses from summary features.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model used to embed each summary.",
    )
    parser.add_argument(
        "--gpt2-model",
        default="openai-community/gpt2",
        help="Hugging Face GPT-2 id or local directory for gpt2 / gpt2-pool features.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for summary feature extraction.",
    )
    parser.add_argument(
        "--ba-dir",
        default=str(REPO_DIR / "ba_indices"),
        help="Directory containing per-subject Brodmann area indices.",
    )
    parser.add_argument("--nboots", type=int, default=config.NBOOTS)
    parser.add_argument(
        "--single-alpha",
        type=float,
        default=None,
        help="Use a single fixed ridge alpha instead of cross-validated search (ridge / pls-ridge).",
    )
    parser.add_argument(
        "--pls-n-components",
        type=int,
        default=64,
        help="Number of latent components for --encoding-model pls-ridge.",
    )
    parser.add_argument(
        "--pls-scale",
        action="store_true",
        help="Scale X and Y inside PLSRegression for --encoding-model pls-ridge.",
    )
    parser.add_argument(
        "--pls-max-iter",
        type=int,
        default=500,
        help="Maximum iterations for the PLS solver in --encoding-model pls-ridge.",
    )
    parser.add_argument(
        "--pls-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for the PLS solver in --encoding-model pls-ridge.",
    )
    parser.add_argument(
        "--mlp-output-chunk-size",
        type=int,
        default=512,
        help="Number of voxel outputs per one-hidden-layer PyTorch MLP when --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=128,
        help="Hidden width for the one-hidden-layer PyTorch MLP when --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-activation",
        default="relu",
        choices=["identity", "logistic", "tanh", "relu"],
        help="Hidden activation for the PyTorch MLP when --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-alpha",
        type=float,
        default=1e-4,
        help="Adam weight decay for the PyTorch MLP when --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=1e-3,
        help="Initial learning rate for the PyTorch MLP when --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-batch-size",
        type=int,
        default=64,
        help="Mini-batch size for --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=200,
        help="Maximum iterations for --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-early-stopping",
        action="store_true",
        help="Enable internal validation-based early stopping for --encoding-model mlp.",
    )
    parser.add_argument(
        "--mlp-validation-fraction",
        type=float,
        default=0.1,
        help="Validation fraction used when --mlp-early-stopping is enabled.",
    )
    parser.add_argument(
        "--mlp-n-iter-no-change",
        type=int,
        default=10,
        help="Patience for --encoding-model mlp.",
    )
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-num-leaves",
        type=int,
        default=31,
        help="Number of leaves per tree for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-max-depth",
        type=int,
        default=-1,
        help="Maximum tree depth for --encoding-model lgbm (-1 = unlimited).",
    )
    parser.add_argument(
        "--lgbm-min-child-samples",
        type=int,
        default=20,
        help="Minimum child samples for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-subsample",
        type=float,
        default=0.8,
        help="Row subsampling fraction for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-colsample-bytree",
        type=float,
        default=0.8,
        help="Feature subsampling fraction per tree for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-reg-alpha",
        type=float,
        default=0.0,
        help="L1 regularization for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-reg-lambda",
        type=float,
        default=0.0,
        help="L2 regularization for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=1,
        help="Threads per voxel model for --encoding-model lgbm.",
    )
    parser.add_argument(
        "--output-dir",
        default="summaries_encoding_results",
        help="Results directory root relative to the repo root.",
    )
    parser.add_argument(
        "--voxels-from-rois",
        action="store_true",
        help="Restrict to frontal voxels from ba_indices/ using BA_full_frontal.json.",
    )
    parser.add_argument(
        "--all-voxels",
        action="store_true",
        help="Use all voxels instead of the pretrained language-responsive set.",
    )
    parser.add_argument(
        "--voxel-chunk-size",
        type=int,
        default=10000,
        help="Voxel chunk size when using large voxel sets.",
    )
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Also save full regression weights (large files).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse any existing per-horizon .npz file instead of recomputing it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = config.EM_DEVICE
    log.info("Device: %s", device)

    stories = load_story_list(args)
    train_stories, test_stories = split_story_list(stories, args)
    log.info("Selected stories (%d): %s", len(stories), stories)
    log.info("Training stories (%d): %s", len(train_stories), train_stories)
    if test_stories:
        log.info("Held-out test stories (%d): %s", len(test_stories), test_stories)
    else:
        log.info("Story-level holdout disabled; scoring will use bootstrap CV within training stories")

    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    if not summaries_dir.is_dir():
        raise FileNotFoundError(f"No summaries directory found at {summaries_dir}")

    summary_index = build_summary_index(summaries_dir)
    summary_model = resolve_summary_model(summary_index, stories, args.summary_model)
    summary_horizons = resolve_summary_horizons(
        summary_index, stories, summary_model, args.summary_horizons
    )
    log.info("Summary model: %s", summary_model)
    log.info("Summary horizons: %s", summary_horizons)
    log.info("Feature models: %s", args.models)
    log.info("Encoding model: %s", args.encoding_model)

    train_resp_lengths, total_voxels = load_resp_info(args.subject, train_stories)
    if test_stories:
        test_resp_lengths, test_total_voxels = load_resp_info(args.subject, test_stories)
        if test_total_voxels != total_voxels:
            raise ValueError(
                f"Train/test response voxel counts do not match: "
                f"{total_voxels} vs {test_total_voxels}"
            )
    else:
        test_resp_lengths = {}
    log.info("Response matrix width: %d voxels", total_voxels)

    uts_id = SUBJECT_TO_UTS.get(args.subject)
    ba_subject_dir = os.path.join(args.ba_dir, uts_id) if uts_id else None

    if args.voxels_from_rois:
        if not ba_subject_dir or not os.path.isdir(ba_subject_dir):
            log.error(
                "--voxels-from-rois: no BA directory found at %s (subject %s -> %s)",
                ba_subject_dir,
                args.subject,
                uts_id,
            )
            sys.exit(1)
        frontal_path = os.path.join(ba_subject_dir, "BA_full_frontal.json")
        with open(frontal_path, encoding="utf-8") as f:
            frontal = json.load(f)
        frontal_voxels = list(frontal.values())[0]
        vox = np.sort(np.array(frontal_voxels, dtype=int))
        vox = vox[vox < total_voxels]
        log.info("Using %d frontal voxels from %s", len(vox), frontal_path)
    elif args.all_voxels:
        vox = np.arange(total_voxels)
        log.info("Using ALL %d voxels (chunked processing)", len(vox))
    else:
        vox, _ = load_voxel_set(args.subject, total_voxels)

    train_rresp = get_resp(args.subject, train_stories, stack=True, vox=vox)
    log.info("Training response matrix: %s (TRs x voxels)", train_rresp.shape)
    if test_stories:
        test_rresp = get_resp(args.subject, test_stories, stack=True, vox=vox)
        log.info("Held-out test response matrix: %s (TRs x voxels)", test_rresp.shape)
    else:
        test_rresp = None

    if args.encoding_model == "pls-ridge":
        pls_supervision_idx, pls_supervision_mode = select_pls_supervision_columns(
            args.subject,
            vox,
        )
        pls_fit_rresp = train_rresp[:, pls_supervision_idx]
        log.info(
            "PLS supervision targets: %d voxels (%s)",
            pls_fit_rresp.shape[1],
            pls_supervision_mode,
        )
    else:
        pls_supervision_idx = None
        pls_supervision_mode = None
        pls_fit_rresp = None

    out_dir = Path(config.REPO_DIR) / args.output_dir / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    all_corrs = {}
    label_to_meta = {}
    safe_summary_model = sanitize_name(summary_model)
    safe_encoding_model = sanitize_name(args.encoding_model)
    split_tag = make_story_split_tag(train_stories, test_stories)
    summary_cache = {}

    def get_cached_summaries(horizon):
        if horizon in summary_cache:
            return summary_cache[horizon]

        texts_by_story = {}
        summary_word_values = set()
        for story in stories:
            path = summary_index[(story, summary_model, horizon)]
            loaded = load_summary_texts(
                path=path,
                expected_story=story,
                expected_model=summary_model,
                expected_horizon=horizon,
            )
            texts_by_story[story] = loaded["texts"]
            if loaded["summary_words"] is not None:
                summary_word_values.add(loaded["summary_words"])

        if len(summary_word_values) > 1:
            raise ValueError(
                f"Summary horizon {horizon} has inconsistent summary_words values: "
                f"{sorted(summary_word_values)}"
            )

        summary_cache[horizon] = (
            texts_by_story,
            next(iter(summary_word_values)) if summary_word_values else -1,
        )
        return summary_cache[horizon]

    for model_type in args.models:
        encoder = make_summary_encoder(model_type=model_type, args=args, device=device)
        safe_feature_model = sanitize_name(model_type)
        safe_feature_backend = sanitize_name(encoder.backend)

        try:
            for horizon in summary_horizons:
                label = (
                    f"{safe_encoding_model}__{safe_feature_model}__{safe_feature_backend}__"
                    f"{safe_summary_model}__h{horizon}__{split_tag}"
                )
                out_path = out_dir / f"{label}.npz"
                meta = {
                    "encoding_model": args.encoding_model,
                    "feature_model": model_type,
                    "feature_backend": encoder.backend,
                    "summary_horizon": horizon,
                    "summary_model": summary_model,
                    "split_tag": split_tag,
                    "evaluation_split": (
                        "story_holdout" if test_stories else "bootstrap_cv"
                    ),
                }
                if pls_fit_rresp is not None:
                    meta["pls_supervision_mode"] = pls_supervision_mode
                    meta["pls_supervision_voxels"] = int(pls_fit_rresp.shape[1])
                label_to_meta[label] = meta

                if args.skip_existing and out_path.exists():
                    existing = np.load(out_path, allow_pickle=True)
                    all_corrs[label] = existing["corrs"]
                    log.info("Skipping existing condition %s", label)
                    continue

                log.info("=" * 60)
                log.info("Condition: %s", label)
                log.info("=" * 60)

                texts_by_story, summary_words = get_cached_summaries(horizon)

                log.info("Extracting %s summary features and building training design matrix...", model_type)
                train_rstim = build_design_matrix(
                    train_stories,
                    texts_by_story,
                    train_resp_lengths,
                    encoder,
                )
                if train_rstim.shape[0] != train_rresp.shape[0]:
                    raise ValueError(
                        f"Training design matrix rows ({train_rstim.shape[0]}) do not match "
                        f"training response rows ({train_rresp.shape[0]})."
                    )
                log.info("Training design matrix: %s (TRs x delayed features)", train_rstim.shape)

                if test_stories:
                    test_rstim = build_design_matrix(
                        test_stories,
                        texts_by_story,
                        test_resp_lengths,
                        encoder,
                    )
                    if test_rstim.shape[0] != test_rresp.shape[0]:
                        raise ValueError(
                            f"Test design matrix rows ({test_rstim.shape[0]}) do not match "
                            f"test response rows ({test_rresp.shape[0]})."
                        )
                    log.info("Held-out test design matrix: %s (TRs x delayed features)", test_rstim.shape)
                else:
                    test_rstim = None

                corrs, wt, cv_corrs, train_fit_corrs = fit_encoding_model(
                    args,
                    train_rstim,
                    train_rresp,
                    test_rstim,
                    test_rresp,
                    pls_fit_rresp=pls_fit_rresp,
                )
                del train_rstim, test_rstim

                all_corrs[label] = corrs
                log.info(
                    "  %s mean r=%.4f | max r=%.4f | n(r>0.1)=%d",
                    "Held-out test" if test_stories else "Bootstrap-CV",
                    corrs.mean(),
                    corrs.max(),
                    (corrs > 0.1).sum(),
                )
                if args.encoding_model == "ridge" and test_stories:
                    log.info(
                        "  Training bootstrap-CV mean r=%.4f | max r=%.4f | n(r>0.1)=%d",
                        cv_corrs.mean(),
                        cv_corrs.max(),
                        (cv_corrs > 0.1).sum(),
                    )
                elif args.encoding_model in {"lgbm", "mlp"}:
                    log.info(
                        "  Training fit mean r=%.4f | max r=%.4f | n(r>0.1)=%d",
                        train_fit_corrs.mean(),
                        train_fit_corrs.max(),
                        (train_fit_corrs > 0.1).sum(),
                    )

                save_dict = dict(
                    corrs=corrs,
                    cv_corrs=cv_corrs,
                    train_fit_corrs=(
                        train_fit_corrs if train_fit_corrs is not None else np.array([])
                    ),
                    voxels=vox,
                    encoding_model=np.array(args.encoding_model),
                    pls_n_components=np.array(args.pls_n_components),
                    pls_scale=np.array(args.pls_scale),
                    mlp_output_chunk_size=np.array(args.mlp_output_chunk_size),
                    mlp_hidden_dim=np.array(args.mlp_hidden_dim),
                    mlp_activation=np.array(args.mlp_activation),
                    mlp_alpha=np.array(args.mlp_alpha),
                    mlp_learning_rate_init=np.array(args.mlp_learning_rate_init),
                    mlp_batch_size=np.array(args.mlp_batch_size),
                    mlp_max_iter=np.array(args.mlp_max_iter),
                    mlp_early_stopping=np.array(args.mlp_early_stopping),
                    feature_model=np.array(model_type),
                    feature_backend=np.array(encoder.backend),
                    summary_horizon=np.array(horizon),
                    summary_model=np.array(summary_model),
                    embedding_model=np.array(args.embedding_model),
                    summary_words=np.array(summary_words),
                    stories=np.array(stories),
                    train_stories=np.array(train_stories),
                    test_stories=np.array(test_stories),
                    n_train_stories=np.array(len(train_stories)),
                    n_test_stories=np.array(len(test_stories)),
                    split_tag=np.array(split_tag),
                    evaluation_split=np.array(
                        "story_holdout" if test_stories else "bootstrap_cv"
                    ),
                    condition_label=np.array(label),
                )
                if args.save_weights and wt is not None:
                    save_dict["weights"] = wt

                np.savez(out_path, **save_dict)
                del wt
                log.info("  -> saved %s", out_path)
        finally:
            encoder.close()

    summary = {label: corr for label, corr in all_corrs.items()}
    summary["summary_horizons"] = np.array(summary_horizons)
    summary["summary_model"] = np.array(summary_model)
    summary["feature_models"] = np.array(args.models)
    summary["encoding_model"] = np.array(args.encoding_model)
    summary["pls_n_components"] = np.array(args.pls_n_components)
    summary["pls_scale"] = np.array(args.pls_scale)
    summary["mlp_output_chunk_size"] = np.array(args.mlp_output_chunk_size)
    summary["mlp_hidden_dim"] = np.array(args.mlp_hidden_dim)
    summary["mlp_activation"] = np.array(args.mlp_activation)
    summary["mlp_alpha"] = np.array(args.mlp_alpha)
    summary["mlp_learning_rate_init"] = np.array(args.mlp_learning_rate_init)
    summary["mlp_batch_size"] = np.array(args.mlp_batch_size)
    summary["mlp_max_iter"] = np.array(args.mlp_max_iter)
    summary["mlp_early_stopping"] = np.array(args.mlp_early_stopping)
    summary["embedding_model"] = np.array(args.embedding_model)
    summary["gpt2_model"] = np.array(args.gpt2_model)
    summary["voxels"] = vox
    summary["stories"] = np.array(stories)
    summary["train_stories"] = np.array(train_stories)
    summary["test_stories"] = np.array(test_stories)
    summary["split_tag"] = np.array(split_tag)
    summary["evaluation_split"] = np.array("story_holdout" if test_stories else "bootstrap_cv")
    np.savez(out_dir / "summary.npz", **summary)

    log.info("")
    log.info("=" * 60)
    log.info(
        "SUMMARY — per-voxel encoding correlation (%d voxels, %s)",
        len(vox),
        "held-out stories" if test_stories else "bootstrap CV",
    )
    log.info("=" * 60)
    for label in sorted(
        all_corrs,
        key=lambda item: (
            label_to_meta[item]["encoding_model"],
            label_to_meta[item]["feature_model"],
            label_to_meta[item]["summary_horizon"],
        ),
    ):
        corrs = all_corrs[label]
        log.info(
            "  %-8s %-18s h=%-5d mean=%.4f  max=%.4f  n(r>0.1)=%d",
            label_to_meta[label]["encoding_model"],
            label_to_meta[label]["feature_model"],
            label_to_meta[label]["summary_horizon"],
            corrs.mean(),
            corrs.max(),
            (corrs > 0.1).sum(),
        )

    if ba_subject_dir and os.path.isdir(ba_subject_dir):
        print_roi_summary(all_corrs, ba_subject_dir, vox, label_to_meta)

    log.info("All results saved to %s", out_dir)


if __name__ == "__main__":
    main()
