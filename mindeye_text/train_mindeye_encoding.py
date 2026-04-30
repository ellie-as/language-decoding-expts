#!/usr/bin/env python3
"""Train a MindEye-style neural encoding model.

This keeps the Huth/Tang text feature pipeline fixed:

* finetuned GPT-1 layer ``config.GPT_LAYER``
* short word window ``config.GPT_WORDS`` (current word + 5 previous words)
* ``utils_stim.get_stim`` TR interpolation, normalization, and FIR delays

The learned architecture is the encoding-direction analogue of
``mindeye_text/model.py``'s decoder:

``delayed text features -> shared text projection/backbone -> subject-specific voxel head``
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "gpt1_encoding_comparison"))

import config  # noqa: E402
from compare_gpt1_encoding import (  # noqa: E402
    DEFAULT_SESSIONS,
    HuthFinetunedGPT1Features,
    configure_data_root,
)
from model import MindEyeEncoding  # noqa: E402
from utils_resp import get_resp  # noqa: E402
from utils_ridge.ridge import ridge  # noqa: E402
from utils_stim import get_stim  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mindeye_encoding")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_SESSIONS)
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument(
        "--mounted-project-root",
        default="smb://ceph-gw02.hpc.swc.ucl.ac.uk/behrens/ellie/language-decoding-expts",
    )
    p.add_argument("--finetuned-checkpoint", default="perceived")
    p.add_argument(
        "--voxel-source",
        default="paper",
        choices=["paper"],
        help="Use Huth paper selected decoding voxels from models/<subject>/encoding_model_perceived.npz.",
    )
    p.add_argument(
        "--target-mode",
        default="response",
        choices=["response", "ridge_residual"],
        help=(
            "Predict responses directly, or predict residuals on top of a ridge "
            "baseline trained on the same split."
        ),
    )
    p.add_argument("--latent-dim", type=int, default=4096)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--input-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument(
        "--val-mode",
        default="story",
        choices=["story", "row"],
        help="Use held-out stories for validation, or random rows for quick debugging.",
    )
    p.add_argument(
        "--val-stories",
        nargs="+",
        default=None,
        help="Explicit held-out training stories for validation when --val-mode story.",
    )
    p.add_argument(
        "--val-story-count",
        type=int,
        default=8,
        help="Number of stories to hold out for validation when --val-stories is not set.",
    )
    p.add_argument(
        "--selection-metric",
        default="val_mean_r",
        choices=["val_mean_r", "val_loss"],
        help="Checkpoint/early-stop on heldout mean r or validation MSE.",
    )
    p.add_argument(
        "--skip-ridge-like-for-like",
        action="store_true",
        help="Skip the ridge baseline trained/scored on the same validation split.",
    )
    p.add_argument("--torch-device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=str(THIS_DIR / "encoding_results"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda":
        return torch.device("cuda")
    if pref == "mps":
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_stories(sessions: List[int]) -> List[str]:
    with open(Path(config.DATA_TRAIN_DIR) / "sess_to_story.json", encoding="utf-8") as f:
        sess_to_story = json.load(f)
    stories: List[str] = []
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])
    return stories


def load_paper_voxels(subject: str) -> np.ndarray:
    path = Path(config.MODEL_DIR) / subject / "encoding_model_perceived.npz"
    if not path.exists():
        raise FileNotFoundError(f"Paper encoding model not found: {path}")
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["voxels"], dtype=np.int64)


def load_paper_alphas(subject: str, voxels: np.ndarray) -> np.ndarray:
    path = Path(config.MODEL_DIR) / subject / "encoding_model_perceived.npz"
    if not path.exists():
        raise FileNotFoundError(f"Paper encoding model not found: {path}")
    data = np.load(path, allow_pickle=True)
    alphas = np.asarray(data["alphas"], dtype=np.float32)
    return alphas[np.asarray(voxels, dtype=np.int64)]


def load_paper_weights(subject: str, voxels: np.ndarray) -> np.ndarray:
    path = Path(config.MODEL_DIR) / subject / "encoding_model_perceived.npz"
    if not path.exists():
        raise FileNotFoundError(f"Paper encoding model not found: {path}")
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["weights"][:, np.asarray(voxels, dtype=np.int64)], dtype=np.float32)


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_frac))))
    return idx[n_val:], idx[:n_val]


def split_stories(stories: List[str], args: argparse.Namespace) -> tuple[List[str], List[str]]:
    if args.val_mode == "row":
        return list(stories), []
    if args.val_stories:
        val = [story for story in args.val_stories if story in stories]
        missing = sorted(set(args.val_stories) - set(val))
        if missing:
            raise ValueError(f"--val-stories not found in training story list: {missing}")
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = list(stories)
        rng.shuffle(shuffled)
        val = sorted(shuffled[:max(1, int(args.val_story_count))])
    val_set = set(val)
    train = [story for story in stories if story not in val_set]
    if not train:
        raise ValueError("Story validation split left no training stories.")
    return train, val


def story_slices(stories: List[str], responses_by_story: Dict[str, np.ndarray]) -> Dict[str, slice]:
    cursor = 0
    out = {}
    for story in stories:
        n = int(responses_by_story[story].shape[0])
        out[story] = slice(cursor, cursor + n)
        cursor += n
    return out


def indices_for_stories(stories: List[str], slices: Dict[str, slice]) -> np.ndarray:
    parts = [np.arange(slices[story].start, slices[story].stop, dtype=np.int64) for story in stories]
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def corr_per_voxel(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred * pred).sum(axis=0) * (true * true).sum(axis=0))
    corr = np.divide((pred * true).sum(axis=0), denom, out=np.zeros(pred.shape[1]), where=denom != 0)
    return np.nan_to_num(corr).astype(np.float32)


@torch.no_grad()
def predict_array(
    model: MindEyeEncoding,
    x: np.ndarray,
    subject: str,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).float().to(device)
        with autocast_ctx(device, use_amp):
            pred = model(xb, subject)
        outs.append(pred.detach().to("cpu").float().numpy())
    return np.vstack(outs).astype(np.float32)


def predict_encoding_array(
    model: MindEyeEncoding,
    x: np.ndarray,
    subject: str,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    base_weights: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    pred = predict_array(model, x, subject, device, batch_size, use_amp)
    if base_weights is not None:
        pred = pred + x.dot(base_weights[subject]).astype(np.float32)
    return pred.astype(np.float32, copy=False)


def residual_noise_model(
    model: MindEyeEncoding,
    rstim: np.ndarray,
    responses_by_subject: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    base_weights: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    out = {}
    for subject, resp in responses_by_subject.items():
        pred = predict_encoding_array(
            model,
            rstim,
            subject,
            device,
            batch_size,
            use_amp,
            base_weights=base_weights,
        )
        residual = (resp - pred).astype(np.float32)
        cov = residual.T @ residual
        diag_mean = float(np.diag(cov).mean())
        if diag_mean <= 0:
            diag_mean = 1.0
        out[subject] = (cov / diag_mean).astype(np.float32)
    return out


def ridge_like_for_like(
    rstim: np.ndarray,
    responses: Dict[str, np.ndarray],
    voxels: Dict[str, np.ndarray],
    subjects: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    weights_by_subject: Dict[str, np.ndarray] | None = None,
) -> Dict[str, dict]:
    """Train ridge on the neural train split and evaluate on the same val split."""
    out = {}
    vals = []
    x_train = rstim[train_idx]
    x_val = rstim[val_idx]
    for subject in subjects:
        if weights_by_subject is None:
            alphas = load_paper_alphas(subject, voxels[subject])
            weights = ridge(x_train, responses[subject][train_idx], alphas)
        else:
            weights = weights_by_subject[subject]
        pred = x_val.dot(weights).astype(np.float32)
        corrs = corr_per_voxel(pred, responses[subject][val_idx])
        mean_r = float(corrs.mean())
        out[subject] = {
            "ridge_like_for_like_mean_r": mean_r,
            "ridge_like_for_like_median_r": float(np.median(corrs)),
            "ridge_like_for_like_max_r": float(corrs.max()),
        }
        vals.append(mean_r)
        del pred, corrs
    out["ALL"] = {
        "ridge_like_for_like_mean_r": float(np.mean(vals)) if vals else float("nan"),
    }
    return out


def fit_split_ridge_weights(
    rstim: np.ndarray,
    responses: Dict[str, np.ndarray],
    voxels: Dict[str, np.ndarray],
    subjects: List[str],
    train_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    x_train = rstim[train_idx]
    out = {}
    for subject in subjects:
        alphas = load_paper_alphas(subject, voxels[subject])
        out[subject] = ridge(x_train, responses[subject][train_idx], alphas).astype(np.float32)
    return out


def build_tag(args: argparse.Namespace) -> str:
    subj = "-".join(args.subjects)
    return (
        f"mindeye_encoding__{subj}__latent{args.latent_dim}-blocks{args.n_blocks}"
        f"__lr{args.lr:g}-wd{args.weight_decay:g}__seed{args.seed}"
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root
    if args.local_compute_mode and data_root is None:
        data_root = args.mounted_project_root
    configure_data_root(data_root)

    device = resolve_device(args.torch_device)
    config.GPT_DEVICE = str(device)
    use_amp = bool(args.amp) and device.type == "cuda"
    log.info("Device: %s | AMP: %s", device, "bf16" if use_amp else "off")

    stories = load_stories(args.sessions)
    log.info("Stories: %d", len(stories))

    features = HuthFinetunedGPT1Features(args.finetuned_checkpoint, str(device))
    try:
        log.info("Extracting Huth-style delayed GPT-1 stimulus features ...")
        rstim, tr_stats, word_stats = get_stim(stories, features)
    finally:
        features.close()
    rstim = rstim.astype(np.float32, copy=False)
    log.info("Stimulus matrix: %s", rstim.shape)

    voxels = {subject: load_paper_voxels(subject) for subject in args.subjects}
    responses = {
        subject: get_resp(subject, stories, stack=True, vox=voxels[subject]).astype(np.float32)
        for subject in args.subjects
    }
    output_dims = {subject: responses[subject].shape[1] for subject in args.subjects}
    for subject in args.subjects:
        log.info("[%s] selected voxels=%d response=%s", subject, len(voxels[subject]), responses[subject].shape)

    first_subject = args.subjects[0]
    first_responses_by_story = get_resp(
        first_subject,
        stories,
        stack=False,
        vox=voxels[first_subject],
    )
    slices = story_slices(stories, first_responses_by_story)
    train_stories, val_stories = split_stories(stories, args)
    if args.val_mode == "story":
        train_idx = indices_for_stories(train_stories, slices)
        val_idx = indices_for_stories(val_stories, slices)
        log.info("Validation stories (%d): %s", len(val_stories), ", ".join(val_stories))
    else:
        train_idx, val_idx = split_indices(rstim.shape[0], args.val_frac, args.seed)
        train_stories = list(stories)
        val_stories = []
        log.info("Validation mode: random rows (%d train, %d val)", len(train_idx), len(val_idx))
    if max(train_idx.max(initial=0), val_idx.max(initial=0)) >= rstim.shape[0]:
        raise RuntimeError(
            "Story-derived response row indices exceed stimulus rows. "
            f"max index={max(train_idx.max(initial=0), val_idx.max(initial=0))}, "
            f"stim rows={rstim.shape[0]}"
        )
    if args.skip_ridge_like_for_like:
        ridge_lfl = {}
        split_ridge_weights = {}
        log.info("Skipping like-for-like ridge validation baseline.")
    else:
        log.info("Fitting like-for-like ridge baseline on the same validation split ...")
        split_ridge_weights = fit_split_ridge_weights(
            rstim,
            responses,
            voxels,
            args.subjects,
            train_idx,
        )
        ridge_lfl = ridge_like_for_like(
            rstim,
            responses,
            voxels,
            args.subjects,
            train_idx,
            val_idx,
            weights_by_subject=split_ridge_weights,
        )
        log.info(
            "Like-for-like ridge heldout mean r: %s",
            " ".join(
                f"{subject}={ridge_lfl[subject]['ridge_like_for_like_mean_r']:.4f}"
                for subject in args.subjects
            ),
        )
    full_paper_weights = {
        subject: load_paper_weights(subject, voxels[subject])
        for subject in args.subjects
    }
    if args.target_mode == "ridge_residual" and not split_ridge_weights:
        raise ValueError("--target-mode ridge_residual requires the like-for-like ridge baseline.")
    train_targets = {}
    val_targets = {}
    val_base_weights = None
    if args.target_mode == "ridge_residual":
        val_base_weights = split_ridge_weights
        for subject in args.subjects:
            base_pred = rstim.dot(split_ridge_weights[subject]).astype(np.float32)
            residual_target = (responses[subject] - base_pred).astype(np.float32)
            train_targets[subject] = residual_target
            val_targets[subject] = residual_target
            del base_pred, residual_target
        log.info("Training neural model to predict ridge residuals.")
    else:
        for subject in args.subjects:
            train_targets[subject] = responses[subject]
            val_targets[subject] = responses[subject]
        log.info("Training neural model to predict responses directly.")

    train_loaders = {}
    val_tensors = {}
    for subject in args.subjects:
        ds = TensorDataset(
            torch.from_numpy(rstim[train_idx]),
            torch.from_numpy(train_targets[subject][train_idx]),
        )
        train_loaders[subject] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        val_tensors[subject] = (
            rstim[val_idx],
            val_targets[subject][val_idx],
            responses[subject][val_idx],
        )

    model = MindEyeEncoding(
        output_dims=output_dims,
        input_dim=rstim.shape[1],
        latent_dim=args.latent_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        input_norm=bool(args.input_norm),
        head_norm=bool(args.head_norm),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %.1fM trainable params", n_params / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = min(len(loader) for loader in train_loaders.values())
    if n_steps == 0:
        raise RuntimeError("At least one train loader is empty; lower --batch-size.")

    best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    best_val = -float("inf") if args.selection_metric == "val_mean_r" else float("inf")
    bad_epochs = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        iters = {subject: iter(loader) for subject, loader in train_loaders.items()}
        losses = []
        for _step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for subject, iterator in iters.items():
                xb, yb = next(iterator)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast_ctx(device, use_amp):
                    pred = model(xb, subject)
                    loss = torch.nn.functional.mse_loss(pred, yb)
                total_loss = total_loss + loss
            total_loss = total_loss / len(args.subjects)
            total_loss.backward()
            opt.step()
            losses.append(float(total_loss.detach().to("cpu")))

        model.eval()
        val_losses = {}
        val_corrs = {}
        with torch.no_grad():
            for subject, (xv, yv_target, yv_response) in val_tensors.items():
                pred_target = predict_array(model, xv, subject, device, args.batch_size, use_amp)
                mse = float(np.mean((pred_target - yv_target) ** 2))
                if args.target_mode == "ridge_residual":
                    pred_response = pred_target + xv.dot(val_base_weights[subject]).astype(np.float32)
                else:
                    pred_response = pred_target
                corrs = corr_per_voxel(pred_response, yv_response)
                val_losses[subject] = mse
                val_corrs[subject] = corrs
        val_loss = float(np.mean(list(val_losses.values())))
        train_loss = float(np.mean(losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_r_all": float(np.mean([val_corrs[s].mean() for s in args.subjects])),
            "val_max_r_all": float(np.mean([val_corrs[s].max() for s in args.subjects])),
            **{f"val_loss_{s}": v for s, v in val_losses.items()},
            **{f"val_mean_r_{s}": float(val_corrs[s].mean()) for s in args.subjects},
            **{f"val_selected_mean_r_{s}": float(val_corrs[s].mean()) for s in args.subjects},
            **{f"val_max_r_{s}": float(val_corrs[s].max()) for s in args.subjects},
        }
        history.append(row)
        log.info(
            "epoch %03d train=%.5f val=%.5f | %s",
            epoch,
            train_loss,
            val_loss,
            " ".join(f"{s}:r={val_corrs[s].mean():.4f}/max={val_corrs[s].max():.3f}" for s in args.subjects),
        )
        selection_value = row["val_mean_r_all"] if args.selection_metric == "val_mean_r" else val_loss
        improved = (
            selection_value > best_val + 1e-6
            if args.selection_metric == "val_mean_r"
            else selection_value < best_val - 1e-6
        )
        if improved:
            best_val = selection_value
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info(
                    "Early stopping at epoch %d (best %s=%.5f)",
                    epoch,
                    args.selection_metric,
                    best_val,
                )
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    log.info("Estimating residual noise models from training residuals ...")
    decode_base_weights = full_paper_weights if args.target_mode == "ridge_residual" else None
    noise_model = residual_noise_model(
        model,
        rstim,
        responses,
        device,
        args.batch_size,
        use_amp,
        base_weights=decode_base_weights,
    )

    tag = args.tag or build_tag(args)
    out_dir = Path(args.output_dir).expanduser().resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"
    with open(history_path, "w", encoding="utf-8", newline="") as f:
        fields = sorted({k for row in history for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    best_loss_row = min(history, key=lambda row: row["val_loss"])
    best_r_row = max(history, key=lambda row: row["val_mean_r_all"])
    summary = {
        subject: {
            "best_selection_value": float(best_val),
            "n_voxels": int(len(voxels[subject])),
            **ridge_lfl.get(subject, {}),
            "best_loss_epoch_val_mean_r": float(best_loss_row[f"val_mean_r_{subject}"]),
            "best_r_epoch_val_mean_r": float(best_r_row[f"val_mean_r_{subject}"]),
            "delta_best_r_minus_ridge_like_for_like": (
                float(best_r_row[f"val_mean_r_{subject}"])
                - float(ridge_lfl[subject]["ridge_like_for_like_mean_r"])
                if subject in ridge_lfl
                else ""
            ),
            "best_r_epoch_val_max_r": float(best_r_row[f"val_max_r_{subject}"]),
            "last_val_mean_r": float(history[-1][f"val_mean_r_{subject}"]),
        }
        for subject in args.subjects
    }
    summary["ALL"] = {
        "best_selection_value": float(best_val),
        "selection_metric": str(args.selection_metric),
        "best_loss_epoch": int(best_loss_row["epoch"]),
        "best_loss_epoch_val_mean_r": float(best_loss_row["val_mean_r_all"]),
        "best_r_epoch": int(best_r_row["epoch"]),
        "best_r_epoch_val_mean_r": float(best_r_row["val_mean_r_all"]),
        "best_r_epoch_val_max_r": float(best_r_row["val_max_r_all"]),
        "last_val_mean_r": float(history[-1]["val_mean_r_all"]),
        "val_mode": str(args.val_mode),
        "val_stories": list(val_stories),
        "target_mode": str(args.target_mode),
        **ridge_lfl.get("ALL", {}),
        "delta_best_r_minus_ridge_like_for_like": (
            float(best_r_row["val_mean_r_all"])
            - float(ridge_lfl["ALL"]["ridge_like_for_like_mean_r"])
            if "ALL" in ridge_lfl
            else ""
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    checkpoint = {
        "model_state_dict": {k: v for k, v in best_state.items()},
        "subjects": list(args.subjects),
        "input_dim": int(rstim.shape[1]),
        "output_dims": output_dims,
        "latent_dim": int(args.latent_dim),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),
        "input_norm": bool(args.input_norm),
        "head_norm": bool(args.head_norm),
        "target_mode": str(args.target_mode),
        "feature_checkpoint": str(args.finetuned_checkpoint),
        "feature_layer": int(config.GPT_LAYER),
        "context_words": int(config.GPT_WORDS),
        "stim_delays": list(config.STIM_DELAYS),
        "sessions": list(args.sessions),
        "stories": np.array(stories),
        "train_stories": np.array(train_stories),
        "val_stories": np.array(val_stories),
        "val_mode": str(args.val_mode),
        "voxels": voxels,
        "base_weights": full_paper_weights if args.target_mode == "ridge_residual" else {},
        "noise_model": noise_model,
        "tr_stats": np.array(tr_stats, dtype=object),
        "word_stats": np.array(word_stats, dtype=object),
        "selection_metric": str(args.selection_metric),
        "best_selection_value": float(best_val),
        "args": vars(args),
        "total_train_seconds": float(time.time() - start_time),
    }
    torch.save(checkpoint, out_dir / "model.pt")
    log.info("Wrote %s", out_dir / "model.pt")
    log.info("Wrote %s", history_path)


if __name__ == "__main__":
    main()
