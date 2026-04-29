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


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_frac))))
    return idx[n_val:], idx[:n_val]


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


def residual_noise_model(
    model: MindEyeEncoding,
    rstim: np.ndarray,
    responses_by_subject: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> Dict[str, np.ndarray]:
    out = {}
    for subject, resp in responses_by_subject.items():
        pred = predict_array(model, rstim, subject, device, batch_size, use_amp)
        residual = (resp - pred).astype(np.float32)
        cov = residual.T @ residual
        diag_mean = float(np.diag(cov).mean())
        if diag_mean <= 0:
            diag_mean = 1.0
        out[subject] = (cov / diag_mean).astype(np.float32)
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

    train_idx, val_idx = split_indices(rstim.shape[0], args.val_frac, args.seed)
    train_loaders = {}
    val_tensors = {}
    for subject in args.subjects:
        ds = TensorDataset(
            torch.from_numpy(rstim[train_idx]),
            torch.from_numpy(responses[subject][train_idx]),
        )
        train_loaders[subject] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        val_tensors[subject] = (rstim[val_idx], responses[subject][val_idx])

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
    best_val = float("inf")
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
            for subject, (xv, yv) in val_tensors.items():
                pred = predict_array(model, xv, subject, device, args.batch_size, use_amp)
                mse = float(np.mean((pred - yv) ** 2))
                corrs = corr_per_voxel(pred, yv)
                val_losses[subject] = mse
                val_corrs[subject] = corrs
        val_loss = float(np.mean(list(val_losses.values())))
        train_loss = float(np.mean(losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
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
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("Early stopping at epoch %d (best val=%.5f)", epoch, best_val)
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    log.info("Estimating residual noise models from training residuals ...")
    noise_model = residual_noise_model(model, rstim, responses, device, args.batch_size, use_amp)

    tag = args.tag or build_tag(args)
    out_dir = Path(args.output_dir).expanduser().resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"
    with open(history_path, "w", encoding="utf-8", newline="") as f:
        fields = sorted({k for row in history for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    summary = {
        subject: {
            "best_val_loss": float(best_val),
            "n_voxels": int(len(voxels[subject])),
            "last_val_mean_r": float(history[-1][f"val_mean_r_{subject}"]),
            "last_val_max_r": float(history[-1][f"val_max_r_{subject}"]),
        }
        for subject in args.subjects
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
        "feature_checkpoint": str(args.finetuned_checkpoint),
        "feature_layer": int(config.GPT_LAYER),
        "context_words": int(config.GPT_WORDS),
        "stim_delays": list(config.STIM_DELAYS),
        "sessions": list(args.sessions),
        "stories": np.array(stories),
        "voxels": voxels,
        "noise_model": noise_model,
        "tr_stats": np.array(tr_stats, dtype=object),
        "word_stats": np.array(word_stats, dtype=object),
        "best_val_loss": float(best_val),
        "args": vars(args),
        "total_train_seconds": float(time.time() - start_time),
    }
    torch.save(checkpoint, out_dir / "model.pt")
    log.info("Wrote %s", out_dir / "model.pt")
    log.info("Wrote %s", history_path)


if __name__ == "__main__":
    main()
