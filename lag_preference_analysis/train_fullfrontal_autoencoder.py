#!/usr/bin/env python3
"""Compare nonlinear autoencoders against PCA for full_frontal responses.

This is a reconstruction-only experiment. It asks whether the held-out
full_frontal response matrix has nonlinear low-dimensional structure that a
small autoencoder captures better than PCA at the same latent dimension.

By default, the train/validation stories are loaded from a previous combo
encoding run so the split matches the ridge results.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
import torch  # noqa: E402
from train_lag_encoding import (  # noqa: E402
    SUBJECT_TO_UTS,
    configure_data_root,
    load_full_frontal_voxels,
    load_stories,
    split_stories,
)
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fullfrontal_ae")


DEFAULT_SPLIT_TAG = "S1__embedding-summary-combo-h20-50-200__lags1-10__chunk1tr__seed0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="S1", choices=sorted(SUBJECT_TO_UTS))
    p.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    p.add_argument("--stories", nargs="+", default=None)
    p.add_argument("--val-story-count", type=int, default=8)
    p.add_argument("--val-stories", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--split-results-dir",
        default=str(THIS_DIR / "results" / DEFAULT_SPLIT_TAG),
        help="Existing run directory with lag_corrs.npz; train/val stories are reused if present.",
    )
    p.add_argument("--ba-dir", default=str(rse.LOCAL_DEFAULT_BA_DIR))
    p.add_argument("--data-root", default=None)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default="/Volumes/ellie/language-decoding-expts")
    p.add_argument("--local-cache-root", default=str(REPO_DIR / "local_compute_cache"))
    p.add_argument("--summaries-dir", default=str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))

    p.add_argument("--latent-dims", nargs="+", type=int, default=[128])
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--early-stop-frac", type=float, default=0.10)
    p.add_argument("--early-stop-min-trs", type=int, default=2000)
    p.add_argument("--pca-iterated-power", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--output-dir", default=str(THIS_DIR / "results" / "fullfrontal_autoencoder"))
    p.add_argument("--tag", default=None)
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(args: argparse.Namespace, stories: Sequence[str]) -> tuple[List[str], List[str], str]:
    split_dir = Path(args.split_results_dir).expanduser().resolve()
    split_npz = split_dir / "lag_corrs.npz"
    if split_npz.is_file():
        with np.load(split_npz, allow_pickle=True) as payload:
            if "train_stories" in payload.files and "val_stories" in payload.files:
                train = [str(x) for x in payload["train_stories"]]
                val = [str(x) for x in payload["val_stories"]]
                log.info("Using train/val split from %s", split_npz)
                return train, val, str(split_npz)

    train, val = split_stories(list(stories), args)
    log.info("Using newly generated story split.")
    return train, val, ""


def load_responses(args: argparse.Namespace, stories: Sequence[str], response_root: str) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    sample = get_resp(args.subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    total_voxels = int(sample.shape[1])
    voxels = load_full_frontal_voxels(args.subject, total_voxels, args.ba_dir)
    log.info("Full-frontal voxels: %d / %d", len(voxels), total_voxels)

    responses = get_resp(args.subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses = {story: arr.astype(np.float32, copy=False) for story, arr in responses.items()}
    return responses, voxels


def stack_stories(responses: Dict[str, np.ndarray], stories: Sequence[str]) -> np.ndarray:
    return np.vstack([responses[story] for story in stories]).astype(np.float32, copy=False)


def zscore_train_val(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        mean,
        std,
    )


def reconstruction_metrics(pred: np.ndarray, true: np.ndarray, voxel_chunk_size: int = 2000) -> dict:
    mse = float(np.mean((pred - true) ** 2))
    variance = float(np.mean(true ** 2))
    variance_explained = 1.0 - mse / variance

    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    true_centered = true - true.mean(axis=1, keepdims=True)
    denom = np.sqrt((pred_centered * pred_centered).sum(axis=1) * (true_centered * true_centered).sum(axis=1))
    pattern_r = np.divide(
        (pred_centered * true_centered).sum(axis=1),
        denom,
        out=np.zeros(pred.shape[0], dtype=np.float32),
        where=denom > 0,
    )

    voxel_rs = []
    for start in range(0, true.shape[1], voxel_chunk_size):
        p = pred[:, start : start + voxel_chunk_size]
        y = true[:, start : start + voxel_chunk_size]
        p = p - p.mean(axis=0)
        y = y - y.mean(axis=0)
        denom = np.sqrt((p * p).sum(axis=0) * (y * y).sum(axis=0))
        voxel_rs.append(np.divide((p * y).sum(axis=0), denom, out=np.zeros(p.shape[1]), where=denom > 0))
    voxel_r = np.concatenate(voxel_rs)

    return {
        "mse": mse,
        "variance_explained": float(variance_explained),
        "pattern_r_mean": float(pattern_r.mean()),
        "pattern_r_median": float(np.median(pattern_r)),
        "voxel_r_mean": float(voxel_r.mean()),
        "voxel_r_median": float(np.median(voxel_r)),
        "voxel_r_p95": float(np.quantile(voxel_r, 0.95)),
    }


class NonlinearAutoencoder(nn.Module):
    def __init__(self, n_voxels: int, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_voxels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_voxels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    latent_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, dict, int, float]:
    rng = np.random.default_rng(args.seed + latent_dim)
    perm = rng.permutation(x_train.shape[0])
    es_n = max(int(args.early_stop_min_trs), int(args.early_stop_frac * x_train.shape[0]))
    es_n = min(es_n, max(1, x_train.shape[0] - 1))
    es_idx = perm[:es_n]
    fit_idx = perm[es_n:]

    x_fit = torch.from_numpy(x_train[fit_idx])
    x_es = torch.from_numpy(x_train[es_idx])
    x_final = torch.from_numpy(x_val)

    model = NonlinearAutoencoder(
        n_voxels=x_train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_fit), batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        es_losses = []
        with torch.no_grad():
            for start in range(0, x_es.shape[0], max(1, int(args.batch_size) * 4)):
                xb = x_es[start : start + int(args.batch_size) * 4].to(device)
                es_losses.append(float(loss_fn(model(xb), xb).detach().cpu()))
        es_loss = float(np.mean(es_losses))

        if es_loss < best_loss - 1e-4:
            best_loss = es_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        log.info(
            "ae latent=%d epoch=%d train_mse=%.4f early_stop_mse=%.4f best_epoch=%d",
            latent_dim,
            epoch,
            float(np.mean(train_losses)),
            es_loss,
            best_epoch,
        )
        if bad_epochs >= int(args.patience) and epoch >= 10:
            break

    if best_state is None:
        raise RuntimeError("Autoencoder did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "latent_dim": latent_dim,
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "best_epoch": best_epoch,
                "best_early_stop_mse": best_loss,
            },
            checkpoint_dir / f"ae_latent{latent_dim}.pt",
        )

    preds = []
    with torch.no_grad():
        for start in range(0, x_final.shape[0], int(args.batch_size)):
            preds.append(model(x_final[start : start + int(args.batch_size)].to(device)).cpu().numpy())
    pred_val = np.vstack(preds).astype(np.float32)
    elapsed = time.time() - t0
    return pred_val, {"best_epoch": best_epoch, "best_early_stop_mse": best_loss}, best_epoch, elapsed


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mounted_root = configure_data_root(args)
    stories = load_stories(args)
    train_stories, val_stories, split_source = load_split(args, stories)
    log.info("Stories: %d train | %d val", len(train_stories), len(val_stories))
    log.info("Validation stories: %s", ", ".join(val_stories))

    response_root = config.DATA_TRAIN_DIR
    if args.local_compute_mode and mounted_root is not None:
        response_root = str(
            rse.stage_local_response_cache(
                args.subject,
                stories,
                Path(config.DATA_TRAIN_DIR),
                Path(args.local_cache_root).expanduser().resolve(),
            )
        )
        log.info("Using staged local response root: %s", response_root)

    t0 = time.time()
    responses, voxels = load_responses(args, stories, response_root)
    x_train = stack_stories(responses, train_stories)
    x_val = stack_stories(responses, val_stories)
    log.info("Loaded X_train=%s X_val=%s in %.1fs", x_train.shape, x_val.shape, time.time() - t0)

    x_train, x_val, train_mean, train_std = zscore_train_val(x_train, x_val)
    log.info("Z-scored responses with train-only mean/std.")

    tag = args.tag or f"{args.subject}__fullfrontal_ae_vs_pca__latent{'-'.join(map(str, args.latent_dims))}__seed{args.seed}"
    out_dir = Path(args.output_dir).expanduser().resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    device = resolve_device(args.device)
    log.info("Torch device: %s", device)

    rows = []
    for latent_dim in sorted({int(x) for x in args.latent_dims}):
        log.info("==== PCA latent=%d ====", latent_dim)
        t = time.time()
        pca = PCA(
            n_components=latent_dim,
            svd_solver="randomized",
            random_state=args.seed,
            iterated_power=int(args.pca_iterated_power),
        )
        pca.fit(x_train)
        pred = pca.inverse_transform(pca.transform(x_val)).astype(np.float32)
        metric = reconstruction_metrics(pred, x_val)
        rows.append({"model": "pca", "latent_dim": latent_dim, "elapsed_sec": time.time() - t, **metric})
        log.info(
            "pca latent=%d ve=%.4f pattern_r=%.4f voxel_r=%.4f",
            latent_dim,
            metric["variance_explained"],
            metric["pattern_r_mean"],
            metric["voxel_r_mean"],
        )

        log.info("==== nonlinear AE latent=%d ====", latent_dim)
        pred, ae_info, _, elapsed = train_autoencoder(x_train, x_val, latent_dim, args, device, checkpoint_dir)
        metric = reconstruction_metrics(pred, x_val)
        rows.append(
            {
                "model": "nonlinear_ae",
                "latent_dim": latent_dim,
                "elapsed_sec": elapsed,
                **metric,
                **ae_info,
            }
        )
        log.info(
            "ae latent=%d ve=%.4f pattern_r=%.4f voxel_r=%.4f best_epoch=%s",
            latent_dim,
            metric["variance_explained"],
            metric["pattern_r_mean"],
            metric["voxel_r_mean"],
            ae_info["best_epoch"],
        )

    csv_path = out_dir / "autoencoder_vs_pca.csv"
    fieldnames = [
        "model",
        "latent_dim",
        "elapsed_sec",
        "mse",
        "variance_explained",
        "pattern_r_mean",
        "pattern_r_median",
        "voxel_r_mean",
        "voxel_r_median",
        "voxel_r_p95",
        "best_epoch",
        "best_early_stop_mse",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject": args.subject,
                "stories": list(stories),
                "train_stories": train_stories,
                "val_stories": val_stories,
                "split_source": split_source,
                "n_voxels": int(voxels.size),
                "voxels": str(out_dir / "voxels.npy"),
                "data_train_dir": config.DATA_TRAIN_DIR,
                "response_root": response_root,
                "ba_dir": str(Path(args.ba_dir).expanduser().resolve()),
                "latent_dims": [int(x) for x in args.latent_dims],
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            },
            f,
            indent=2,
        )
    np.save(out_dir / "voxels.npy", voxels)
    np.savez(out_dir / "zscore_stats.npz", mean=train_mean, std=train_std)
    log.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
