#!/usr/bin/env python3
"""Train a masked 3D convolutional AE for full_frontal responses.

The response HDF5 columns are assumed to follow pycortex's ``mask_thick`` voxel
order: the position of each nonzero mask voxel in C-ravel order. This matches
the convention used by ``create_subj_BA_jsons.py`` and the existing pycortex
flatmap code.

The model reconstructs a cropped functional-space 3D volume, but the loss and
metrics are computed only on BA_full_frontal voxels. PCA is evaluated on the
same train/heldout split as a matched vector-space baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from train_fullfrontal_autoencoder import reconstruction_metrics, resolve_device  # noqa: E402
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
log = logging.getLogger("fullfrontal_volume_ae")


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

    p.add_argument("--pycortex-filestore", default=None)
    p.add_argument("--pycortex-subject", default=None)
    p.add_argument("--xfm-name", default=None)
    p.add_argument("--mask-type", default="thick")
    p.add_argument("--crop-pad", type=int, default=4)
    p.add_argument(
        "--pad-to-multiple",
        type=int,
        default=4,
        help="Pad cropped volume spatial dims to this multiple for stride-2 convs.",
    )

    p.add_argument("--latent-dims", nargs="+", type=int, default=[128])
    p.add_argument("--base-channels", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--input-noise-std", type=float, default=0.05)
    p.add_argument("--input-mask-prob", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--early-stop-frac", type=float, default=0.10)
    p.add_argument("--early-stop-min-trs", type=int, default=2000)
    p.add_argument("--pca-iterated-power", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--output-dir", default=str(THIS_DIR / "results" / "fullfrontal_volume_autoencoder"))
    p.add_argument("--tag", default=None)
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def configure_pycortex_filestore(explicit_path: str | None) -> str | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    candidates.append(REPO_DIR / "pycortex-db")
    candidates.append(Path.cwd() / "pycortex-db")

    chosen = next((path for path in candidates if path.is_dir()), None)
    if chosen is None:
        return None

    store = str(chosen)
    os.environ["PYCORTEX_FILESTORE"] = store
    import cortex

    try:
        from cortex.options import config as cortex_config
        if not cortex_config.has_section("basic"):
            cortex_config.add_section("basic")
        cortex_config.set("basic", "filestore", store)
    except Exception as err:
        log.warning("Could not update cortex.options.config: %s", err)

    try:
        cortex.db.filestore = store
        if hasattr(cortex.db, "_subjects"):
            cortex.db._subjects = None
    except Exception:
        pass
    try:
        cortex.db = cortex.database.Database()
        cortex.db.filestore = store
    except Exception:
        pass
    log.info("Pycortex filestore -> %s", getattr(cortex.db, "filestore", store))
    log.info("Pycortex subjects -> %s", sorted(cortex.db.subjects.keys()))
    return store


def list_subject_transforms(filestore: str | None, subject: str) -> list[str]:
    if not filestore:
        return []
    base = Path(filestore) / subject / "transforms"
    if not base.is_dir():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def resolve_pycortex_subject(args: argparse.Namespace, cortex_module) -> str:
    available = sorted(cortex_module.db.subjects.keys())
    if args.pycortex_subject:
        candidates = [args.pycortex_subject]
    else:
        candidates = [
            SUBJECT_TO_UTS.get(args.subject, ""),
            args.subject,
            f"sub-{SUBJECT_TO_UTS.get(args.subject, '')}",
            f"sub-{args.subject}",
        ]
    pycortex_subject = next((name for name in candidates if name and name in available), None)
    if pycortex_subject is None:
        raise SystemExit(
            f"pycortex does not know about any of {candidates!r}; available subjects: {available}"
        )
    return pycortex_subject


def find_matching_xfm(cortex_module, filestore: str | None, subject: str, n_total: int, preferred: str | None, mask_type: str) -> str:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)

    add(preferred)
    add("fullhead")
    add(f"{subject}_auto")
    for name in list_subject_transforms(filestore, subject):
        add(name)

    best: tuple[str, int] | None = None
    for name in candidates:
        try:
            mask = np.asarray(cortex_module.db.get_mask(subject, name, mask_type), dtype=bool)
        except Exception as err:
            log.debug("Could not load mask %s/%s/%s: %s", subject, name, mask_type, err)
            continue
        n_mask = int(mask.sum())
        log.info("Transform candidate %s/%s mask_%s voxels=%d", subject, name, mask_type, n_mask)
        if n_mask == int(n_total):
            return name
        if best is None or abs(n_mask - n_total) < abs(best[1] - n_total):
            best = (name, n_mask)

    if best is None:
        raise RuntimeError(f"No usable pycortex transform found for {subject}.")
    log.warning("No transform exactly matched n_total=%d; using closest %s (%d voxels).", n_total, best[0], best[1])
    return best[0]


def load_split(args: argparse.Namespace, stories: Sequence[str]) -> tuple[List[str], List[str], str]:
    split_npz = Path(args.split_results_dir).expanduser().resolve() / "lag_corrs.npz"
    if split_npz.is_file():
        with np.load(split_npz, allow_pickle=True) as payload:
            if "train_stories" in payload.files and "val_stories" in payload.files:
                return [str(x) for x in payload["train_stories"]], [str(x) for x in payload["val_stories"]], str(split_npz)
    train, val = split_stories(list(stories), args)
    return train, val, ""


def load_responses(args: argparse.Namespace, stories: Sequence[str], response_root: str) -> tuple[Dict[str, np.ndarray], np.ndarray, int]:
    sample = get_resp(args.subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    total_voxels = int(sample.shape[1])
    voxels = load_full_frontal_voxels(args.subject, total_voxels, args.ba_dir)
    responses = get_resp(args.subject, stories, stack=False, vox=voxels, response_root=response_root)
    responses = {story: arr.astype(np.float32, copy=False) for story, arr in responses.items()}
    return responses, voxels, total_voxels


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


def make_volume_mapper(mask_3d: np.ndarray, voxels: np.ndarray, pad: int, pad_to_multiple: int) -> dict:
    """Return crop/mask metadata mapping full_frontal columns to crop flat positions."""
    mask_coords = np.array(np.where(mask_3d), dtype=np.int64).T
    if mask_coords.shape[0] <= int(voxels.max()):
        raise ValueError(f"Voxel index {int(voxels.max())} outside pycortex mask with {mask_coords.shape[0]} voxels.")

    ff_coords = mask_coords[np.asarray(voxels, dtype=np.int64)]
    lo = np.maximum(ff_coords.min(axis=0) - int(pad), 0)
    hi = np.minimum(ff_coords.max(axis=0) + int(pad) + 1, np.asarray(mask_3d.shape))
    if pad_to_multiple > 1:
        size = hi - lo
        extra = (int(pad_to_multiple) - (size % int(pad_to_multiple))) % int(pad_to_multiple)
        hi = np.minimum(hi + extra, np.asarray(mask_3d.shape))

    crop_shape = tuple((hi - lo).astype(int).tolist())
    crop_coords = ff_coords - lo[None, :]
    crop_flat = np.ravel_multi_index(crop_coords.T, crop_shape).astype(np.int64)
    crop_mask = np.zeros(int(np.prod(crop_shape)), dtype=bool)
    crop_mask[crop_flat] = True
    log.info("3D crop lo=%s hi=%s shape=%s full_frontal=%d", lo.tolist(), hi.tolist(), crop_shape, int(crop_mask.sum()))
    return {
        "shape": crop_shape,
        "lo": lo.astype(int),
        "hi": hi.astype(int),
        "crop_flat": crop_flat,
        "crop_mask": crop_mask.reshape(crop_shape),
    }


class VolumeDataset(Dataset):
    def __init__(self, x: np.ndarray, crop_flat: np.ndarray, crop_shape: Sequence[int]) -> None:
        self.x = x
        self.crop_flat = torch.as_tensor(crop_flat, dtype=torch.long)
        self.crop_shape = tuple(int(v) for v in crop_shape)
        self.n_flat = int(np.prod(self.crop_shape))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        values = torch.from_numpy(self.x[idx])
        vol = torch.zeros(self.n_flat, dtype=torch.float32)
        vol[self.crop_flat] = values
        return vol.reshape(1, *self.crop_shape)


class Conv3DAutoencoder(nn.Module):
    def __init__(self, crop_shape: Sequence[int], latent_dim: int, base_channels: int, dropout: float) -> None:
        super().__init__()
        c = int(base_channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *tuple(int(v) for v in crop_shape))
            encoded = self.encoder_conv(dummy)
        self.encoded_shape = tuple(encoded.shape[1:])
        encoded_dim = int(np.prod(self.encoded_shape))
        self.to_latent = nn.Linear(encoded_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, encoded_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(c, 1, kernel_size=3, padding=1),
        )
        self.crop_shape = tuple(int(v) for v in crop_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_conv(x).flatten(1)
        latent = self.to_latent(z)
        y = self.from_latent(latent).reshape(x.shape[0], *self.encoded_shape)
        out = self.decoder_conv(y)
        return out[:, :, : self.crop_shape[0], : self.crop_shape[1], : self.crop_shape[2]]


def corrupt_batch(x: torch.Tensor, mask: torch.Tensor, noise_std: float, mask_prob: float) -> torch.Tensor:
    out = x
    if noise_std > 0:
        out = out + torch.randn_like(out) * float(noise_std) * mask
    if mask_prob > 0:
        keep = (torch.rand_like(out) >= float(mask_prob)).to(out.dtype)
        out = out * (keep * mask + (1.0 - mask))
    return out


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    return (diff * diff).sum() / (mask.sum() * pred.shape[0]).clamp_min(1.0)


def volume_pred_to_vector(pred_volumes: np.ndarray, crop_flat: np.ndarray) -> np.ndarray:
    flat = pred_volumes.reshape(pred_volumes.shape[0], -1)
    return flat[:, crop_flat].astype(np.float32, copy=False)


def train_conv_ae(
    x_train: np.ndarray,
    x_val: np.ndarray,
    mapper: dict,
    latent_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, dict, float]:
    rng = np.random.default_rng(args.seed + latent_dim)
    perm = rng.permutation(x_train.shape[0])
    es_n = max(int(args.early_stop_min_trs), int(args.early_stop_frac * x_train.shape[0]))
    es_n = min(es_n, max(1, x_train.shape[0] - 1))
    es_idx = perm[:es_n]
    fit_idx = perm[es_n:]

    crop_shape = mapper["shape"]
    crop_flat = mapper["crop_flat"]
    train_ds = VolumeDataset(x_train[fit_idx], crop_flat, crop_shape)
    es_ds = VolumeDataset(x_train[es_idx], crop_flat, crop_shape)
    val_ds = VolumeDataset(x_val, crop_flat, crop_shape)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    es_loader = DataLoader(es_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    mask = torch.from_numpy(mapper["crop_mask"].astype(np.float32))[None, None].to(device)
    model = Conv3DAutoencoder(crop_shape, latent_dim, int(args.base_channels), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for xb in train_loader:
            xb = xb.to(device)
            xb_corrupt = corrupt_batch(xb, mask, args.input_noise_std, args.input_mask_prob)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(xb_corrupt), xb, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        es_losses = []
        with torch.no_grad():
            for xb in es_loader:
                xb = xb.to(device)
                xb_corrupt = corrupt_batch(xb, mask, args.input_noise_std, args.input_mask_prob)
                es_losses.append(float(masked_loss(model(xb_corrupt), xb, mask).detach().cpu()))
        es_loss = float(np.mean(es_losses))

        if es_loss < best_loss - 1e-4:
            best_loss = es_loss
            best_epoch = epoch
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        log.info(
            "conv3d latent=%d epoch=%d train_mse=%.4f early_stop_mse=%.4f best_epoch=%d",
            latent_dim,
            epoch,
            float(np.mean(train_losses)),
            es_loss,
            best_epoch,
        )
        if bad_epochs >= int(args.patience) and epoch >= 10:
            break

    if best_state is None:
        raise RuntimeError("Conv3D AE did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "latent_dim": latent_dim,
                "base_channels": int(args.base_channels),
                "dropout": float(args.dropout),
                "best_epoch": best_epoch,
                "best_early_stop_mse": best_loss,
                "crop_shape": crop_shape,
            },
            checkpoint_dir / f"conv3d_ae_latent{latent_dim}.pt",
        )

    pred_vols = []
    with torch.no_grad():
        for xb in val_loader:
            pred_vols.append(model(xb.to(device)).cpu().numpy()[:, 0])
    pred_vec = volume_pred_to_vector(np.concatenate(pred_vols, axis=0), crop_flat)
    return pred_vec, {"best_epoch": best_epoch, "best_early_stop_mse": best_loss}, time.time() - t0


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

    responses, voxels, total_voxels = load_responses(args, stories, response_root)
    log.info("Full-frontal voxels: %d / %d", len(voxels), total_voxels)
    x_train = stack_stories(responses, train_stories)
    x_val = stack_stories(responses, val_stories)
    log.info("Loaded X_train=%s X_val=%s", x_train.shape, x_val.shape)
    x_train, x_val, train_mean, train_std = zscore_train_val(x_train, x_val)

    configure_pycortex_filestore(args.pycortex_filestore)
    import cortex

    pycortex_subject = resolve_pycortex_subject(args, cortex)
    filestore = os.environ.get("PYCORTEX_FILESTORE") or args.pycortex_filestore
    xfm_name = find_matching_xfm(cortex, filestore, pycortex_subject, total_voxels, args.xfm_name, args.mask_type)
    mask_3d = np.asarray(cortex.db.get_mask(pycortex_subject, xfm_name, args.mask_type), dtype=bool)
    if int(mask_3d.sum()) != int(total_voxels):
        raise ValueError(f"pycortex mask has {int(mask_3d.sum())} voxels but responses have {total_voxels}.")
    mapper = make_volume_mapper(mask_3d, voxels, args.crop_pad, args.pad_to_multiple)

    tag = args.tag or f"{args.subject}__fullfrontal_conv3d_ae_vs_pca__latent{'-'.join(map(str, args.latent_dims))}__seed{args.seed}"
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
        pca_pred = pca.inverse_transform(pca.transform(x_val)).astype(np.float32)
        metric = reconstruction_metrics(pca_pred, x_val)
        rows.append({"model": "pca", "latent_dim": latent_dim, "elapsed_sec": time.time() - t, **metric})
        log.info(
            "pca latent=%d ve=%.4f pattern_r=%.4f voxel_r=%.4f",
            latent_dim,
            metric["variance_explained"],
            metric["pattern_r_mean"],
            metric["voxel_r_mean"],
        )

        log.info("==== Conv3D denoising AE latent=%d ====", latent_dim)
        pred, info, elapsed = train_conv_ae(x_train, x_val, mapper, latent_dim, args, device, checkpoint_dir)
        metric = reconstruction_metrics(pred, x_val)
        rows.append({"model": "conv3d_denoising_ae", "latent_dim": latent_dim, "elapsed_sec": elapsed, **metric, **info})
        log.info(
            "conv3d ae latent=%d ve=%.4f pattern_r=%.4f voxel_r=%.4f best_epoch=%s",
            latent_dim,
            metric["variance_explained"],
            metric["pattern_r_mean"],
            metric["voxel_r_mean"],
            info["best_epoch"],
        )

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
    csv_path = out_dir / "conv3d_autoencoder_vs_pca.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    np.save(out_dir / "voxels.npy", voxels)
    np.savez(out_dir / "zscore_stats.npz", mean=train_mean, std=train_std)
    np.savez(
        out_dir / "volume_mapper.npz",
        crop_flat=mapper["crop_flat"],
        crop_mask=mapper["crop_mask"],
        crop_lo=mapper["lo"],
        crop_hi=mapper["hi"],
        mask_shape=np.asarray(mask_3d.shape, dtype=int),
    )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject": args.subject,
                "pycortex_subject": pycortex_subject,
                "xfm_name": xfm_name,
                "mask_type": args.mask_type,
                "split_source": split_source,
                "train_stories": train_stories,
                "val_stories": val_stories,
                "n_voxels": int(voxels.size),
                "total_voxels": int(total_voxels),
                "crop_shape": list(map(int, mapper["shape"])),
                "latent_dims": [int(x) for x in args.latent_dims],
                "base_channels": int(args.base_channels),
                "dropout": float(args.dropout),
                "input_noise_std": float(args.input_noise_std),
                "input_mask_prob": float(args.input_mask_prob),
                "data_train_dir": config.DATA_TRAIN_DIR,
                "response_root": response_root,
                "ba_dir": str(Path(args.ba_dir).expanduser().resolve()),
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
