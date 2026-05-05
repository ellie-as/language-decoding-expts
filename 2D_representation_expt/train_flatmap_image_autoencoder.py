#!/usr/bin/env python3
"""Train a 2D convolutional AE on per-TR pycortex flatmap images.

Why bother with this view at all? mask_thick voxels are a folded 3D ribbon, but
cortex itself is approximately 2D. Pycortex's flatmap projection unrolls the
ribbon into a single 2D pixel grid. Treating each TR as a 2D image lets a conv
network exploit cortical adjacency (clusters near each other on the surface
share weights) and avoids the sulcus-bridging problems of 3D conv kernels.

Pipeline:
  1. Load mask_thick BOLD responses for one subject across all stories.
  2. Train/val story split (reused from a prior run if --split-results-dir
     points to one).
  3. Z-score per voxel using train mean/std.
  4. For each TR, project the voxel vector onto a flatmap pixel grid using
     pycortex (``cortex.quickflat.utils.make_flatmap_image``) and stack into a
     ``(T, H, W)`` tensor. Pixels outside cortex are NaN-masked.
  5. Cache the rendered images on disk so re-runs avoid re-rendering.
  6. Train a 2D conv denoising autoencoder with masked MSE (loss only on cortex
     pixels). Compare against a PCA baseline at the same latent dims, fit on
     the masked pixel vector.
  7. Save reconstruction metrics, sample PNGs, optional checkpoints.

Run:
    python 2D_representation_expt/train_flatmap_image_autoencoder.py \\
        --subject S1 --image-height 192 --latent-dims 128 \\
        --pycortex-filestore <repo>/pycortex-db
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
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
LAG_DIR = REPO_DIR / "lag_preference_analysis"

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "mindeye_text"))
sys.path.insert(0, str(LAG_DIR))
sys.path.insert(0, str(THIS_DIR))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from train_fullfrontal_autoencoder import reconstruction_metrics, resolve_device  # noqa: E402
from train_fullfrontal_volume_autoencoder import (  # noqa: E402
    configure_pycortex_filestore,
    find_matching_xfm,
    resolve_pycortex_subject,
)
from train_lag_encoding import (  # noqa: E402
    SUBJECT_TO_UTS,
    configure_data_root,
    load_stories,
    split_stories,
)
from utils_resp import get_resp  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("flatmap_image_ae")


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
        default=str(LAG_DIR / "results" / DEFAULT_SPLIT_TAG),
        help="Existing run directory with lag_corrs.npz; train/val stories reused if present.",
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
    p.add_argument("--image-height", type=int, default=192,
                   help="Flatmap pixel height; width is auto-inferred from pycortex extents.")
    p.add_argument("--pad-to-multiple", type=int, default=8,
                   help="Pad H and W up to this multiple (must be a power of 2 covering all stride-2 convs).")
    p.add_argument("--with-curvature", action="store_true",
                   help="Use pycortex curvature underlay when rendering reference images. The training tensor itself is always the projected response value, never blended with curvature.")

    p.add_argument(
        "--image-cache-dir",
        default=None,
        help="Directory for cached flatmap images (default: <output-dir>/flatmap_image_cache).",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild of cached flatmap images even if a matching cache exists.",
    )
    p.add_argument(
        "--render-batch-log-every",
        type=int,
        default=200,
        help="Log progress every N TRs while rendering.",
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

    p.add_argument(
        "--sample-png-count",
        type=int,
        default=8,
        help="Number of validation TRs to save as input/reconstruction PNG comparisons after training finishes.",
    )
    p.add_argument(
        "--epoch-png-count",
        type=int,
        default=4,
        help="If >0, save this many validation input/reconstruction pairs to <out>/training_progress/latent<dim>/epoch<NN>/ at the end of every epoch (set 0 to disable).",
    )
    p.add_argument(
        "--epoch-png-stride",
        type=int,
        default=1,
        help="Save epoch PNGs every N epochs (1 = every epoch).",
    )

    p.add_argument("--output-dir", default=str(THIS_DIR / "results" / "flatmap_image_autoencoder"))
    p.add_argument("--tag", default=None)
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def load_split(args: argparse.Namespace, stories: Sequence[str]) -> tuple[list[str], list[str], str]:
    split_npz = Path(args.split_results_dir).expanduser().resolve() / "lag_corrs.npz"
    if split_npz.is_file():
        with np.load(split_npz, allow_pickle=True) as payload:
            if "train_stories" in payload.files and "val_stories" in payload.files:
                return [str(x) for x in payload["train_stories"]], [str(x) for x in payload["val_stories"]], str(split_npz)
    train, val = split_stories(list(stories), args)
    return train, val, ""


def load_responses(args: argparse.Namespace, stories: Sequence[str], response_root: str) -> tuple[Dict[str, np.ndarray], int]:
    sample = get_resp(args.subject, [stories[0]], stack=True, vox=None, response_root=response_root)
    total_voxels = int(sample.shape[1])
    responses = get_resp(args.subject, stories, stack=False, vox=None, response_root=response_root)
    responses = {story: arr.astype(np.float32, copy=False) for story, arr in responses.items()}
    return responses, total_voxels


def stack_stories(responses: Dict[str, np.ndarray], stories: Sequence[str]) -> np.ndarray:
    return np.vstack([responses[story] for story in stories]).astype(np.float32, copy=False)


def zscore_train_val(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """In-place z-score (mutates x_train and x_val) to keep memory low for huge BOLD arrays."""
    if x_train.dtype != np.float32:
        x_train = x_train.astype(np.float32, copy=False)
    if x_val.dtype != np.float32:
        x_val = x_val.astype(np.float32, copy=False)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    np.subtract(x_train, mean, out=x_train)
    np.divide(x_train, std, out=x_train)
    np.subtract(x_val, mean, out=x_val)
    np.divide(x_val, std, out=x_val)
    return x_train, x_val, mean, std


def cache_paths(cache_dir: Path, subject: str, xfm_name: str, mask_type: str, image_height: int) -> dict[str, Path]:
    base = cache_dir / f"{subject}__{xfm_name}__{mask_type}__h{int(image_height)}"
    return {
        "train_images": base.with_name(base.name + "__train_images.npy"),
        "val_images": base.with_name(base.name + "__val_images.npy"),
        "pixel_mask": base.with_name(base.name + "__pixel_mask.npy"),
        "meta": base.with_name(base.name + "__meta.json"),
    }


def render_flatmap_images(
    voxel_data: np.ndarray,
    *,
    pycortex_subject: str,
    xfm_name: str,
    image_height: int,
    log_every: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Render a (T, H, W) flatmap stack from (T, V) voxel data.

    Pixels outside the cortical sheet remain NaN; the corresponding boolean
    pixel_mask is also returned. ``extents`` is what pycortex returned.
    """
    import cortex
    from cortex.quickflat.utils import make_flatmap_image  # noqa: WPS433

    n_t, n_v = voxel_data.shape
    log.info("Rendering %s: %d TRs at height=%d", label, n_t, int(image_height))
    sample_volume = cortex.Volume(
        voxel_data[0].astype(np.float32),
        pycortex_subject,
        xfm_name,
    )
    sample_image, extents = make_flatmap_image(sample_volume, height=int(image_height))
    sample_image = np.asarray(sample_image, dtype=np.float32)
    pixel_mask = np.isfinite(sample_image)
    height, width = sample_image.shape
    images = np.full((n_t, height, width), np.nan, dtype=np.float32)
    images[0] = sample_image

    t0 = time.time()
    for i in range(1, n_t):
        vol = cortex.Volume(voxel_data[i].astype(np.float32), pycortex_subject, xfm_name)
        img, _ = make_flatmap_image(vol, height=int(image_height))
        images[i] = np.asarray(img, dtype=np.float32)
        if log_every > 0 and (i + 1) % int(log_every) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(1e-6, elapsed)
            remaining = (n_t - i - 1) / max(1e-6, rate)
            log.info(
                "%s: rendered %d/%d (%.2f TR/s, ~%.0fs remaining)",
                label,
                i + 1,
                n_t,
                rate,
                remaining,
            )
    log.info("%s: rendered %d frames in %.1fs", label, n_t, time.time() - t0)
    return images, pixel_mask, extents


def pad_amount(image_hw: tuple[int, int], multiple: int) -> tuple[int, int]:
    if multiple <= 1:
        return 0, 0
    h, w = image_hw
    return int((-h) % int(multiple)), int((-w) % int(multiple))


def pad_image_stack(images: np.ndarray, pad_hw: tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return images
    if images.ndim == 3:
        spec = ((0, 0), (0, pad_h), (0, pad_w))
    elif images.ndim == 2:
        spec = ((0, pad_h), (0, pad_w))
    else:
        raise ValueError(f"Cannot pad ndim={images.ndim}")
    return np.pad(images, spec, mode="constant", constant_values=np.nan)


def pad_mask(mask: np.ndarray, pad_hw: tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return mask
    return np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)


def fill_nan_inplace(images: np.ndarray, mask: np.ndarray) -> None:
    flat = images.reshape(images.shape[0], -1)
    flat[:, ~mask.reshape(-1)] = 0.0
    flat[~np.isfinite(flat)] = 0.0


def masked_pixel_vector(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)
    return flat[:, mask.reshape(-1)].astype(np.float32, copy=False)


def vector_to_image(vector: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flat = np.zeros((vector.shape[0], mask.size), dtype=np.float32)
    flat[:, mask.reshape(-1)] = vector
    return flat.reshape(vector.shape[0], *mask.shape)


class FlatmapDataset(Dataset):
    def __init__(self, images: np.ndarray) -> None:
        self.images = images

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = torch.from_numpy(self.images[idx])
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x


class Conv2DAutoencoder(nn.Module):
    def __init__(self, image_shape: Sequence[int], latent_dim: int, base_channels: int, dropout: float) -> None:
        super().__init__()
        c = int(base_channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, int(image_shape[0]), int(image_shape[1]))
            encoded = self.encoder_conv(dummy)
        self.encoded_shape = tuple(encoded.shape[1:])
        encoded_dim = int(np.prod(self.encoded_shape))
        self.to_latent = nn.Linear(encoded_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, encoded_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
        )
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_conv(x).flatten(1)
        latent = self.to_latent(z)
        y = self.from_latent(latent).reshape(x.shape[0], *self.encoded_shape)
        out = self.decoder_conv(y)
        return out[:, :, : self.image_shape[0], : self.image_shape[1]]


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


def train_conv_ae(
    images_train: np.ndarray,
    images_val: np.ndarray,
    mask: np.ndarray,
    latent_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
    epoch_sample_dir: Path | None = None,
) -> tuple[np.ndarray, dict, float]:
    rng = np.random.default_rng(args.seed + latent_dim)
    perm = rng.permutation(images_train.shape[0])
    es_n = max(int(args.early_stop_min_trs), int(args.early_stop_frac * images_train.shape[0]))
    es_n = min(es_n, max(1, images_train.shape[0] - 1))
    es_idx = perm[:es_n]
    fit_idx = perm[es_n:]

    image_shape = images_train.shape[1:]
    train_ds = FlatmapDataset(images_train[fit_idx])
    es_ds = FlatmapDataset(images_train[es_idx])
    val_ds = FlatmapDataset(images_val)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    es_loader = DataLoader(es_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    pixel_mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    model = Conv2DAutoencoder(image_shape, latent_dim, int(args.base_channels), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    epoch_sample_count = int(getattr(args, "epoch_png_count", 0))
    epoch_sample_stride = max(1, int(getattr(args, "epoch_png_stride", 1)))
    epoch_sample_targets: torch.Tensor | None = None
    if epoch_sample_dir is not None and epoch_sample_count > 0 and images_val.shape[0] > 0:
        epoch_sample_dir.mkdir(parents=True, exist_ok=True)
        n_take = min(epoch_sample_count, images_val.shape[0])
        sample_indices = np.linspace(0, images_val.shape[0] - 1, num=n_take, dtype=int)
        epoch_sample_targets = (
            torch.from_numpy(images_val[sample_indices])
            .unsqueeze(1)
            .to(device)
        )
        sample_indices_path = epoch_sample_dir / "sample_indices.json"
        if not sample_indices_path.is_file():
            with open(sample_indices_path, "w", encoding="utf-8") as f:
                json.dump({"val_indices": [int(i) for i in sample_indices]}, f)

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
            xb_corrupt = corrupt_batch(xb, pixel_mask_t, args.input_noise_std, args.input_mask_prob)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(xb_corrupt), xb, pixel_mask_t)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        es_losses = []
        with torch.no_grad():
            for xb in es_loader:
                xb = xb.to(device)
                xb_corrupt = corrupt_batch(xb, pixel_mask_t, args.input_noise_std, args.input_mask_prob)
                es_losses.append(float(masked_loss(model(xb_corrupt), xb, pixel_mask_t).detach().cpu()))
        es_loss = float(np.mean(es_losses))

        if es_loss < best_loss - 1e-4:
            best_loss = es_loss
            best_epoch = epoch
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        log.info(
            "conv2d latent=%d epoch=%d train_mse=%.4f early_stop_mse=%.4f best_epoch=%d",
            latent_dim,
            epoch,
            float(np.mean(train_losses)),
            es_loss,
            best_epoch,
        )

        if (
            epoch_sample_targets is not None
            and epoch_sample_dir is not None
            and (epoch == 1 or epoch % epoch_sample_stride == 0)
        ):
            with torch.no_grad():
                pred_batch = model(epoch_sample_targets).cpu().numpy()[:, 0]
            target_batch = epoch_sample_targets.cpu().numpy()[:, 0]
            save_sample_pngs(
                pred_batch.astype(np.float32),
                target_batch.astype(np.float32),
                mask,
                epoch_sample_dir / f"epoch{epoch:03d}",
                pred_batch.shape[0],
                label=f"ep{epoch:03d}",
            )

        if bad_epochs >= int(args.patience) and epoch >= 10:
            break

    if best_state is None:
        raise RuntimeError("Conv2D AE did not produce a checkpoint.")
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
                "image_shape": image_shape,
            },
            checkpoint_dir / f"conv2d_ae_latent{latent_dim}.pt",
        )

    pred_images = []
    with torch.no_grad():
        for xb in val_loader:
            pred_images.append(model(xb.to(device)).cpu().numpy()[:, 0])
    pred_imgs = np.concatenate(pred_images, axis=0).astype(np.float32)
    pred_vec = masked_pixel_vector(pred_imgs, mask)
    return pred_vec, {"best_epoch": best_epoch, "best_early_stop_mse": best_loss}, time.time() - t0


def save_sample_pngs(
    pred_imgs: np.ndarray,
    target_imgs: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    n_samples: int,
    label: str,
) -> None:
    if n_samples <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(int(n_samples), pred_imgs.shape[0], target_imgs.shape[0])
    indices = np.linspace(0, pred_imgs.shape[0] - 1, num=n, dtype=int)
    for k, idx in enumerate(indices):
        target = target_imgs[idx].copy()
        pred = pred_imgs[idx].copy()
        target[~mask] = np.nan
        pred[~mask] = np.nan
        vmax = float(np.nanmax(np.abs(np.stack([target, pred])))) or 1.0
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, img, title in zip(axes, (target, pred), ("target", "reconstruction")):
            im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title(f"{label} TR={int(idx)} {title}")
            ax.set_axis_off()
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        cbar.set_label("z-scored BOLD")
        out_path = out_dir / f"{label}_sample_{k:02d}_TR{int(idx)}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


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

    responses, total_voxels = load_responses(args, stories, response_root)
    log.info("Loaded responses for %s: %d voxels (mask_thick)", args.subject, total_voxels)
    x_train = stack_stories(responses, train_stories)
    x_val = stack_stories(responses, val_stories)
    del responses
    import gc
    gc.collect()
    log.info("Raw response shapes: train=%s val=%s", x_train.shape, x_val.shape)
    x_train, x_val, train_mean, train_std = zscore_train_val(x_train, x_val)
    log.info("Z-scored in place; current dtype=%s", x_train.dtype)

    configure_pycortex_filestore(args.pycortex_filestore)
    import cortex  # noqa: WPS433

    pycortex_subject = resolve_pycortex_subject(args, cortex)
    filestore = os.environ.get("PYCORTEX_FILESTORE") or args.pycortex_filestore
    xfm_name = find_matching_xfm(cortex, filestore, pycortex_subject, total_voxels, args.xfm_name, args.mask_type)
    mask_3d = np.asarray(cortex.db.get_mask(pycortex_subject, xfm_name, args.mask_type), dtype=bool)
    if int(mask_3d.sum()) != int(total_voxels):
        raise ValueError(
            f"pycortex mask has {int(mask_3d.sum())} voxels but responses have {total_voxels}; "
            "check --xfm-name / --mask-type."
        )

    output_root = Path(args.output_dir).expanduser().resolve()
    image_cache_dir = (
        Path(args.image_cache_dir).expanduser().resolve()
        if args.image_cache_dir
        else output_root / "flatmap_image_cache"
    )
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(image_cache_dir, args.subject, xfm_name, args.mask_type, args.image_height)

    cache_ok = (
        not args.rebuild_cache
        and paths["train_images"].is_file()
        and paths["val_images"].is_file()
        and paths["pixel_mask"].is_file()
        and paths["meta"].is_file()
    )

    if cache_ok:
        log.info("Loading cached flatmap images from %s (no longer need BOLD arrays)", image_cache_dir)
        del x_train, x_val
        gc.collect()
        images_train = np.load(paths["train_images"]).astype(np.float32, copy=False)
        images_val = np.load(paths["val_images"]).astype(np.float32, copy=False)
        pixel_mask = np.load(paths["pixel_mask"]).astype(bool, copy=False)
        with open(paths["meta"], encoding="utf-8") as f:
            cache_meta = json.load(f)
        log.info(
            "Cached images: train=%s val=%s pixel_mask=%s",
            images_train.shape,
            images_val.shape,
            pixel_mask.shape,
        )
    else:
        log.info("Rendering flatmap images (train) ...")
        images_train, pixel_mask, extents = render_flatmap_images(
            x_train,
            pycortex_subject=pycortex_subject,
            xfm_name=xfm_name,
            image_height=int(args.image_height),
            log_every=int(args.render_batch_log_every),
            label="train",
        )
        del x_train
        gc.collect()
        log.info("Rendering flatmap images (val) ...")
        images_val, _val_mask, _ = render_flatmap_images(
            x_val,
            pycortex_subject=pycortex_subject,
            xfm_name=xfm_name,
            image_height=int(args.image_height),
            log_every=int(args.render_batch_log_every),
            label="val",
        )
        del x_val
        gc.collect()
        np.save(paths["train_images"], images_train)
        np.save(paths["val_images"], images_val)
        np.save(paths["pixel_mask"], pixel_mask)
        cache_meta = {
            "subject": args.subject,
            "pycortex_subject": pycortex_subject,
            "xfm_name": xfm_name,
            "mask_type": args.mask_type,
            "image_height": int(args.image_height),
            "extents": list(map(float, extents)),
            "n_train": int(images_train.shape[0]),
            "n_val": int(images_val.shape[0]),
            "image_shape": [int(s) for s in images_train.shape[1:]],
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(cache_meta, f, indent=2)
        log.info("Cached flatmap images at %s", image_cache_dir)

    if images_train.shape[1:] != images_val.shape[1:]:
        raise RuntimeError(
            f"Train/val rendered image shapes disagree: {images_train.shape[1:]} vs {images_val.shape[1:]}"
        )
    if pixel_mask.shape != images_train.shape[1:]:
        log.warning(
            "Cached pixel_mask shape %s does not match image shape %s; recomputing from images_train[0].",
            pixel_mask.shape,
            images_train.shape[1:],
        )
        pixel_mask = np.isfinite(images_train[0]).copy()

    pad_hw = pad_amount(images_train.shape[1:], int(args.pad_to_multiple))
    images_train = pad_image_stack(images_train, pad_hw)
    images_val = pad_image_stack(images_val, pad_hw)
    pixel_mask = pad_mask(pixel_mask, pad_hw)
    log.info(
        "Image shape after pad-to-multiple=%d (pad_hw=%s): train=%s val=%s pixel_mask=%s n_pixels=%d",
        int(args.pad_to_multiple),
        pad_hw,
        images_train.shape,
        images_val.shape,
        pixel_mask.shape,
        int(pixel_mask.sum()),
    )

    fill_nan_inplace(images_train, pixel_mask)
    fill_nan_inplace(images_val, pixel_mask)

    target_vec_train = masked_pixel_vector(images_train, pixel_mask)
    target_vec_val = masked_pixel_vector(images_val, pixel_mask)
    log.info(
        "Masked pixel vector dim=%d (train rows=%d, val rows=%d)",
        target_vec_train.shape[1],
        target_vec_train.shape[0],
        target_vec_val.shape[0],
    )

    tag = (
        args.tag
        or f"{args.subject}__flatmap_conv2d_ae_vs_pca__h{int(args.image_height)}__latent{'-'.join(map(str, args.latent_dims))}__seed{args.seed}"
    )
    out_dir = output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "sample_pngs"
    training_progress_dir = out_dir / "training_progress"
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
        pca.fit(target_vec_train)
        pca_pred_vec = pca.inverse_transform(pca.transform(target_vec_val)).astype(np.float32)
        metric = reconstruction_metrics(pca_pred_vec, target_vec_val)
        rows.append({"model": "pca", "latent_dim": latent_dim, "elapsed_sec": time.time() - t, **metric})
        log.info(
            "pca latent=%d ve=%.4f pattern_r=%.4f voxel_r=%.4f",
            latent_dim,
            metric["variance_explained"],
            metric["pattern_r_mean"],
            metric["voxel_r_mean"],
        )

        log.info("==== Conv2D denoising AE latent=%d ====", latent_dim)
        epoch_sample_dir = (
            training_progress_dir / f"latent{latent_dim}"
            if int(args.epoch_png_count) > 0
            else None
        )
        ae_pred_vec, info, elapsed = train_conv_ae(
            images_train,
            images_val,
            pixel_mask,
            latent_dim,
            args,
            device,
            checkpoint_dir,
            epoch_sample_dir=epoch_sample_dir,
        )
        metric_ae = reconstruction_metrics(ae_pred_vec, target_vec_val)
        rows.append({"model": "conv2d_denoising_ae", "latent_dim": latent_dim, "elapsed_sec": elapsed, **metric_ae, **info})
        log.info(
            "conv2d ae latent=%d ve=%.4f pattern_r=%.4f voxel_r=%.4f best_epoch=%s",
            latent_dim,
            metric_ae["variance_explained"],
            metric_ae["pattern_r_mean"],
            metric_ae["voxel_r_mean"],
            info["best_epoch"],
        )

        if args.sample_png_count > 0:
            ae_pred_imgs = vector_to_image(ae_pred_vec, pixel_mask)
            pca_pred_imgs = vector_to_image(pca_pred_vec, pixel_mask)
            target_imgs = vector_to_image(target_vec_val, pixel_mask)
            save_sample_pngs(
                ae_pred_imgs,
                target_imgs,
                pixel_mask,
                samples_dir / f"latent{latent_dim}",
                int(args.sample_png_count),
                label="ae",
            )
            save_sample_pngs(
                pca_pred_imgs,
                target_imgs,
                pixel_mask,
                samples_dir / f"latent{latent_dim}",
                int(args.sample_png_count),
                label="pca",
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
    csv_path = out_dir / "conv2d_autoencoder_vs_pca.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    np.savez(
        out_dir / "zscore_stats.npz",
        mean=train_mean,
        std=train_std,
    )
    np.save(out_dir / "pixel_mask.npy", pixel_mask)
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
                "n_total_voxels": int(total_voxels),
                "image_height": int(args.image_height),
                "image_shape_padded": [int(s) for s in images_train.shape[1:]],
                "n_pixels_in_mask": int(pixel_mask.sum()),
                "latent_dims": [int(x) for x in args.latent_dims],
                "base_channels": int(args.base_channels),
                "dropout": float(args.dropout),
                "input_noise_std": float(args.input_noise_std),
                "input_mask_prob": float(args.input_mask_prob),
                "data_train_dir": config.DATA_TRAIN_DIR,
                "response_root": response_root,
                "image_cache_dir": str(image_cache_dir),
                "image_cache_meta": cache_meta,
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
