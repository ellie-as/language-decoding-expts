#!/usr/bin/env python3
"""MindEye-style multi-subject text decoder.

Decodes the lagged 5-TR text-embedding window from a single TR of fMRI by:

1. Mapping each subject's ROI voxels into a shared ``latent_dim`` (4096) via a
   subject-specific ``Linear``.
2. Running a shared 4-block residual MLP backbone over the latent.
3. Projecting to the GTR text embedding via a shared ``LayerNorm + Linear``
   head.

Training pools batches across all selected subjects (one batch per subject per
optimizer step), so the shared backbone and head see ``≈ n_subjects ×`` more
data than a single-subject model. The CLIP-style InfoNCE term in
``--loss mse_clip`` uses negatives from every subject in the same step.

Designed to run on an A100 cluster job. Bf16 mixed precision is on by default.
Pass ``--no-amp`` to disable, or ``--torch-device cpu/mps`` for local debug.
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import (  # noqa: E402
    EMBEDDING_MODELS,
    grouped_train_val_split,
    mean_cosine,
    resolve_torch_device,
    retrieval_metrics,
    rse,
)
from data import (  # noqa: E402
    SingleTRChunkDataset,
    SubjectData,
    compute_target_stats,
    load_subject_data,
    make_subject_dataset,
)
from model import MindEyeText, compute_loss  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mindeye_text")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"],
                   help="Subjects to pool. Each contributes its own input projection.")
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
    p.add_argument("--roi", default="full_frontal",
                   help="Voxel set per subject. Same name applied to every subject.")
    p.add_argument("--brain-pca", type=int, default=0,
                   help="If >0, fit per-subject PCA on training z-scored voxels and feed PCs "
                        "(post-PC z-score) to the subject-specific Linear instead of raw voxels. "
                        "Reduces the per-subject input projection from ~25k*latent to "
                        "n_components*latent params per subject.")

    p.add_argument("--feature-model", default="gtr-base", choices=list(EMBEDDING_MODELS.keys()))
    p.add_argument("--chunk-trs", type=int, default=5,
                   help="Width of the text window in TRs (5-TR target).")
    p.add_argument("--lag-trs", type=int, default=3,
                   help="HRF lag in TRs from text-window start to brain TR.")
    p.add_argument("--brain-offset", type=int, default=0,
                   help="Index within the [i+lag, i+lag+chunk_trs) window for the input TR. "
                        "0 = first TR (matches lag), chunk_trs//2 = middle TR.")
    p.add_argument("--normalize-targets", action="store_true",
                   help="L2-normalize raw embedding targets before per-dim z-score (matches GTR usage).")
    p.add_argument("--shared-target-stats", action=argparse.BooleanOptionalAction, default=True,
                   help="Pool training embeddings across subjects when computing target z-score stats.")

    p.add_argument("--latent-dim", type=int, default=4096)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--head-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="LayerNorm before the final linear head.")

    p.add_argument("--loss", default="mse_clip",
                   choices=["mse", "cosine", "mse_cosine", "mse_clip"])
    p.add_argument("--cosine-weight", type=float, default=0.5,
                   help="Weight on the cosine term in --loss mse_cosine.")
    p.add_argument("--clip-weight", type=float, default=0.5,
                   help="Weight on the InfoNCE term in --loss mse_clip.")
    p.add_argument("--clip-temp", type=float, default=0.05,
                   help="Temperature for the InfoNCE term.")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-subject batch size; effective batch ≈ N_subjects * this.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers per subject. Datasets fit in RAM, so 0 is usually fastest.")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30,
                   help="Early-stopping patience on combined validation loss.")
    p.add_argument("--val-frac", type=float, default=0.12,
                   help="Per-subject grouped (story-level) validation fraction.")
    p.add_argument("--torch-device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Use bf16 autocast on CUDA. Recommended for A100.")

    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--embedding-cache-dir", default="27-04-expts/cache",
                   help="Shares the cache produced by 27-04-expts/train_5tr_chunk_nn.py.")
    p.add_argument("--output-dir", default="mindeye_text/results")
    p.add_argument("--tag", default=None,
                   help="Optional run name. Defaults to a hyperparameter-derived tag.")
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_default_tag(args: argparse.Namespace) -> str:
    subj_tag = "-".join(args.subjects)
    norm_tag = "_norm" if args.normalize_targets else ""
    loss_tag = args.loss
    if args.loss == "mse_clip":
        loss_tag += f"-cw{args.clip_weight:g}-t{args.clip_temp:g}"
    elif args.loss == "mse_cosine":
        loss_tag += f"-w{args.cosine_weight:g}"
    pca_tag = f"__brainpca{args.brain_pca}" if int(getattr(args, "brain_pca", 0) or 0) > 0 else ""
    return (
        f"mindeye_text__{args.feature_model}{norm_tag}__{subj_tag}__{args.roi}"
        f"__latent{args.latent_dim}-blocks{args.n_blocks}{pca_tag}__loss-{loss_tag}"
        f"__lag{args.lag_trs}-chunk{args.chunk_trs}-off{args.brain_offset}__seed{args.seed}"
    )


def grouped_split_for_dataset(
    dataset: SingleTRChunkDataset, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    groups = dataset.chunk_story_groups()
    return grouped_train_val_split(groups, val_frac, seed)


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def dim_r(true: np.ndarray, pred: np.ndarray) -> float:
    a = true.astype(np.float64, copy=False) - true.mean(axis=0, keepdims=True)
    b = pred.astype(np.float64, copy=False) - pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    r = np.divide((a * b).sum(axis=0), denom, out=np.zeros(a.shape[1], dtype=np.float64), where=denom != 0)
    return float(np.nan_to_num(r).mean())


@torch.no_grad()
def predict_subject(
    model: MindEyeText,
    dataset: SingleTRChunkDataset,
    subject: str,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pred_z, true_z)`` arrays in z-scored target space."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast_ctx(device, use_amp):
            pred = model(xb, subject)
        preds.append(pred.detach().to("cpu").float().numpy())
        targets.append(yb.numpy())
    return np.vstack(preds).astype(np.float32), np.vstack(targets).astype(np.float32)


def denormalize(pred_z: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    return (pred_z * target_std + target_mean).astype(np.float32)


def per_subject_metrics(
    true_emb: np.ndarray,
    pred_emb: np.ndarray,
) -> Dict[str, float]:
    top1, mrr, mean_rank = retrieval_metrics(true_emb, pred_emb)
    return {
        "embedding_dim_r": dim_r(true_emb, pred_emb),
        "embedding_cosine": mean_cosine(true_emb, pred_emb),
        "retrieval_top1": float(top1),
        "retrieval_mrr": float(mrr),
        "retrieval_mean_rank": float(mean_rank),
        "n_test_chunks": int(true_emb.shape[0]),
    }


def run() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_torch_device(args.torch_device)
    use_amp = bool(args.amp) and device.type == "cuda"
    log.info("Device: %s | AMP: %s", device, "bf16" if use_amp else "off")

    log.info("Loading subjects: %s", args.subjects)
    subject_data: Dict[str, SubjectData] = {}
    for subj in args.subjects:
        subject_data[subj] = load_subject_data(args, subj)

    embedding_dims = {sd.embedding_dim for sd in subject_data.values()}
    if len(embedding_dims) != 1:
        raise ValueError(f"Inconsistent text embedding dim across subjects: {embedding_dims}")
    embed_dim = int(next(iter(embedding_dims)))
    log.info("Text embedding dim: %d (%s)", embed_dim, args.feature_model)

    embeddings_train_per_subject = {s: subject_data[s].embeddings_train for s in args.subjects}
    train_stories_per_subject = {s: subject_data[s].train_stories for s in args.subjects}
    target_means, target_stds = compute_target_stats(
        embeddings_train_per_subject,
        train_stories_per_subject,
        normalize_targets=bool(args.normalize_targets),
        shared=bool(args.shared_target_stats),
    )

    train_datasets: Dict[str, SingleTRChunkDataset] = {}
    test_datasets: Dict[str, SingleTRChunkDataset] = {}
    for subj, sd in subject_data.items():
        train_datasets[subj] = make_subject_dataset(
            sd,
            chunk_trs=args.chunk_trs,
            lag_trs=args.lag_trs,
            brain_offset=args.brain_offset,
            target_mean=target_means[subj],
            target_std=target_stds[subj],
            normalize_targets=bool(args.normalize_targets),
            split="train",
        )
        test_datasets[subj] = make_subject_dataset(
            sd,
            chunk_trs=args.chunk_trs,
            lag_trs=args.lag_trs,
            brain_offset=args.brain_offset,
            target_mean=target_means[subj],
            target_std=target_stds[subj],
            normalize_targets=bool(args.normalize_targets),
            split="test",
        )
        log.info(
            "[%s] train chunks=%d, test chunks=%d, voxels=%d, feat_dim=%d%s",
            subj,
            len(train_datasets[subj]),
            len(test_datasets[subj]),
            sd.n_voxels,
            sd.feat_dim,
            f" (brain PCA, evr={sd.pca_explained_variance_ratio.sum():.3f})"
            if sd.pca_components is not None and sd.pca_explained_variance_ratio is not None
            else "",
        )

    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    val_indices_per_subject: Dict[str, np.ndarray] = {}
    for subj, ds in train_datasets.items():
        train_idx, val_idx = grouped_split_for_dataset(ds, args.val_frac, args.seed)
        log.info("[%s] split: %d train, %d val (story-grouped)", subj, len(train_idx), len(val_idx))
        train_loaders[subj] = DataLoader(
            Subset(ds, train_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        val_loaders[subj] = DataLoader(
            Subset(ds, val_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_indices_per_subject[subj] = val_idx

    feat_dims = {subj: subject_data[subj].feat_dim for subj in args.subjects}
    voxel_counts = {subj: subject_data[subj].n_voxels for subj in args.subjects}
    model = MindEyeText(
        voxel_counts=feat_dims,
        embed_dim=embed_dim,
        latent_dim=args.latent_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_norm=bool(args.head_norm),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model: %.1fM trainable params (feat_dims=%s, latent_dim=%d, blocks=%d)",
        n_params / 1e6, feat_dims, args.latent_dim, args.n_blocks,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_steps_per_epoch = min(len(loader) for loader in train_loaders.values())
    if n_steps_per_epoch == 0:
        raise RuntimeError("At least one subject's train loader is empty (batch_size > #train chunks?)")

    history: List[dict] = []
    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    total_train_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_iters = {subj: iter(loader) for subj, loader in train_loaders.items()}
        train_losses: List[float] = []
        epoch_start = time.time()
        for _step in range(n_steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            latents: List[torch.Tensor] = []
            targets: List[torch.Tensor] = []
            for subj, it in train_iters.items():
                xb, yb = next(it)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast_ctx(device, use_amp):
                    latents.append(model.encode(xb, subj))
                targets.append(yb)
            with autocast_ctx(device, use_amp):
                latent = torch.cat(latents, dim=0)
                target = torch.cat(targets, dim=0)
                pred = model.decode(latent)
                loss = compute_loss(
                    pred,
                    target,
                    kind=args.loss,
                    cosine_weight=args.cosine_weight,
                    clip_weight=args.clip_weight,
                    clip_temp=args.clip_temp,
                )
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().to("cpu")))

        model.eval()
        val_total = 0.0
        val_count = 0
        per_subject_val: Dict[str, float] = {}
        with torch.no_grad():
            for subj, loader in val_loaders.items():
                subj_total = 0.0
                subj_count = 0
                for xb, yb in loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with autocast_ctx(device, use_amp):
                        pred = model(xb, subj)
                        loss = compute_loss(
                            pred,
                            yb,
                            kind=args.loss,
                            cosine_weight=args.cosine_weight,
                            clip_weight=args.clip_weight,
                            clip_temp=args.clip_temp,
                        )
                    n = int(xb.size(0))
                    subj_total += float(loss.detach().to("cpu")) * n
                    subj_count += n
                if subj_count > 0:
                    per_subject_val[subj] = subj_total / subj_count
                    val_total += subj_total
                    val_count += subj_count
        val_loss = val_total / max(val_count, 1)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_elapsed = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_kind": args.loss,
                "elapsed_sec": float(epoch_elapsed),
                **{f"val_loss_{subj}": float(v) for subj, v in per_subject_val.items()},
            }
        )
        if epoch == 1 or epoch % 5 == 0:
            per_subj_str = " ".join(f"{s}={v:.4f}" for s, v in per_subject_val.items())
            log.info(
                "epoch %03d train=%.5f val=%.5f (%s) | per-subj %s | %.1fs",
                epoch, train_loss, val_loss, args.loss, per_subj_str, epoch_elapsed,
            )

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("Early stopping at epoch %d (best val_loss=%.5f)", epoch, best_val)
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    log.info("Computing per-subject test metrics ...")
    metrics_rows: List[dict] = []
    pooled_pred_emb: List[np.ndarray] = []
    pooled_true_emb: List[np.ndarray] = []
    pooled_subjects: List[np.ndarray] = []
    predictions_payload: Dict[str, np.ndarray] = {}
    for subj in args.subjects:
        sd = subject_data[subj]
        ds = test_datasets[subj]
        pred_z, _true_z_unused = predict_subject(
            model, ds, subj, device, args.batch_size, use_amp
        )
        pred_emb = denormalize(pred_z, target_means[subj], target_stds[subj])
        true_emb = ds.stack_targets_raw()
        if args.normalize_targets:
            norms_true = np.linalg.norm(true_emb, axis=1, keepdims=True)
            norms_true[norms_true == 0] = 1.0
            true_emb = (true_emb / norms_true).astype(np.float32)
        m = per_subject_metrics(true_emb, pred_emb)
        m.update(
            {
                "subject": subj,
                "roi": sd.roi_name,
                "n_voxels": sd.n_voxels,
                "n_train_chunks": int(len(train_datasets[subj])),
            }
        )
        metrics_rows.append(m)
        log.info(
            "[%s] top1=%.4f mrr=%.4f mean_rank=%.2f cos=%.4f dim_r=%.4f (n=%d)",
            subj,
            m["retrieval_top1"],
            m["retrieval_mrr"],
            m["retrieval_mean_rank"],
            m["embedding_cosine"],
            m["embedding_dim_r"],
            m["n_test_chunks"],
        )
        pooled_pred_emb.append(pred_emb)
        pooled_true_emb.append(true_emb)
        pooled_subjects.append(np.full(pred_emb.shape[0], subj))
        if args.save_predictions:
            predictions_payload[f"pred_emb__{subj}"] = pred_emb
            predictions_payload[f"true_emb__{subj}"] = true_emb

    pooled_pred = np.vstack(pooled_pred_emb)
    pooled_true = np.vstack(pooled_true_emb)
    pooled_meta = np.concatenate(pooled_subjects)
    combined = per_subject_metrics(pooled_true, pooled_pred)
    combined.update(
        {
            "subject": "ALL",
            "roi": next(iter(subject_data.values())).roi_name,
            "n_voxels": -1,
            "n_train_chunks": int(sum(len(d) for d in train_datasets.values())),
        }
    )
    metrics_rows.append(combined)
    log.info(
        "[ALL] top1=%.4f mrr=%.4f mean_rank=%.2f cos=%.4f dim_r=%.4f (n=%d)",
        combined["retrieval_top1"],
        combined["retrieval_mrr"],
        combined["retrieval_mean_rank"],
        combined["embedding_cosine"],
        combined["embedding_dim_r"],
        combined["n_test_chunks"],
    )

    tag = args.tag or build_default_tag(args)
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    fieldnames = sorted({k for row in metrics_rows for k in row.keys()})
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    log.info("Wrote metrics: %s", metrics_path)

    history_path = out_dir / "history.csv"
    if history:
        history_fields = sorted({k for row in history for k in row.keys()})
        with open(history_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writeheader()
            writer.writerows(history)
        log.info("Wrote history: %s", history_path)

    metadata = {
        "args": vars(args),
        "tag": tag,
        "embed_dim": embed_dim,
        "voxel_counts": {subj: int(sd.n_voxels) for subj, sd in subject_data.items()},
        "feat_dims": {subj: int(sd.feat_dim) for subj, sd in subject_data.items()},
        "brain_pca_explained_variance": {
            subj: float(sd.pca_explained_variance_ratio.sum())
            for subj, sd in subject_data.items()
            if sd.pca_explained_variance_ratio is not None
        },
        "train_stories": {subj: sd.train_stories for subj, sd in subject_data.items()},
        "test_stories": {subj: sd.test_stories for subj, sd in subject_data.items()},
        "embedding_cache": {subj: sd.embedding_cache for subj, sd in subject_data.items()},
        "best_val_loss": float(best_val),
        "n_train_chunks": {subj: int(len(d)) for subj, d in train_datasets.items()},
        "n_test_chunks": {subj: int(len(d)) for subj, d in test_datasets.items()},
        "device": str(device),
        "amp": bool(use_amp),
        "model_params_M": float(n_params / 1e6),
        "total_train_seconds": float(time.time() - total_train_start),
    }
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Wrote metadata: %s", metadata_path)

    checkpoint = {
        "model_state_dict": {k: v for k, v in best_state.items()},
        "voxel_counts": voxel_counts,
        "feat_dims": feat_dims,
        "embed_dim": int(embed_dim),
        "latent_dim": int(args.latent_dim),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),
        "head_norm": bool(args.head_norm),
        "feature_model": str(args.feature_model),
        "chunk_trs": int(args.chunk_trs),
        "lag_trs": int(args.lag_trs),
        "brain_offset": int(args.brain_offset),
        "normalize_targets": bool(args.normalize_targets),
        "brain_pca": int(getattr(args, "brain_pca", 0) or 0),
        "subjects": list(args.subjects),
        "voxels": {subj: subject_data[subj].voxels for subj in args.subjects},
        "voxel_mean": {subj: subject_data[subj].voxel_mean for subj in args.subjects},
        "voxel_std": {subj: subject_data[subj].voxel_std for subj in args.subjects},
        "pca_components": {subj: subject_data[subj].pca_components for subj in args.subjects},
        "pca_mean": {subj: subject_data[subj].pca_mean for subj in args.subjects},
        "pc_mean": {subj: subject_data[subj].pc_mean for subj in args.subjects},
        "pc_std": {subj: subject_data[subj].pc_std for subj in args.subjects},
        "pca_explained_variance_ratio": {
            subj: subject_data[subj].pca_explained_variance_ratio for subj in args.subjects
        },
        "target_mean": {subj: target_means[subj] for subj in args.subjects},
        "target_std": {subj: target_stds[subj] for subj in args.subjects},
        "roi_name": {subj: subject_data[subj].roi_name for subj in args.subjects},
    }
    model_path = out_dir / "model.pt"
    torch.save(checkpoint, model_path)
    log.info("Wrote checkpoint: %s", model_path)

    if args.save_predictions:
        pred_path = out_dir / "predictions.npz"
        np.savez_compressed(
            pred_path,
            pooled_pred=pooled_pred,
            pooled_true=pooled_true,
            pooled_subject=pooled_meta,
            **predictions_payload,
        )
        log.info("Wrote predictions: %s", pred_path)


if __name__ == "__main__":
    run()
