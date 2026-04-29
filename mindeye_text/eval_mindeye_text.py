#!/usr/bin/env python3
"""Evaluate a saved MindEye-style multi-subject text decoder.

Loads a ``model.pt`` checkpoint from a training run and re-computes per-subject
and pooled retrieval / cosine / dim-r metrics on the held-out test stories.
The checkpoint already stores per-subject voxels, voxel z-score stats, target
z-score stats, and the chunk/lag/offset settings, so all you have to pass is
``--checkpoint <path>`` plus any data-source overrides for the cluster.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))
sys.path.insert(0, str(REPO_DIR / "27-04-expts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_summaries_encoding as rse  # noqa: E402
from run_summary_decoding import retrieval_metrics  # noqa: E402
from train_lagged_text_pca_mlp import (  # noqa: E402
    mean_cosine,
    resolve_torch_device,
)

from data import load_subject_data, make_subject_dataset  # noqa: E402
from model import MindEyeText  # noqa: E402
from train_mindeye_text import (  # noqa: E402
    autocast_ctx,
    denormalize,
    dim_r,
    per_subject_metrics,
    predict_subject,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mindeye_text.eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to model.pt produced by train_mindeye_text.py")
    p.add_argument("--metadata", default=None,
                   help="Optional metadata.json; defaults to <checkpoint dir>/metadata.json.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write eval CSV (defaults to checkpoint directory).")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subset of checkpoint subjects to evaluate. Defaults to all.")
    p.add_argument("--stories", nargs="+", default=None,
                   help="Override the story list used for both train selection and held-out test.")
    p.add_argument("--sessions", nargs="+", type=int, default=None)
    p.add_argument("--holdout-stories", nargs="+", default=None)
    p.add_argument("--holdout-count", type=int, default=None)
    p.add_argument("--no-story-holdout", action="store_true")
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument("--mounted-project-root", default=None)
    p.add_argument("--local-cache-root", default=None)
    p.add_argument("--summaries-dir", default=None)
    p.add_argument("--ba-dir", default=None)
    p.add_argument("--embedding-cache-dir", default=None)
    p.add_argument("--feature-model", default=None)
    p.add_argument("--roi", default=None)
    p.add_argument("--torch-device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _resolve_metadata(checkpoint_path: Path, override: str | None) -> dict:
    meta_path = Path(override) if override else checkpoint_path.parent / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Cannot find metadata.json next to checkpoint: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _build_data_args(eval_args: argparse.Namespace, train_args: dict) -> argparse.Namespace:
    """Reconstruct an argparse.Namespace compatible with ``data.load_subject_data``.

    Train args are used as defaults; CLI overrides on the eval call take precedence.
    """
    merged = deepcopy(train_args)
    overrides = {
        k: v
        for k, v in vars(eval_args).items()
        if v is not None and k != "checkpoint" and k != "metadata" and k != "output_dir"
    }
    merged.update(overrides)
    if eval_args.no_story_holdout:
        merged["no_story_holdout"] = True
    merged.setdefault("mounted_project_root", str(rse.DEFAULT_MOUNTED_PROJECT_ROOT))
    merged.setdefault("local_cache_root", str(rse.DEFAULT_LOCAL_CACHE_ROOT))
    merged.setdefault("summaries_dir", str(rse.LOCAL_DEFAULT_SUMMARIES_DIR))
    merged.setdefault("ba_dir", str(rse.LOCAL_DEFAULT_BA_DIR))
    merged.setdefault("embedding_cache_dir", "27-04-expts/cache")
    merged.setdefault("local_compute_mode", False)
    return argparse.Namespace(**merged)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    metadata = _resolve_metadata(checkpoint_path, args.metadata)
    train_args = metadata["args"]

    data_args = _build_data_args(args, train_args)
    requested_subjects = args.subjects or train_args["subjects"]

    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    voxel_counts = {s: int(c) for s, c in ckpt["voxel_counts"].items()}
    missing = [s for s in requested_subjects if s not in voxel_counts]
    if missing:
        raise ValueError(f"Checkpoint has no projection for subjects: {missing}")

    device = resolve_torch_device(args.torch_device)
    use_amp = bool(args.amp) and device.type == "cuda"
    log.info("Device: %s | AMP: %s", device, "bf16" if use_amp else "off")

    model = MindEyeText(
        voxel_counts=voxel_counts,
        embed_dim=int(ckpt["embed_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        n_blocks=int(ckpt["n_blocks"]),
        dropout=float(ckpt["dropout"]),
        head_norm=bool(ckpt.get("head_norm", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    metrics_rows: List[dict] = []
    pooled_pred_emb: List[np.ndarray] = []
    pooled_true_emb: List[np.ndarray] = []
    pooled_subject: List[np.ndarray] = []
    predictions_payload: Dict[str, np.ndarray] = {}

    target_means = {s: np.asarray(ckpt["target_mean"][s], dtype=np.float32) for s in requested_subjects}
    target_stds = {s: np.asarray(ckpt["target_std"][s], dtype=np.float32) for s in requested_subjects}

    for subj in requested_subjects:
        subject_data_obj = load_subject_data(data_args, subj)
        ds = make_subject_dataset(
            subject_data_obj,
            chunk_trs=int(ckpt["chunk_trs"]),
            lag_trs=int(ckpt["lag_trs"]),
            brain_offset=int(ckpt["brain_offset"]),
            target_mean=target_means[subj],
            target_std=target_stds[subj],
            normalize_targets=bool(ckpt.get("normalize_targets", False)),
            split="test",
        )
        log.info("[%s] %d test chunks, %d voxels", subj, len(ds), subject_data_obj.n_voxels)
        if subject_data_obj.n_voxels != voxel_counts[subj]:
            raise ValueError(
                f"Subject {subj}: voxel count mismatch (data={subject_data_obj.n_voxels}, "
                f"checkpoint={voxel_counts[subj]}). Re-run with the same ROI."
            )
        pred_z, _ = predict_subject(model, ds, subj, device, args.batch_size, use_amp)
        pred_emb = denormalize(pred_z, target_means[subj], target_stds[subj])
        true_emb = ds.stack_targets_raw()
        if bool(ckpt.get("normalize_targets", False)):
            norms = np.linalg.norm(true_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            true_emb = (true_emb / norms).astype(np.float32)

        m = per_subject_metrics(true_emb, pred_emb)
        m.update(
            {
                "subject": subj,
                "roi": subject_data_obj.roi_name,
                "n_voxels": subject_data_obj.n_voxels,
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
        pooled_subject.append(np.full(pred_emb.shape[0], subj))
        if args.save_predictions:
            predictions_payload[f"pred_emb__{subj}"] = pred_emb
            predictions_payload[f"true_emb__{subj}"] = true_emb

    pooled_pred = np.vstack(pooled_pred_emb)
    pooled_true = np.vstack(pooled_true_emb)
    pooled_meta = np.concatenate(pooled_subject)
    combined = per_subject_metrics(pooled_true, pooled_pred)
    combined.update({"subject": "ALL", "roi": metrics_rows[0]["roi"], "n_voxels": -1})
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

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "eval_metrics.csv"
    fieldnames = sorted({k for row in metrics_rows for k in row.keys()})
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    log.info("Wrote eval metrics: %s", metrics_path)

    if args.save_predictions:
        pred_path = out_dir / "eval_predictions.npz"
        np.savez_compressed(
            pred_path,
            pooled_pred=pooled_pred,
            pooled_true=pooled_true,
            pooled_subject=pooled_meta,
            **predictions_payload,
        )
        log.info("Wrote eval predictions: %s", pred_path)


if __name__ == "__main__":
    main()
