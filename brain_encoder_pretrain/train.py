#!/usr/bin/env python3
"""Train a cross-subject MAE-style brain encoder on the LeBel dataset.

Each step samples random TR-chunks from multiple subjects, masks 50% of the
TR tokens, and asks the model to reconstruct the masked TRs in each subject's
native voxel space. The encoder is shared across subjects; only the input /
output projection heads are per-subject.

Example:
    python -m brain_encoder_pretrain.train \\
        --subjects S1 S2 S3 \\
        --d-model 256 --n-enc-layers 4 --n-dec-layers 2 \\
        --chunk-len 64 --batch-size 32 \\
        --steps 20000 --lr 3e-4 \\
        --output-dir brain_encoder_pretrain/runs/run1

Outputs (under --output-dir):
    config.json              training / model config (for loading later)
    train_log.csv            per-step loss log
    ckpt_step_<N>.pt         periodic checkpoints
    ckpt_best.pt             best-by-eval-loss checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402

from brain_encoder_pretrain.dataset import (  # noqa: E402
    MultiSubjectChunkSampler,
    SUBJECT_TO_UTS,
    load_subject_data,
)
from brain_encoder_pretrain.model import (  # noqa: E402
    BrainEncoderMAE,
    masked_mse_loss,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brain_encoder_pretrain.train")


def _resolve_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def parse_args():
    p = argparse.ArgumentParser()
    # Data selection
    p.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
        help="Sessions to draw stories from. Defaults to the full training-session list "
             "used by run_summaries_encoding.py (sessions 1 / 13 / 16 / 17 / 19 are absent "
             "in sess_to_story.json).",
    )
    p.add_argument(
        "--stories",
        nargs="*",
        default=None,
        help="Explicit story list. If provided, overrides --sessions.",
    )
    # Story holdout — these stories are NEVER seen during pretraining.
    p.add_argument("--holdout-stories", nargs="*", default=None)
    p.add_argument("--holdout-count", type=int, default=5)
    p.add_argument("--no-story-holdout", action="store_true")

    # Local-compute-mode passthroughs (to read from mounted Ceph if desired)
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument(
        "--mounted-project-root",
        default="/ceph/behrens/ellie/language-decoding-expts",
    )
    p.add_argument(
        "--local-cache-root",
        default=str(REPO_DIR / "local_cache"),
    )

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Per-subject bottleneck dim: voxels -> latent -> d_model. Keeping this "
             "small is what makes the per-subject heads cheap; most capacity lives in "
             "the shared latent<->d_model maps and the Transformer.",
    )
    p.add_argument("--n-enc-layers", type=int, default=4)
    p.add_argument("--n-dec-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask-ratio", type=float, default=0.5)

    # Training
    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    # Output / misc
    p.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "brain_encoder_pretrain" / "runs" / "run1"),
    )
    # Passthrough args used by run_summaries_encoding helpers
    p.add_argument("--summaries-dir", default="")
    p.add_argument("--ba-dir", default="")
    return p.parse_args()


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mounted_root = None
    if args.local_compute_mode:
        try:
            mounted_root = rse.configure_local_compute_mode(args)
        except FileNotFoundError as e:
            log.warning("%s", e)
            log.warning("Falling back to local cache only (no mounted volume).")

    # Resolve response root.
    local_cache_root = Path(args.local_cache_root).expanduser().resolve()
    cached_base = local_cache_root / "data_train"
    fallback_response_root = str(cached_base) if cached_base.exists() else config.DATA_TRAIN_DIR

    # Resolve story list (same pattern as run_h20_decoder_sweep.py).
    try:
        all_stories = rse.load_story_list(args)
    except FileNotFoundError as e:
        log.warning("%s", e)
        # Intersect subject story files when sess_to_story.json is unavailable.
        per_subj = []
        for subj in args.subjects:
            subj_dir = Path(fallback_response_root) / "train_response" / subj
            per_subj.append(set(p.stem for p in subj_dir.glob("*.hf5")))
        all_stories = sorted(set.intersection(*per_subj))
        log.warning("Falling back to story intersection (%d stories).", len(all_stories))

    train_stories, test_stories = rse.split_story_list(all_stories, args)
    log.info(
        "Pretraining on %d stories; holding out %d (never seen): %s",
        len(train_stories), len(test_stories), test_stories,
    )

    # Stage local cache per subject if in local-compute-mode.
    if args.local_compute_mode and mounted_root is not None:
        for subj in args.subjects:
            response_root = rse.stage_local_response_cache(
                subj, all_stories,
                mounted_data_train_dir=config.DATA_TRAIN_DIR,
                cache_root=local_cache_root,
            )
    else:
        response_root = fallback_response_root

    # Load per-subject data.
    subjects_data = {}
    for subj in args.subjects:
        subjects_data[subj] = load_subject_data(subj, train_stories, response_root)

    subject_to_voxels = {s: d.n_voxels for s, d in subjects_data.items()}
    log.info("Subjects and voxel counts: %s", subject_to_voxels)

    # Save config early (useful for feature extraction later).
    cfg = {
        "subjects": list(args.subjects),
        "subject_to_voxels": subject_to_voxels,
        "train_stories": train_stories,
        "heldout_stories": test_stories,
        "model": {
            "d_model": args.d_model,
            "latent_dim": args.latent_dim,
            "n_enc_layers": args.n_enc_layers,
            "n_dec_layers": args.n_dec_layers,
            "n_heads": args.n_heads,
            "ff_mult": args.ff_mult,
            "dropout": args.dropout,
            "max_len": max(256, args.chunk_len * 2),
        },
        "training": {
            "chunk_len": args.chunk_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "mask_ratio": args.mask_ratio,
            "seed": args.seed,
        },
        "response_root": str(response_root),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = _resolve_device(args.device)
    log.info("Using device: %s", device)

    model = BrainEncoderMAE(
        subject_to_voxels=subject_to_voxels,
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_len=cfg["model"]["max_len"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_shared = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("voxel_to_latent.") and not name.startswith("latent_to_voxel.")
    )
    log.info(
        "Model parameters: %.2fM total, %.2fM shared across subjects (%.1f%%)",
        n_params / 1e6, n_shared / 1e6, 100.0 * n_shared / max(1, n_params),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_sampler = MultiSubjectChunkSampler(
        subjects_data=subjects_data,
        chunk_len=args.chunk_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    eval_sampler = MultiSubjectChunkSampler(
        subjects_data=subjects_data,
        chunk_len=args.chunk_len,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )

    log_path = out_dir / "train_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["step", "train_loss", "eval_loss", "lr", "sec"])

    best_eval = float("inf")
    t_start = time.time()
    running_loss = 0.0
    running_n = 0

    for step in range(1, args.steps + 1):
        model.train()
        lr = _cosine_lr(step, args.warmup_steps, args.steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        batch = train_sampler.sample_batch()
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_samples = 0
        for subj, x_np in batch.items():
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            pred, target, mask = model.forward_mae(subj, x, mask_ratio=args.mask_ratio)
            loss = masked_mse_loss(pred, target, mask)
            # Weight sub-losses by number of samples they represent so the
            # gradient matches a single loss computed over the whole batch.
            (loss * x.shape[0]).backward()
            total_loss += float(loss.detach().item()) * x.shape[0]
            total_samples += x.shape[0]
        mean_loss = total_loss / max(1, total_samples)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        running_loss += mean_loss
        running_n += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            avg = running_loss / max(1, running_n)
            log.info(
                "step %6d/%d  train_loss=%.4f  lr=%.2e  (%.1fs)",
                step, args.steps, avg, lr, elapsed,
            )
            running_loss = 0.0
            running_n = 0

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    batch = eval_sampler.sample_batch()
                    for subj, x_np in batch.items():
                        x = torch.from_numpy(x_np).to(device)
                        pred, target, mask = model.forward_mae(subj, x, mask_ratio=args.mask_ratio)
                        eval_losses.append(float(masked_mse_loss(pred, target, mask).item()))
            eval_loss = float(np.mean(eval_losses)) if eval_losses else float("nan")
            log.info("  [eval] step %d  eval_loss=%.4f", step, eval_loss)
            log_writer.writerow([step, f"{mean_loss:.6f}", f"{eval_loss:.6f}", f"{lr:.6e}", f"{time.time()-t_start:.1f}"])
            log_f.flush()

            if eval_loss < best_eval:
                best_eval = eval_loss
                ckpt_path = out_dir / "ckpt_best.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "eval_loss": eval_loss,
                    },
                    ckpt_path,
                )
                log.info("  [eval] saved new best -> %s", ckpt_path)

        if step % args.save_every == 0:
            ckpt_path = out_dir / f"ckpt_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            log.info("  saved periodic checkpoint -> %s", ckpt_path)

    # Final checkpoint
    ckpt_path = out_dir / "ckpt_final.pt"
    torch.save(
        {
            "step": args.steps,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    log.info("Saved final checkpoint -> %s", ckpt_path)

    log_f.close()
    log.info("Done. Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
