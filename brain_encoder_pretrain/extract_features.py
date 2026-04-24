#!/usr/bin/env python3
"""Extract per-TR features from a trained brain encoder.

Given a checkpoint written by `train.py` and a (subject, story list), run the
encoder over the full story sequence (no masking) and save the contextualized
token features [T, d_model] for each story.

Example:
    python -m brain_encoder_pretrain.extract_features \\
        --ckpt brain_encoder_pretrain/runs/run1/ckpt_best.pt \\
        --subject S1 --stories-mode all \\
        --output-dir brain_encoder_pretrain/features/run1

Output layout:
    <output-dir>/<subject>/<story>.npz   -> key "X" with array [T, d_model]

These files are drop-in replacements for raw voxel features and can be loaded
by downstream decoder scripts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "decoding"))

import config  # noqa: E402
import run_summaries_encoding as rse  # noqa: E402

from brain_encoder_pretrain.dataset import load_subject_data  # noqa: E402
from brain_encoder_pretrain.model import BrainEncoderMAE  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brain_encoder_pretrain.extract_features")


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


def encode_story_chunked(
    model: BrainEncoderMAE,
    subject: str,
    X: np.ndarray,
    device: torch.device,
    chunk_len: int,
    overlap: int,
) -> np.ndarray:
    """Encode a full story sequence using overlapping windows averaged at overlaps.

    If T <= model.max_len we encode in a single pass. Otherwise we slide a
    `chunk_len` window with `overlap` TRs shared between neighbours and average
    the feature vectors at overlapping positions.
    """
    T = int(X.shape[0])
    max_len = int(model.max_len)
    d_model = int(model.d_model)

    model.eval()
    with torch.no_grad():
        if T <= max_len:
            x = torch.from_numpy(X[None, :, :]).to(device)
            z = model.encode(subject, x)[0].cpu().numpy()
            return z.astype(np.float32)

        if chunk_len > max_len:
            chunk_len = max_len
        step = max(1, chunk_len - overlap)
        feats = np.zeros((T, d_model), dtype=np.float32)
        counts = np.zeros((T,), dtype=np.int32)
        start = 0
        while start < T:
            end = min(T, start + chunk_len)
            x_slice = X[start:end]
            # Skip tiny tail slices that would break the shape expectations.
            if x_slice.shape[0] < 2:
                break
            x = torch.from_numpy(x_slice[None, :, :]).to(device)
            z = model.encode(subject, x)[0].cpu().numpy().astype(np.float32)
            feats[start:end] += z
            counts[start:end] += 1
            if end == T:
                break
            start += step
        counts = np.maximum(counts, 1)
        feats = feats / counts[:, None]
        return feats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt) from train.py")
    p.add_argument("--subject", required=True, help="Subject to extract features for (e.g. S1)")
    p.add_argument(
        "--stories-mode",
        choices=["all", "train", "heldout", "explicit"],
        default="all",
        help="Which stories to extract. 'all' = train + heldout from the ckpt.",
    )
    p.add_argument("--stories", nargs="*", default=None, help="Used when --stories-mode=explicit")

    p.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "brain_encoder_pretrain" / "features" / "default"),
    )

    # Local-compute-mode passthroughs
    p.add_argument("--local-compute-mode", action="store_true")
    p.add_argument(
        "--mounted-project-root",
        default="/ceph/behrens/ellie/language-decoding-expts",
    )
    p.add_argument(
        "--local-cache-root",
        default=str(REPO_DIR / "local_cache"),
    )

    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    # Passthroughs expected by run_summaries_encoding helpers (unused here but
    # set so configure_local_compute_mode does not fail when reassigning).
    p.add_argument("--summaries-dir", default="")
    p.add_argument("--ba-dir", default="")
    return p.parse_args()


def main():
    args = parse_args()

    if args.local_compute_mode:
        try:
            rse.configure_local_compute_mode(args)
        except FileNotFoundError as e:
            log.warning("%s", e)
            log.warning("Falling back to local cache only (no mounted volume).")

    local_cache_root = Path(args.local_cache_root).expanduser().resolve()
    cached_base = local_cache_root / "data_train"
    response_root = str(cached_base) if cached_base.exists() else config.DATA_TRAIN_DIR

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    subject_to_voxels = cfg["subject_to_voxels"]
    if args.subject not in subject_to_voxels:
        raise ValueError(
            f"Subject {args.subject} not in checkpoint's subjects: {list(subject_to_voxels)}"
        )

    # Decide story list.
    if args.stories_mode == "explicit":
        if not args.stories:
            raise ValueError("--stories-mode=explicit requires --stories")
        stories = list(args.stories)
    elif args.stories_mode == "train":
        stories = list(cfg["train_stories"])
    elif args.stories_mode == "heldout":
        stories = list(cfg.get("heldout_stories", []))
    else:
        stories = list(cfg["train_stories"]) + list(cfg.get("heldout_stories", []))

    log.info("Extracting features for %s over %d stories", args.subject, len(stories))

    # Load data (z-scored per story, matching training normalization).
    subj_data = load_subject_data(args.subject, stories, response_root)

    # Build model + load weights.
    device = _resolve_device(args.device)
    model_cfg = cfg["model"]
    model = BrainEncoderMAE(
        subject_to_voxels=subject_to_voxels,
        d_model=model_cfg["d_model"],
        latent_dim=model_cfg.get("latent_dim", model_cfg["d_model"]),
        n_enc_layers=model_cfg["n_enc_layers"],
        n_dec_layers=model_cfg["n_dec_layers"],
        n_heads=model_cfg["n_heads"],
        ff_mult=model_cfg["ff_mult"],
        dropout=0.0,
        max_len=model_cfg["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_root = Path(args.output_dir).expanduser().resolve()
    subj_out = out_root / args.subject
    subj_out.mkdir(parents=True, exist_ok=True)

    # Save extraction metadata next to the features.
    meta = {
        "ckpt": str(ckpt_path),
        "subject": args.subject,
        "d_model": model_cfg["d_model"],
        "stories": list(subj_data.stories),
        "chunk_len": args.chunk_len,
        "overlap": args.overlap,
    }
    with open(subj_out / "_extraction_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Encode each story from the loaded concatenated data.
    X_full = subj_data.X
    for story, a, b in subj_data.story_spans:
        X_story = X_full[a:b]
        feats = encode_story_chunked(
            model, args.subject, X_story, device,
            chunk_len=args.chunk_len, overlap=args.overlap,
        )
        np.savez_compressed(subj_out / f"{story}.npz", X=feats)
        log.info("  %s: saved [%d, %d]", story, feats.shape[0], feats.shape[1])

    log.info("Done. Features under: %s", subj_out)


if __name__ == "__main__":
    main()
