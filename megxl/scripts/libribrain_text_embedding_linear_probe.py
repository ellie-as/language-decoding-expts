#!/usr/bin/env python3
"""Probe MEG-XL features against LibriBrain text-window embeddings.

This script expects the MEG-XL repository to be checked out at ../MEG-XL relative
to this file. It downloads the public MEG-XL checkpoint from Hugging Face when
needed, builds word-aligned LibriBrain windows, embeds each text window with a
Hugging Face text encoder, and trains a projection on top of MEG-XL features.
By default MEG-XL is frozen; pass --finetune-backbone to update the MEG-XL
transformer while keeping the BioCodec tokenizer frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
MEGXL_REPO = WORK_DIR / "MEG-XL"
if not MEGXL_REPO.exists():
    raise FileNotFoundError(
        f"Expected MEG-XL checkout at {MEGXL_REPO}. "
        "Run: git clone https://github.com/neural-processing-lab/MEG-XL megxl/MEG-XL"
    )
sys.path.insert(0, str(MEGXL_REPO))

from brainstorm.data.libribrain_word_aligned_dataset import LibriBrainWordAlignedDataset  # noqa: E402
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule  # noqa: E402
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel  # noqa: E402


LOG = logging.getLogger("libribrain_text_embedding_probe")


@dataclass
class RunConfig:
    libribrain_root: str
    output_dir: str
    cache_dir: str
    checkpoint_dir: str
    megxl_repo: str
    hf_model_repo: str
    hf_checkpoint_filename: str
    text_model_name: str
    subjects: List[str] | None
    sessions: List[str] | None
    tasks: List[str] | None
    words_per_segment: int
    subsegment_duration: float
    window_onset_offset: float
    target_sfreq: float
    l_freq: float
    h_freq: float
    max_channel_dim: int
    train_ratio: float
    val_ratio: float
    seed: int
    max_segments: int | None
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    backbone_lr: float
    weight_decay: float
    grad_clip: float
    finetune_backbone: bool
    pooling: str
    loss: str
    temperature: float
    eval_max_batches: int | None
    device: str
    text_device: str
    amp: bool


def parse_csv(values: str | None) -> List[str] | None:
    if values is None or values.strip() == "":
        return None
    return [v.strip() for v in values.split(",") if v.strip()]


def text_hash(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return (int(digest, 16) % 1_000_000) / 1_000_000.0


def segment_text(dataset: LibriBrainWordAlignedDataset, idx: int) -> str:
    rec_idx, group_idx = dataset.segment_index[idx]
    words = [w["word"] for w in dataset.word_groups[rec_idx][group_idx]]
    return " ".join(str(w).strip() for w in words if str(w).strip())


def split_indices(
    dataset: LibriBrainWordAlignedDataset,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_segments: int | None,
) -> Tuple[List[int], List[int], List[int], Dict[int, str]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must be positive and sum to less than 1")

    all_indices = list(range(len(dataset)))
    if max_segments is not None:
        rng = np.random.default_rng(seed)
        all_indices = sorted(rng.choice(all_indices, size=min(max_segments, len(all_indices)), replace=False).tolist())

    idx_to_text = {idx: segment_text(dataset, idx) for idx in all_indices}
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for idx, text in idx_to_text.items():
        bucket = text_hash(text, seed)
        if bucket < train_ratio:
            train.append(idx)
        elif bucket < train_ratio + val_ratio:
            val.append(idx)
        else:
            test.append(idx)

    if (not train or not val or not test) and len(all_indices) >= 3:
        LOG.warning(
            "Hash split produced an empty split for %d examples; "
            "falling back to deterministic shuffled split for this run",
            len(all_indices),
        )
        rng = np.random.default_rng(seed)
        shuffled = list(all_indices)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = max(1, int(round(train_ratio * n_total)))
        n_val = max(1, int(round(val_ratio * n_total)))
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1
        train = sorted(shuffled[:n_train])
        val = sorted(shuffled[n_train : n_train + n_val])
        test = sorted(shuffled[n_train + n_val :])

    return train, val, test, idx_to_text


def require_hf_hub() -> Any:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub") from exc
    return hf_hub_download


def ensure_libribrain_metadata(data_root: Path) -> None:
    metadata_dir = data_root / "metadata"
    channels_path = metadata_dir / "channels.tsv"
    sensor_xyz_path = metadata_dir / "sensor_xyz.json"
    if channels_path.exists() and sensor_xyz_path.exists():
        return

    hf_hub_download = require_hf_hub()
    LOG.info("LibriBrain metadata missing under %s; downloading small metadata files", metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["metadata/channels.tsv", "metadata/sensor_xyz.json"]:
        hf_hub_download(
            repo_id="pnpl/LibriBrain",
            filename=filename,
            repo_type="dataset",
            local_dir=data_root,
        )


def write_megxl_sensor_json(data_root: Path, output_path: Path) -> None:
    ensure_libribrain_metadata(data_root)
    channels_path = data_root / "metadata" / "channels.tsv"
    sensor_xyz_path = data_root / "metadata" / "sensor_xyz.json"

    with sensor_xyz_path.open("r") as f:
        xyz_values = json.load(f)

    sensors = []
    meg_xyz_idx = 0
    with channels_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ch_name = row.get("name") or row.get("ch_name")
            if not ch_name or not ch_name.startswith("MEG"):
                continue
            if meg_xyz_idx >= len(xyz_values):
                break

            ch_type = (row.get("type") or row.get("kind") or "").lower()
            is_mag = "mag" in ch_type and "grad" not in ch_type
            coil_type = 3024 if is_mag else 3012

            pos = [float(v) for v in xyz_values[meg_xyz_idx][:3]]
            meg_xyz_idx += 1
            # MEG-XL expects MNE-style 12-value loc arrays. The public metadata
            # provides positions only, so use a fixed orientation as a lightweight
            # compatibility value for the frozen model's spatial embedding.
            loc = pos + [0.0, 0.0, 1.0] + [0.0, 1.0, 0.0] + [0.0, 0.0, 1.0]
            sensors.append({"ch_name": ch_name, "loc": loc, "coil_type": coil_type})

    if not sensors:
        raise RuntimeError(f"Could not build sensor metadata from {channels_path} and {sensor_xyz_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(sensors, f)
    LOG.info("Wrote MEG-XL-compatible sensor metadata: %s", output_path)


def prepare_libribrain_root(data_root: Path, cache_dir: Path) -> Path:
    """Return a root compatible with MEG-XL's LibriBrainWordAlignedDataset."""
    sensor_json = data_root / "meg_sensors_information.json"
    has_serialized = (data_root / "serialized").exists()
    if sensor_json.exists() and has_serialized:
        return data_root

    compat_root = cache_dir / "libribrain_megxl_compat"
    compat_root.mkdir(parents=True, exist_ok=True)

    compat_sensor_json = compat_root / "meg_sensors_information.json"
    if sensor_json.exists():
        if not compat_sensor_json.exists():
            compat_sensor_json.symlink_to(sensor_json)
    else:
        write_megxl_sensor_json(data_root, compat_sensor_json)

    compat_serialized = compat_root / "serialized"
    if not compat_serialized.exists():
        serialized_source = data_root / "serialized" if has_serialized else data_root
        compat_serialized.symlink_to(serialized_source, target_is_directory=True)

    LOG.info("Using MEG-XL-compatible LibriBrain root: %s", compat_root)
    return compat_root


def download_megxl_checkpoint(checkpoint_dir: Path, repo_id: str, filename: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / filename
    if target.exists():
        LOG.info("Using existing MEG-XL checkpoint: %s", target)
        return target

    hf_hub_download = require_hf_hub()
    LOG.info("Downloading %s/%s to %s", repo_id, filename, checkpoint_dir)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
    return Path(downloaded)


def load_biocodec_tokenizer(checkpoint_path: Path, device: torch.device) -> BioCodecModel:
    LOG.info("Loading BioCodec tokenizer: %s", checkpoint_path)
    tokenizer = BioCodecModel._get_optimized_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    tokenizer.load_state_dict(state)
    tokenizer.to(device)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    return tokenizer


def load_megxl_model(
    checkpoint_path: Path,
    tokenizer: BioCodecModel,
    device: torch.device,
    freeze_backbone: bool,
) -> CrissCrossTransformerModule:
    LOG.info("Loading MEG-XL checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CrissCrossTransformerModule(tokenizer=tokenizer, **checkpoint["hyper_parameters"])
    state = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if "rope_embedding_layer.rotate" not in key
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        LOG.warning("Unexpected checkpoint keys: %s", unexpected)
    LOG.info("Loaded MEG-XL with %d missing deterministic RoPE buffers", len(missing))
    model.to(device)
    for name, param in model.named_parameters():
        if name.startswith("tokenizer."):
            param.requires_grad_(False)
        else:
            param.requires_grad_(not freeze_backbone)
    if freeze_backbone:
        model.eval()
    else:
        model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("MEG-XL trainable parameters: %d", n_trainable)
    return model


def mean_pool_text(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def text_embedding_cache_path(cache_dir: Path, model_name: str, texts: Sequence[str]) -> Path:
    joined_hash = hashlib.sha256("\n".join(sorted(set(texts))).encode("utf-8")).hexdigest()[:16]
    model_slug = model_name.replace("/", "__")
    return cache_dir / "text_embeddings" / f"{model_slug}_{joined_hash}.pt"


def compute_text_embeddings(
    texts: Sequence[str],
    model_name: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    unique_texts = sorted(set(texts))
    cache_path = text_embedding_cache_path(cache_dir, model_name, unique_texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        LOG.info("Loading cached text embeddings: %s", cache_path)
        payload = torch.load(cache_path, map_location="cpu")
        return payload["embeddings"]

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers first: pip install transformers") from exc

    LOG.info("Downloading/loading text encoder: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings: Dict[str, torch.Tensor] = {}
    for start in tqdm(range(0, len(unique_texts), batch_size), desc="Embedding text windows"):
        batch_texts = unique_texts[start : start + batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            pooled = mean_pool_text(outputs.last_hidden_state, tokens["attention_mask"])
            pooled = F.normalize(pooled, dim=-1).cpu()
        for text, emb in zip(batch_texts, pooled):
            embeddings[text] = emb

    torch.save({"model_name": model_name, "embeddings": embeddings}, cache_path)
    return embeddings


class IndexedEmbeddingDataset(Dataset):
    def __init__(self, base: LibriBrainWordAlignedDataset, indices: Sequence[int], idx_to_text: Dict[int, str], text_embs: Dict[str, torch.Tensor]):
        self.base = base
        self.indices = list(indices)
        self.idx_to_text = idx_to_text
        self.text_embs = text_embs

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = self.indices[item]
        sample = self.base[idx]
        text = self.idx_to_text[idx]
        return {
            "meg": sample["meg"],
            "sensor_xyzdir": sample["sensor_xyzdir"],
            "sensor_types": sample["sensor_types"],
            "sensor_mask": sample["sensor_mask"],
            "target_embedding": self.text_embs[text],
            "text": text,
            "index": idx,
        }


def collate_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    min_time = min(s["meg"].shape[-1] for s in samples)
    return {
        "meg": torch.stack([s["meg"][..., :min_time] for s in samples]),
        "sensor_xyzdir": torch.stack([s["sensor_xyzdir"] for s in samples]),
        "sensor_types": torch.stack([s["sensor_types"] for s in samples]),
        "sensor_mask": torch.stack([s["sensor_mask"] for s in samples]),
        "target_embedding": torch.stack([s["target_embedding"] for s in samples]),
        "text": [s["text"] for s in samples],
        "index": torch.tensor([s["index"] for s in samples], dtype=torch.long),
    }


class WindowLinearProbe(nn.Module):
    def __init__(self, num_channels: int, latent_dim: int, output_dim: int, pooling: str):
        super().__init__()
        if pooling not in {"channel_mean", "flatten_channels"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.pooling = pooling
        input_dim = latent_dim if pooling == "channel_mean" else num_channels * latent_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        time_pooled = features.mean(dim=2)
        if self.pooling == "channel_mean":
            weights = sensor_mask.to(time_pooled.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (time_pooled * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = time_pooled.flatten(start_dim=1)
        return self.proj(pooled)


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def batch_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str, temperature: float) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))
    if loss_name == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    if loss_name == "contrastive":
        return contrastive_loss(pred, target, temperature)
    raise ValueError(f"Unknown loss: {loss_name}")


def forward_probe(
    megxl: CrissCrossTransformerModule,
    probe: WindowLinearProbe,
    batch: Dict[str, Any],
    device: torch.device,
    use_amp: bool,
    finetune_backbone: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    meg = batch["meg"].to(device, non_blocking=True)
    sensor_xyzdir = batch["sensor_xyzdir"].to(device, non_blocking=True)
    sensor_types = batch["sensor_types"].to(device, non_blocking=True)
    sensor_mask = batch["sensor_mask"].to(device, non_blocking=True)
    target = batch["target_embedding"].to(device, non_blocking=True)

    sensor_xyz = sensor_xyzdir[..., :3]
    sensor_abc = sensor_xyzdir[..., 3:]

    if finetune_backbone:
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            output = megxl(meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False)
            features = output["features"].float()
    else:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            output = megxl(meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False)
            features = output["features"].float()

    pred = probe(features, sensor_mask)
    return pred, target


@torch.no_grad()
def evaluate(
    megxl: CrissCrossTransformerModule,
    probe: WindowLinearProbe,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    temperature: float,
    use_amp: bool,
    finetune_backbone: bool,
    max_batches: int | None = None,
) -> Dict[str, float]:
    megxl.eval()
    probe.eval()
    losses: List[float] = []
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluate", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        pred, target = forward_probe(megxl, probe, batch, device, use_amp, finetune_backbone)
        loss = batch_loss(pred, target, loss_name, temperature)
        losses.append(float(loss.detach().cpu()))
        preds.append(F.normalize(pred.detach().cpu(), dim=-1))
        targets.append(F.normalize(target.detach().cpu(), dim=-1))

    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    sims = pred_all @ target_all.T
    ranks = sims.argsort(dim=1, descending=True)
    labels = torch.arange(sims.shape[0]).unsqueeze(1)
    top1 = (ranks[:, :1] == labels).any(dim=1).float().mean().item()
    top5 = (ranks[:, : min(5, sims.shape[1])] == labels).any(dim=1).float().mean().item()
    top10 = (ranks[:, : min(10, sims.shape[1])] == labels).any(dim=1).float().mean().item()
    paired_cosine = F.cosine_similarity(pred_all, target_all, dim=-1).mean().item()
    n_candidates = sims.shape[1]

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "n_eval_examples": float(n_candidates),
        "paired_cosine": paired_cosine,
        "retrieval_top1": top1,
        "retrieval_top5": top5,
        "retrieval_top10": top10,
        "random_top1": min(1.0, 1.0 / n_candidates),
        "random_top5": min(1.0, 5.0 / n_candidates),
        "random_top10": min(1.0, 10.0 / n_candidates),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libribrain-root", required=True, help="Path to LibriBrain dataset root")
    parser.add_argument("--output-dir", default=str(WORK_DIR / "outputs" / "libribrain_text_probe"))
    parser.add_argument("--cache-dir", default=str(WORK_DIR / "cache"))
    parser.add_argument("--checkpoint-dir", default=str(WORK_DIR / "checkpoints"))
    parser.add_argument("--hf-model-repo", default="pnpl/MEG-XL")
    parser.add_argument("--hf-checkpoint-filename", default="meg-xl-med.ckpt")
    parser.add_argument("--text-model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--subjects", default="sub-0", help="Comma-separated subject ids, or empty for all")
    parser.add_argument("--sessions", default="", help="Comma-separated sessions, or empty for all")
    parser.add_argument("--tasks", default="", help="Comma-separated LibriBrain tasks, or empty for all")
    parser.add_argument("--words-per-segment", type=int, default=5)
    parser.add_argument("--subsegment-duration", type=float, default=3.0)
    parser.add_argument("--window-onset-offset", type=float, default=-0.5)
    parser.add_argument("--target-sfreq", type=float, default=50.0)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--max-channel-dim", type=int, default=306)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for MEG-XL parameters when --finetune-backbone is set",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        help="Unfreeze MEG-XL transformer/backbone parameters; BioCodec tokenizer stays frozen",
    )
    parser.add_argument(
        "--pooling",
        choices=["channel_mean", "flatten_channels"],
        default="channel_mean",
        help="How to pool MEG-XL features before the projection head",
    )
    parser.add_argument("--loss", choices=["contrastive", "mse", "cosine"], default="contrastive")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="Limit validation batches evaluated after each epoch; final test still uses the full test split",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    text_device = torch.device(args.text_device if torch.cuda.is_available() or args.text_device == "cpu" else "cpu")

    cfg = RunConfig(
        libribrain_root=args.libribrain_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        megxl_repo=str(MEGXL_REPO),
        hf_model_repo=args.hf_model_repo,
        hf_checkpoint_filename=args.hf_checkpoint_filename,
        text_model_name=args.text_model_name,
        subjects=parse_csv(args.subjects),
        sessions=parse_csv(args.sessions),
        tasks=parse_csv(args.tasks),
        words_per_segment=args.words_per_segment,
        subsegment_duration=args.subsegment_duration,
        window_onset_offset=args.window_onset_offset,
        target_sfreq=args.target_sfreq,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        max_channel_dim=args.max_channel_dim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_segments=args.max_segments,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        finetune_backbone=args.finetune_backbone,
        pooling=args.pooling,
        loss=args.loss,
        temperature=args.temperature,
        eval_max_batches=args.eval_max_batches,
        device=str(device),
        text_device=str(text_device),
        amp=not args.no_amp,
    )

    output_dir = Path(cfg.output_dir)
    cache_dir = Path(cfg.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    checkpoint_path = download_megxl_checkpoint(
        Path(cfg.checkpoint_dir), cfg.hf_model_repo, cfg.hf_checkpoint_filename
    )
    tokenizer_path = MEGXL_REPO / "brainstorm" / "neuro_tokenizers" / "biocodec_ckpt.pt"
    tokenizer = load_biocodec_tokenizer(tokenizer_path, device)
    megxl = load_megxl_model(checkpoint_path, tokenizer, device, freeze_backbone=not cfg.finetune_backbone)

    LOG.info("Building LibriBrain dataset")
    libribrain_root = prepare_libribrain_root(Path(cfg.libribrain_root), cache_dir)
    dataset = LibriBrainWordAlignedDataset(
        data_root=str(libribrain_root),
        segment_length=cfg.words_per_segment * cfg.subsegment_duration,
        subsegment_duration=cfg.subsegment_duration,
        words_per_segment=cfg.words_per_segment,
        window_onset_offset=cfg.window_onset_offset,
        cache_dir=str(cache_dir / "libribrain_preproc"),
        subjects=cfg.subjects,
        sessions=cfg.sessions,
        tasks=cfg.tasks,
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        target_sfreq=cfg.target_sfreq,
        max_channel_dim=cfg.max_channel_dim,
    )

    train_idx, val_idx, test_idx, idx_to_text = split_indices(
        dataset, cfg.train_ratio, cfg.val_ratio, cfg.seed, cfg.max_segments
    )
    LOG.info("Split sizes: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))
    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError("One split is empty. Increase data size or adjust split ratios.")

    text_embs = compute_text_embeddings(
        [idx_to_text[i] for i in train_idx + val_idx + test_idx],
        cfg.text_model_name,
        cache_dir,
        text_device,
    )
    embed_dim = next(iter(text_embs.values())).numel()
    LOG.info("Text embedding dimension: %d", embed_dim)

    train_ds = IndexedEmbeddingDataset(dataset, train_idx, idx_to_text, text_embs)
    val_ds = IndexedEmbeddingDataset(dataset, val_idx, idx_to_text, text_embs)
    test_ds = IndexedEmbeddingDataset(dataset, test_idx, idx_to_text, text_embs)
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=cfg.loss == "contrastive", **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    probe = WindowLinearProbe(cfg.max_channel_dim, megxl.latent_dim, embed_dim, cfg.pooling).to(device)
    if cfg.finetune_backbone:
        backbone_params = [p for p in megxl.parameters() if p.requires_grad]
        optimizer = AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.lr},
                {"params": backbone_params, "lr": cfg.backbone_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
        clip_params = list(probe.parameters()) + backbone_params
    else:
        optimizer = AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        clip_params = list(probe.parameters())

    best_val = -float("inf")
    best_path = output_dir / "best_probe.pt"
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        if cfg.finetune_backbone:
            megxl.train()
        else:
            megxl.eval()
        probe.train()
        train_losses: List[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            pred, target = forward_probe(megxl, probe, batch, device, cfg.amp, cfg.finetune_backbone)
            loss = batch_loss(pred, target, cfg.loss, cfg.temperature)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(
            megxl,
            probe,
            val_loader,
            device,
            cfg.loss,
            cfg.temperature,
            cfg.amp,
            cfg.finetune_backbone,
            max_batches=cfg.eval_max_batches,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        LOG.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_cos=%.4f "
            "val_top1=%.4f val_top10=%.4f random_top10=%.4f n_eval=%.0f",
            epoch,
            row["train_loss"],
            val_metrics["loss"],
            val_metrics["paired_cosine"],
            val_metrics["retrieval_top1"],
            val_metrics["retrieval_top10"],
            val_metrics["random_top10"],
            val_metrics["n_eval_examples"],
        )

        if val_metrics["paired_cosine"] > best_val:
            best_val = val_metrics["paired_cosine"]
            torch.save(
                {
                    "probe_state_dict": probe.state_dict(),
                    "megxl_state_dict": megxl.state_dict() if cfg.finetune_backbone else None,
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "embed_dim": embed_dim,
                    "latent_dim": megxl.latent_dim,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    probe.load_state_dict(checkpoint["probe_state_dict"])
    if checkpoint.get("megxl_state_dict") is not None:
        missing, unexpected = megxl.load_state_dict(checkpoint["megxl_state_dict"], strict=False)
        if unexpected:
            LOG.warning("Unexpected keys while loading fine-tuned MEG-XL: %s", unexpected)
        if missing:
            LOG.warning("Missing keys while loading fine-tuned MEG-XL: %s", missing)
    test_metrics = evaluate(megxl, probe, test_loader, device, cfg.loss, cfg.temperature, cfg.amp, cfg.finetune_backbone)
    LOG.info("Final test metrics: %s", test_metrics)

    result = {"best_checkpoint": str(best_path), "history": history, "test_metrics": test_metrics}
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    LOG.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()
