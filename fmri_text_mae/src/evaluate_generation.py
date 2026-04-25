#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_retrieval import retrieval_metrics
    from losses import masked_text_cross_entropy
    from models import FmriTextMAE, MaskingConfig
    from tokenization import decode_token_ids, load_tokenizer
    from utils import batch_to_device, ensure_dir, get_device, load_config, save_json
    from window_dataset import FmriTextWindowDataset
else:
    from .evaluate_retrieval import retrieval_metrics
    from .losses import masked_text_cross_entropy
    from .models import FmriTextMAE, MaskingConfig
    from .tokenization import decode_token_ids, load_tokenizer
    from .utils import batch_to_device, ensure_dir, get_device, load_config, save_json
    from .window_dataset import FmriTextWindowDataset


def build_model(cfg, dataset, tokenizer):
    return FmriTextMAE(
        n_features=dataset.n_features,
        vocab_size=len(tokenizer),
        max_bold_tokens=dataset.fmri_window_len_tr,
        max_text_tokens=int(cfg["text"]["max_text_tokens"]),
        pad_token_id=tokenizer.pad_token_id,
        masking=MaskingConfig(**cfg["masking"]),
        **cfg["model"],
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fMRI-only masked-token text decoding.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    out_dir = ensure_dir(Path(cfg["data"]["output_dir"]) / f"eval_{args.split}")
    tokenizer = load_tokenizer(cfg["text"]["tokenizer"])
    dataset = FmriTextWindowDataset(cfg["data"][f"{args.split}_npz"], tokenizer, cfg["text"]["max_text_tokens"])
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    model = build_model(cfg, dataset, tokenizer).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_true, all_pred = [], []
    ce_sum, tok_correct, tok_total = 0.0, 0, 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch["bold"], batch["input_ids"], batch["attention_mask"], mask_mode="fmri_to_text")
        text_mask = out["text_mask"]
        ce_sum += float(masked_text_cross_entropy(out["text_logits"], batch["input_ids"], text_mask).cpu())
        pred_ids = out["text_logits"].argmax(dim=-1)
        valid = batch["attention_mask"].bool()
        tok_correct += int(((pred_ids == batch["input_ids"]) & valid).sum().cpu())
        tok_total += int(valid.sum().cpu())
        for true_ids, pred, mask in zip(batch["input_ids"].cpu(), pred_ids.cpu(), valid.cpu()):
            all_true.append(decode_token_ids(tokenizer, true_ids[mask].tolist()))
            all_pred.append(decode_token_ids(tokenizer, pred[mask].tolist()))

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(cfg["text"]["embedding_model_for_contrastive"], device=str(device))
    true_emb = encoder.encode(all_true, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    pred_emb = encoder.encode(all_pred, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    metrics = retrieval_metrics(true_emb, pred_emb)
    metrics.update({
        "text_cross_entropy": ce_sum / max(len(loader), 1),
        "perplexity": float(math.exp(min(ce_sum / max(len(loader), 1), 20.0))),
        "token_accuracy": float(tok_correct / max(tok_total, 1)),
    })
    save_json(metrics, out_dir / "metrics.json")

    with open(out_dir / "qualitative_generations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_text", "model_generation"])
        writer.writeheader()
        for true_text, pred_text in list(zip(all_true, all_pred))[: args.max_rows]:
            writer.writerow({"true_text": true_text, "model_generation": pred_text})
    print(metrics)


if __name__ == "__main__":
    main()
