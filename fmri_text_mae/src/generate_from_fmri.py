#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_generation import build_model
    from tokenization import decode_token_ids, load_tokenizer
    from utils import batch_to_device, get_device, load_config
    from window_dataset import FmriTextWindowDataset
else:
    from .evaluate_generation import build_model
    from .tokenization import decode_token_ids, load_tokenizer
    from .utils import batch_to_device, get_device, load_config
    from .window_dataset import FmriTextWindowDataset


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Print fMRI-only masked-token generations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)
    tokenizer = load_tokenizer(cfg["text"]["tokenizer"])
    dataset = FmriTextWindowDataset(cfg["data"][f"{args.split}_npz"], tokenizer, cfg["text"]["max_text_tokens"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = build_model(cfg, dataset, tokenizer).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    for idx, batch in enumerate(loader):
        if idx >= args.n:
            break
        batch = batch_to_device(batch, device)
        out = model(batch["bold"], batch["input_ids"], batch["attention_mask"], mask_mode="fmri_to_text")
        pred_ids = out["text_logits"].argmax(dim=-1)[0].cpu()
        mask = batch["attention_mask"][0].cpu().bool()
        true_text = batch["text"][0]
        pred_text = decode_token_ids(tokenizer, pred_ids[mask].tolist())
        print(f"[{idx}] true: {true_text}\n[{idx}] pred: {pred_text}\n")


if __name__ == "__main__":
    main()
