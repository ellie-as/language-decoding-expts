#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from losses import info_nce_loss, masked_bold_mse, masked_text_cross_entropy
    from models import FmriTextMAE, MaskingConfig
    from tokenization import load_tokenizer
    from utils import batch_to_device, ensure_dir, get_device, load_config, save_json, set_seed
    from window_dataset import FmriTextWindowDataset
else:
    from .losses import info_nce_loss, masked_bold_mse, masked_text_cross_entropy
    from .models import FmriTextMAE, MaskingConfig
    from .tokenization import load_tokenizer
    from .utils import batch_to_device, ensure_dir, get_device, load_config, save_json, set_seed
    from .window_dataset import FmriTextWindowDataset


def build_model(cfg, dataset, tokenizer) -> FmriTextMAE:
    masking = MaskingConfig(**cfg["masking"])
    return FmriTextMAE(
        n_features=dataset.n_features,
        vocab_size=len(tokenizer),
        max_bold_tokens=dataset.fmri_window_len_tr,
        max_text_tokens=int(cfg["text"]["max_text_tokens"]),
        pad_token_id=tokenizer.pad_token_id,
        masking=masking,
        **cfg["model"],
    )


def make_text_encoder(cfg, device):
    if float(cfg["loss"].get("lambda_contrastive", 0.0)) <= 0:
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg["text"]["embedding_model_for_contrastive"], device=str(device))


def sentence_embeddings(text_encoder, texts, device):
    if text_encoder is None:
        return None
    emb = text_encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return emb.to(device)


def step(model, batch, cfg, device, text_encoder=None, contrastive_projector=None, train=True):
    batch = batch_to_device(batch, device)
    out = model(batch["bold"], batch["input_ids"], batch["attention_mask"])
    text_ce = masked_text_cross_entropy(out["text_logits"], batch["input_ids"], out["text_mask"])
    bold_mse = masked_bold_mse(out["bold_logits"], batch["bold"], out["bold_mask"])
    contrastive = text_ce.new_tensor(0.0)
    if text_encoder is not None and out["mode"] == "fmri_to_text":
        z_brain = out["text_hidden"].mean(dim=1)
        if contrastive_projector is not None:
            z_brain = contrastive_projector(z_brain)
        z_text = sentence_embeddings(text_encoder, batch["text"], device)
        if z_text is not None:
            contrastive = info_nce_loss(z_brain, z_text, float(cfg["loss"]["temperature"]))
    loss = (
        float(cfg["loss"]["lambda_text_ce"]) * text_ce
        + float(cfg["loss"]["lambda_bold_mse"]) * bold_mse
        + float(cfg["loss"].get("lambda_contrastive", 0.0)) * contrastive
    )
    if train:
        loss.backward()
    return {
        "loss": float(loss.detach().cpu()),
        "text_ce": float(text_ce.detach().cpu()),
        "bold_mse": float(bold_mse.detach().cpu()),
        "contrastive": float(contrastive.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device, text_encoder=None, contrastive_projector=None):
    model.eval()
    totals = {"loss": 0.0, "text_ce": 0.0, "bold_mse": 0.0, "contrastive": 0.0}
    n = 0
    for batch in loader:
        metrics = step(model, batch, cfg, device, text_encoder=text_encoder, contrastive_projector=contrastive_projector, train=False)
        for key, value in metrics.items():
            totals[key] += value
        n += 1
    return {key: value / max(n, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train masked fMRI/text multimodal autoencoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["training"].get("random_seed", 42)))
    device = get_device(args.device)
    out_dir = ensure_dir(cfg["data"]["output_dir"])

    tokenizer = load_tokenizer(cfg["text"]["tokenizer"])
    train_ds = FmriTextWindowDataset(cfg["data"]["train_npz"], tokenizer, cfg["text"]["max_text_tokens"])
    val_ds = FmriTextWindowDataset(cfg["data"]["val_npz"], tokenizer, cfg["text"]["max_text_tokens"])
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"].get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"].get("num_workers", 0)))

    model = build_model(cfg, train_ds, tokenizer).to(device)
    text_encoder = make_text_encoder(cfg, device)
    contrastive_projector = None
    params = list(model.parameters())
    if text_encoder is not None:
        text_dim = int(text_encoder.get_sentence_embedding_dimension())
        contrastive_projector = torch.nn.Linear(int(cfg["model"]["d_model"]), text_dim).to(device)
        params.extend(contrastive_projector.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    best = math.inf
    stale = 0
    history = []
    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        running = {"loss": 0.0, "text_ce": 0.0, "bold_mse": 0.0, "contrastive": 0.0}
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            metrics = step(model, batch, cfg, device, text_encoder=text_encoder, contrastive_projector=contrastive_projector, train=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for key, value in metrics.items():
                running[key] += value
        train_metrics = {f"train_{k}": v / max(len(train_loader), 1) for k, v in running.items()}
        val_metrics = {f"val_{k}": v for k, v in evaluate(model, val_loader, cfg, device, text_encoder=text_encoder, contrastive_projector=contrastive_projector).items()}
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)
        print(row)

        if val_metrics["val_loss"] < best:
            best = val_metrics["val_loss"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "contrastive_projector": None if contrastive_projector is None else contrastive_projector.state_dict(),
                "config": cfg,
            }, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(cfg["training"].get("patience", 8)):
                break
        torch.save({
            "model": model.state_dict(),
            "contrastive_projector": None if contrastive_projector is None else contrastive_projector.state_dict(),
            "config": cfg,
        }, out_dir / "last.pt")
    save_json({"history": history, "best_val_loss": best}, out_dir / "training_metrics.json")


if __name__ == "__main__":
    main()
