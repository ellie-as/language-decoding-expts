#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import flatten_bold_windows
    from evaluate_retrieval import retrieval_metrics
    from utils import ensure_dir, load_config, resolve_path, save_json, set_seed
else:
    from .datasets import flatten_bold_windows
    from .evaluate_retrieval import retrieval_metrics
    from .utils import ensure_dir, load_config, resolve_path, save_json, set_seed


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def embed_texts(model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    return encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def fit_mlp(x_train, y_train, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(x_train.shape[1], y_train.shape[1], int(cfg["mlp_hidden_dim"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["mlp_lr"]), weight_decay=float(cfg["mlp_weight_decay"]))
    ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    dl = DataLoader(ds, batch_size=int(cfg["mlp_batch_size"]), shuffle=True)
    model.train()
    for _ in range(int(cfg["mlp_epochs"])):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return model.cpu().eval()


def predict_mlp(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x).float()).numpy().astype(np.float32)


def zscore_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return np.nan_to_num((x_train - mean) / std).astype(np.float32), np.nan_to_num((x_test - mean) / std).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 1: fMRI to text-window embedding retrieval.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["evaluation"].get("random_seed", 42)))
    out_dir = ensure_dir(cfg["data"]["output_dir"])

    x_train, _, train_texts = flatten_bold_windows(cfg["data"]["train_npz"])
    x_test, _, test_texts = flatten_bold_windows(cfg["data"]["test_npz"])
    x_train, x_test = zscore_train_test(x_train, x_test)

    emb_model = cfg["text"]["embedding_model"]
    batch_size = int(cfg["text"].get("embedding_batch_size", 64))
    y_train = embed_texts(emb_model, train_texts, batch_size)
    y_test = embed_texts(emb_model, test_texts, batch_size)

    emb_mean, emb_std = y_train.mean(axis=0), y_train.std(axis=0)
    emb_std[emb_std == 0] = 1.0
    y_train_z = np.nan_to_num((y_train - emb_mean) / emb_std).astype(np.float32)

    if cfg["model"]["kind"] == "ridge":
        model = Ridge(alpha=float(cfg["model"].get("ridge_alpha", 100.0)))
        model.fit(x_train, y_train_z)
        pred = model.predict(x_test).astype(np.float32) * emb_std + emb_mean
    elif cfg["model"]["kind"] == "mlp":
        model = fit_mlp(x_train, y_train_z, cfg["model"])
        pred = predict_mlp(model, x_test) * emb_std + emb_mean
        torch.save(model.state_dict(), out_dir / "mlp.pt")
    else:
        raise ValueError(f"Unknown model kind: {cfg['model']['kind']}")

    metrics = {"model": retrieval_metrics(y_test, pred)}
    if cfg["evaluation"].get("shuffled_baseline", True):
        rng = np.random.default_rng(int(cfg["evaluation"].get("random_seed", 42)))
        shuffled = pred[rng.permutation(pred.shape[0])]
        metrics["shuffled_pred_baseline"] = retrieval_metrics(y_test, shuffled)

    np.savez_compressed(out_dir / "predictions.npz", pred=pred, true=y_test, texts=np.asarray(test_texts, dtype=object))
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
