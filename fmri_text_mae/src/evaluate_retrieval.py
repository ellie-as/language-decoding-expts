from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def retrieval_metrics(true_emb: np.ndarray, pred_emb: np.ndarray) -> dict[str, float]:
    true_unit = l2_normalize(true_emb.astype(np.float32, copy=False))
    pred_unit = l2_normalize(pred_emb.astype(np.float32, copy=False))
    sim = pred_unit @ true_unit.T
    diag = np.diag(sim)
    ranks = 1 + (sim > diag[:, None]).sum(axis=1)
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "embedding_cosine": float(np.mean(diag)),
    }


def cosine_per_row(true_emb: np.ndarray, pred_emb: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize(true_emb) * l2_normalize(pred_emb), axis=1)
