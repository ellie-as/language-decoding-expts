from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import l2_normalize


def compute_text_embeddings(windows: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    emb_cfg = config["embeddings"]
    if emb_cfg.get("mock"):
        return mock_embeddings(windows["text"].tolist(), int(emb_cfg.get("mock_dim", 768)), bool(emb_cfg.get("normalize", True)))

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(emb_cfg["model_name"], device=device)
    embeddings = model.encode(
        windows["text"].tolist(),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        convert_to_numpy=True,
        normalize_embeddings=bool(emb_cfg.get("normalize", True)),
        show_progress_bar=True,
    ).astype(np.float32)
    if emb_cfg.get("normalize", True):
        embeddings = l2_normalize(embeddings).astype(np.float32)
    return embeddings


def mock_embeddings(texts: list[str], dim: int = 768, normalize: bool = True) -> np.ndarray:
    arr = np.empty((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(tqdm(texts, desc="mock embeddings")):
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        arr[i] = rng.normal(size=dim).astype(np.float32)
    return l2_normalize(arr).astype(np.float32) if normalize else arr


def embedding_sanity_checks(embeddings: np.ndarray, windows: pd.DataFrame) -> dict[str, Any]:
    norms = np.linalg.norm(embeddings, axis=1)
    summary: dict[str, Any] = {
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "norm_mean": float(norms.mean()) if len(norms) else None,
        "norm_min": float(norms.min()) if len(norms) else None,
        "norm_max": float(norms.max()) if len(norms) else None,
    }
    if len(embeddings) >= 4:
        sims = embeddings @ embeddings.T
        nearby = []
        random_sims = []
        for run, idx in windows.groupby("run_group").indices.items():
            indices = np.array(idx)
            if len(indices) > 1:
                nearby.extend(sims[indices[:-1], indices[1:]].tolist())
        rng = np.random.default_rng(0)
        for _ in range(min(1000, len(embeddings))):
            i, j = rng.choice(len(embeddings), size=2, replace=False)
            random_sims.append(float(sims[i, j]))
        summary["nearby_cosine_mean"] = float(np.mean(nearby)) if nearby else None
        summary["random_cosine_mean"] = float(np.mean(random_sims)) if random_sims else None
    return summary
