from __future__ import annotations

import numpy as np

from .utils import batched_indices


class BatchStandardizer:
    """Feature standardizer that can be fit from batches of flattened MEG windows."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, indices: np.ndarray, batch_size: int = 256) -> "BatchStandardizer":
        n_features = int(np.prod(x.shape[1:]))
        total = np.zeros(n_features, dtype=np.float64)
        total_sq = np.zeros(n_features, dtype=np.float64)
        n = 0
        for start, stop in batched_indices(len(indices), batch_size):
            batch = np.asarray(x[indices[start:stop]]).reshape(stop - start, -1).astype(np.float64)
            total += batch.sum(axis=0)
            total_sq += np.square(batch).sum(axis=0)
            n += batch.shape[0]
        self.mean_ = (total / max(n, 1)).astype(np.float32)
        var = total_sq / max(n, 1) - np.square(self.mean_.astype(np.float64))
        self.scale_ = np.sqrt(np.maximum(var, self.eps)).astype(np.float32)
        return self

    def transform_batch(self, batch: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("BatchStandardizer is not fitted")
        flat = batch.reshape(batch.shape[0], -1).astype(np.float32)
        return (flat - self.mean_) / self.scale_

    def transform_indices(self, x: np.ndarray, indices: np.ndarray, batch_size: int = 256):
        for start, stop in batched_indices(len(indices), batch_size):
            yield self.transform_batch(np.asarray(x[indices[start:stop]]))
