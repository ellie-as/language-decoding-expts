from __future__ import annotations

import numpy as np


def shuffled_pair_indices(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def apply_shuffled_text_control(input_ids, attention_mask, seed: int = 42):
    idx = shuffled_pair_indices(input_ids.shape[0], seed)
    return input_ids[idx], attention_mask[idx]


def wrong_lag_window_path(base_npz_path: str, lag_label: str = "wrong_lag") -> str:
    return base_npz_path.replace(".npz", f"_{lag_label}.npz")
