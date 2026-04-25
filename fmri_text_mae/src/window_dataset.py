from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .utils import resolve_path
except ImportError:
    from utils import resolve_path


class FmriTextWindowDataset(Dataset):
    def __init__(self, npz_path: str | Path, tokenizer, max_text_tokens: int):
        data = np.load(resolve_path(npz_path), allow_pickle=True)
        self.bold = data["bold"].astype(np.float32)
        self.texts = [str(t) for t in data["text"].tolist()]
        self.story = [str(s) for s in data["story"].tolist()]
        self.start_tr = data["start_tr"].astype(np.int64)
        self.tokenizer = tokenizer
        self.max_text_tokens = int(max_text_tokens)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_features(self) -> int:
        return int(self.bold.shape[-1])

    @property
    def fmri_window_len_tr(self) -> int:
        return int(self.bold.shape[1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return {
            "bold": torch.from_numpy(self.bold[idx]),
            "input_ids": enc["input_ids"].squeeze(0).long(),
            "attention_mask": enc["attention_mask"].squeeze(0).bool(),
            "text": self.texts[idx],
            "story": self.story[idx],
            "start_tr": int(self.start_tr[idx]),
        }


def flatten_bold_windows(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(resolve_path(npz_path), allow_pickle=True)
    bold = data["bold"].astype(np.float32)
    texts = [str(t) for t in data["text"].tolist()]
    story = np.asarray(data["story"], dtype=str)
    return bold.reshape(bold.shape[0], -1), story, texts
