from __future__ import annotations

import torch
import torch.nn as nn


class FmriToEmbeddingMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, bold_flat: torch.Tensor) -> torch.Tensor:
        return self.net(bold_flat)
