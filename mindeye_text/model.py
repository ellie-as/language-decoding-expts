"""MindEye-style decoder for text embeddings.

Architecture:
- One subject-specific ``Linear(n_vox_subj, latent_dim)`` per subject. This is
  the only place where each subject's voxel layout is seen.
- A shared backbone of ``n_blocks`` pre-norm residual MLP blocks operating in
  the ``latent_dim`` space (defaults to 4096-D, matching MindEyeV2).
- A shared ``LayerNorm + Linear`` head projects to the target text embedding
  dimension (e.g. 768 for ``gtr-base``).

Because all subjects share the backbone and head, training pools roughly
``n_subjects ×`` more samples for the heavy parameters.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + Drop(FC2(Drop(GELU(FC1(LN(x))))))``."""

    def __init__(self, dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class MindEyeText(nn.Module):
    """Subject-specific input projection + shared residual MLP + linear head."""

    def __init__(
        self,
        voxel_counts: Mapping[str, int],
        embed_dim: int,
        latent_dim: int = 4096,
        n_blocks: int = 4,
        dropout: float = 0.15,
        head_norm: bool = True,
    ) -> None:
        super().__init__()
        if not voxel_counts:
            raise ValueError("voxel_counts must be a non-empty {subject: n_vox} mapping.")
        self.subjects = list(voxel_counts.keys())
        self.voxel_counts = {str(k): int(v) for k, v in voxel_counts.items()}
        self.latent_dim = int(latent_dim)
        self.embed_dim = int(embed_dim)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)

        self.subject_proj = nn.ModuleDict(
            {
                str(subj): nn.Linear(int(n_vox), self.latent_dim)
                for subj, n_vox in voxel_counts.items()
            }
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(self.latent_dim, self.dropout) for _ in range(self.n_blocks)]
        )
        head_layers: list[nn.Module] = []
        if head_norm:
            head_layers.append(nn.LayerNorm(self.latent_dim))
        head_layers.append(nn.Linear(self.latent_dim, self.embed_dim))
        self.head = nn.Sequential(*head_layers)

    def encode(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        """Project a subject's voxel batch into the shared latent space."""
        if subject not in self.subject_proj:
            raise KeyError(f"Unknown subject {subject!r}; have {list(self.subject_proj)}")
        return self.subject_proj[subject](x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the shared backbone + head on already-projected latents."""
        return self.head(self.backbone(latent))

    def forward(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        return self.decode(self.encode(x, subject))


class MindEyeEncoding(nn.Module):
    """Shared text-to-brain latent backbone plus subject-specific encoding heads.

    This is the encoding-direction counterpart of :class:`MindEyeText`: text
    features enter a shared projection/backbone, then each subject has a
    private linear head into their selected voxel space.
    """

    def __init__(
        self,
        output_dims: Mapping[str, int],
        input_dim: int,
        latent_dim: int = 4096,
        n_blocks: int = 4,
        dropout: float = 0.15,
        input_norm: bool = True,
        head_norm: bool = True,
    ) -> None:
        super().__init__()
        if not output_dims:
            raise ValueError("output_dims must be a non-empty {subject: n_outputs} mapping.")
        self.subjects = list(output_dims.keys())
        self.output_dims = {str(k): int(v) for k, v in output_dims.items()}
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)

        input_layers: list[nn.Module] = []
        if input_norm:
            input_layers.append(nn.LayerNorm(self.input_dim))
        input_layers.append(nn.Linear(self.input_dim, self.latent_dim))
        self.text_proj = nn.Sequential(*input_layers)
        self.backbone = nn.Sequential(
            *[ResidualBlock(self.latent_dim, self.dropout) for _ in range(self.n_blocks)]
        )
        self.head_norm = nn.LayerNorm(self.latent_dim) if head_norm else nn.Identity()
        self.subject_heads = nn.ModuleDict(
            {
                str(subj): nn.Linear(self.latent_dim, int(n_outputs))
                for subj, n_outputs in output_dims.items()
            }
        )

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        """Project delayed text features into the shared brain latent space."""
        return self.backbone(self.text_proj(x))

    def predict_from_latent(self, latent: torch.Tensor, subject: str) -> torch.Tensor:
        """Run a subject-specific encoding head on shared latents."""
        if subject not in self.subject_heads:
            raise KeyError(f"Unknown subject {subject!r}; have {list(self.subject_heads)}")
        return self.subject_heads[subject](self.head_norm(latent))

    def forward(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        return self.predict_from_latent(self.encode_text(x), subject)


def info_nce_clip(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    story_ids: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Symmetric InfoNCE on (B, D) batches using L2-normalized embeddings.

    When ``story_ids`` is provided (integer tensor of shape ``(B,)``), all
    off-diagonal positions where ``story_ids[i] == story_ids[j]`` are masked
    to ``-inf`` before the softmax.  This prevents the model from being
    penalised for correctly ranking a same-story negative highly — a common
    false-negative problem with temporally overlapping 5-TR windows.
    """
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    logits = (p @ t.t()) / max(float(temperature), 1e-6)
    if story_ids is not None:
        eye = torch.eye(p.shape[0], dtype=torch.bool, device=p.device)
        same = story_ids.unsqueeze(1) == story_ids.unsqueeze(0)   # (B, B)
        mask = same & ~eye   # same-story, off-diagonal
        logits = logits.masked_fill(mask, float("-inf"))
    labels = torch.arange(p.shape[0], device=p.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_p2t + loss_t2p)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    cosine_weight: float = 0.5,
    clip_weight: float = 0.5,
    clip_temp: float = 0.05,
    story_ids: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Multi-flavour decoding loss.

    Supported kinds:
      ``mse``           - plain mean squared error.
      ``cosine``        - mean of (1 - cos sim) on L2-normalized vectors.
      ``mse_cosine``    - convex combination of MSE and cosine, weighted by
                          ``cosine_weight``.
      ``mse_clip``      - convex combination of MSE and a symmetric InfoNCE
                          (CLIP-style) term across the batch.
    """
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "cosine":
        p = F.normalize(pred, dim=-1)
        t = F.normalize(target, dim=-1)
        return (1.0 - (p * t).sum(dim=-1)).mean()
    if kind == "mse_cosine":
        p = F.normalize(pred, dim=-1)
        t = F.normalize(target, dim=-1)
        cos = (1.0 - (p * t).sum(dim=-1)).mean()
        mse = F.mse_loss(pred, target)
        return (1.0 - float(cosine_weight)) * mse + float(cosine_weight) * cos
    if kind == "mse_clip":
        mse = F.mse_loss(pred, target)
        clip = info_nce_clip(pred, target, clip_temp, story_ids=story_ids)
        return (1.0 - float(clip_weight)) * mse + float(clip_weight) * clip
    raise ValueError(f"Unknown loss kind: {kind!r}")
