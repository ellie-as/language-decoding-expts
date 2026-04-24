"""MAE-style multi-subject brain encoder.

Components:
  - per-subject input projection: Linear(V_s, d_model)  (one per subject)
  - shared Transformer encoder over TR-tokens (with sinusoidal position)
  - shared Transformer decoder (lightweight) that fills in masked positions
  - per-subject output head: Linear(d_model, V_s)  (one per subject)

Training loss is MSE between the decoder's predictions and the original voxel
vectors at masked positions, scaled by 1/V_s so subjects contribute comparably.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def sinusoidal_position(max_len: int, d_model: int, device=None) -> torch.Tensor:
    """Standard fixed sinusoidal positional encoding [max_len, d_model]."""
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _SubjectProjection(nn.Module):
    """Linear map V_s -> d_model (input side)."""

    def __init__(self, n_voxels: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_voxels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _SubjectOutput(nn.Module):
    """Linear map d_model -> V_s (reconstruction head)."""

    def __init__(self, d_model: int, n_voxels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_voxels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class BrainEncoderMAE(nn.Module):
    """Masked-autoencoder on TR-level voxel tokens, shared across subjects."""

    def __init__(
        self,
        subject_to_voxels: Dict[str, int],
        d_model: int = 256,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Per-subject input / output heads (handle variable voxel counts).
        self.inputs = nn.ModuleDict(
            {s: _SubjectProjection(n, d_model) for s, n in subject_to_voxels.items()}
        )
        self.outputs = nn.ModuleDict(
            {s: _SubjectOutput(d_model, n) for s, n in subject_to_voxels.items()}
        )

        # Sinusoidal positional embedding (shared, registered as buffer).
        self.register_buffer(
            "pos_embed", sinusoidal_position(max_len, d_model), persistent=False
        )

        # Shared mask token for decoder inputs at masked positions.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_dec_layers)

    # ---------- public helpers ----------

    def subjects(self) -> List[str]:
        return list(self.inputs.keys())

    def encode(self, subject: str, x: torch.Tensor) -> torch.Tensor:
        """Encode a full sequence (no masking).

        Args:
            subject: subject id (must be in self.inputs)
            x: [B, L, V_s] voxel timeseries
        Returns:
            [B, L, d_model] contextualized features
        """
        if subject not in self.inputs:
            raise KeyError(f"Unknown subject head: {subject}")
        L = x.shape[1]
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.max_len}")
        tokens = self.inputs[subject](x)  # [B, L, d]
        tokens = tokens + self.pos_embed[:L].unsqueeze(0)
        return self.encoder(tokens)

    # ---------- training forward ----------

    @staticmethod
    def _random_mask(B: int, L: int, mask_ratio: float, device) -> torch.Tensor:
        """Return a boolean mask [B, L] where True = masked (to be predicted)."""
        n_mask = max(1, int(round(mask_ratio * L)))
        # Rank noise per sample; positions with smallest noise are "kept".
        noise = torch.rand(B, L, device=device)
        _, idx_shuffle = torch.sort(noise, dim=1)
        mask = torch.ones(B, L, device=device, dtype=torch.bool)
        keep_idx = idx_shuffle[:, n_mask:]
        mask.scatter_(1, keep_idx, False)
        return mask  # True where masked

    def forward_mae(
        self,
        subject: str,
        x: torch.Tensor,
        mask_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One MAE forward.

        Args:
            subject: subject id
            x: [B, L, V_s] z-scored voxel timeseries
            mask_ratio: fraction of TRs to mask per sample
        Returns:
            pred:   [B, L, V_s] predicted voxel values at masked positions
            target: [B, L, V_s] ground-truth voxel values (full x)
            mask:   [B, L] boolean, True where masked (to be predicted)
        """
        B, L, _ = x.shape
        device = x.device

        mask = self._random_mask(B, L, mask_ratio, device)

        # Project voxels -> tokens and add position.
        tokens_full = self.inputs[subject](x) + self.pos_embed[:L].unsqueeze(0)

        # Build visible-only sequence for the encoder.
        # We pass full tokens with an attention mask that hides masked positions.
        # Easier: zero out masked token embeddings and rely on attention to
        # ignore them. Use a key_padding_mask in the encoder.
        key_padding_mask = mask  # [B, L] True=ignore
        # Replace masked positions with mask token (any value is fine since they
        # are ignored by attention via key_padding_mask).
        tokens_enc_in = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand(B, L, -1),
            tokens_full,
        )
        enc_out = self.encoder(tokens_enc_in, src_key_padding_mask=key_padding_mask)

        # Decoder: at masked positions use [mask] + pos, at visible use enc_out.
        mask_tokens = self.mask_token.expand(B, L, -1) + self.pos_embed[:L].unsqueeze(0)
        dec_in = torch.where(mask.unsqueeze(-1), mask_tokens, enc_out)
        dec_out = self.decoder(dec_in)

        pred = self.outputs[subject](dec_out)  # [B, L, V_s]
        return pred, x, mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-voxel MSE averaged over masked TRs only.

    Dividing by V_s keeps the per-sample loss scale independent of voxel count,
    which matters when mixing subjects with very different V_s.
    """
    mask_f = mask.float().unsqueeze(-1)  # [B, L, 1]
    sq = (pred - target) ** 2
    numer = (sq * mask_f).sum(dim=(1, 2))  # [B]
    denom = mask_f.sum(dim=1).clamp_min(1.0) * pred.shape[-1]  # n_masked * V_s
    return (numer / denom.squeeze(-1)).mean()
