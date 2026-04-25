from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


MaskMode = Literal["fmri_mask", "text_mask", "fmri_to_text", "text_to_fmri", "both_masked"]


@dataclass
class MaskingConfig:
    p_fmri_mask: float = 0.20
    p_text_mask: float = 0.20
    p_fmri_to_text: float = 0.35
    p_text_to_fmri: float = 0.15
    p_both_masked: float = 0.10
    text_mask_ratio: float = 0.40
    fmri_mask_ratio: float = 0.40

    @property
    def modes(self) -> list[MaskMode]:
        return ["fmri_mask", "text_mask", "fmri_to_text", "text_to_fmri", "both_masked"]

    @property
    def probs(self) -> torch.Tensor:
        probs = torch.tensor([
            self.p_fmri_mask,
            self.p_text_mask,
            self.p_fmri_to_text,
            self.p_text_to_fmri,
            self.p_both_masked,
        ], dtype=torch.float32)
        return probs / probs.sum()


class FmriTextMAE(nn.Module):
    def __init__(
        self,
        n_features: int,
        vocab_size: int,
        max_bold_tokens: int,
        max_text_tokens: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        masking: MaskingConfig | None = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.vocab_size = int(vocab_size)
        self.max_bold_tokens = int(max_bold_tokens)
        self.max_text_tokens = int(max_text_tokens)
        self.pad_token_id = int(pad_token_id)
        self.masking = masking or MaskingConfig()

        self.bold_proj = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.bold_pos = nn.Embedding(max_bold_tokens, d_model)
        self.text_pos = nn.Embedding(max_text_tokens, d_model)
        self.modality_embedding = nn.Embedding(2, d_model)
        self.bold_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.text_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.bold_head = nn.Linear(d_model, n_features)
        self.text_head = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.bold_mask_token, std=0.02)
        nn.init.normal_(self.text_mask_token, std=0.02)

    def sample_mode(self, device: torch.device) -> MaskMode:
        idx = torch.multinomial(self.masking.probs.to(device), 1).item()
        return self.masking.modes[idx]

    def _random_mask(self, valid: torch.Tensor, ratio: float) -> torch.Tensor:
        random_scores = torch.rand(valid.shape, device=valid.device)
        candidate = random_scores < float(ratio)
        mask = candidate & valid.bool()
        empty = mask.sum(dim=1) == 0
        if empty.any():
            first_valid = valid.float().argmax(dim=1)
            rows = torch.nonzero(empty & valid.any(dim=1), as_tuple=False).squeeze(-1)
            if rows.numel():
                mask[rows, first_valid[rows]] = True
        return mask

    def make_masks(
        self,
        bold: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: MaskMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, MaskMode]:
        mode = mode or self.sample_mode(bold.device)
        bold_valid = torch.ones(bold.shape[:2], dtype=torch.bool, device=bold.device)
        text_valid = attention_mask.bool()
        bold_mask = torch.zeros_like(bold_valid)
        text_mask = torch.zeros_like(text_valid)

        if mode == "fmri_mask":
            bold_mask = self._random_mask(bold_valid, self.masking.fmri_mask_ratio)
        elif mode == "text_mask":
            text_mask = self._random_mask(text_valid, self.masking.text_mask_ratio)
        elif mode == "fmri_to_text":
            text_mask = text_valid.clone()
        elif mode == "text_to_fmri":
            bold_mask = bold_valid.clone()
        elif mode == "both_masked":
            bold_mask = self._random_mask(bold_valid, self.masking.fmri_mask_ratio)
            text_mask = self._random_mask(text_valid, self.masking.text_mask_ratio)
        else:
            raise ValueError(f"Unknown masking mode: {mode}")
        return bold_mask, text_mask, mode

    def encode(
        self,
        bold: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_mode: MaskMode | None = None,
        force_text_mask: torch.Tensor | None = None,
        force_bold_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str]:
        bsz, n_bold, _ = bold.shape
        _, n_text = input_ids.shape
        bold_mask, text_mask, mode = self.make_masks(bold, attention_mask, mask_mode)
        if force_text_mask is not None:
            text_mask = force_text_mask.bool() & attention_mask.bool()
        if force_bold_mask is not None:
            bold_mask = force_bold_mask.bool()

        bold_tokens = self.bold_proj(bold)
        text_tokens = self.text_embedding(input_ids)
        bold_tokens = torch.where(bold_mask.unsqueeze(-1), self.bold_mask_token.expand_as(bold_tokens), bold_tokens)
        text_tokens = torch.where(text_mask.unsqueeze(-1), self.text_mask_token.expand_as(text_tokens), text_tokens)

        bold_pos = self.bold_pos(torch.arange(n_bold, device=bold.device)).unsqueeze(0)
        text_pos = self.text_pos(torch.arange(n_text, device=bold.device)).unsqueeze(0)
        bold_mod = self.modality_embedding(torch.zeros(n_bold, dtype=torch.long, device=bold.device)).unsqueeze(0)
        text_mod = self.modality_embedding(torch.ones(n_text, dtype=torch.long, device=bold.device)).unsqueeze(0)

        tokens = torch.cat([bold_tokens + bold_pos + bold_mod, text_tokens + text_pos + text_mod], dim=1)
        key_padding = torch.cat([
            torch.zeros((bsz, n_bold), dtype=torch.bool, device=bold.device),
            ~attention_mask.bool(),
        ], dim=1)
        hidden = self.norm(self.encoder(tokens, src_key_padding_mask=key_padding))
        bold_hidden = hidden[:, :n_bold]
        text_hidden = hidden[:, n_bold:]
        return {
            "bold_hidden": bold_hidden,
            "text_hidden": text_hidden,
            "bold_logits": self.bold_head(bold_hidden),
            "text_logits": self.text_head(text_hidden),
            "bold_mask": bold_mask,
            "text_mask": text_mask,
            "mode": mode,
        }

    def forward(self, bold: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask_mode: MaskMode | None = None):
        return self.encode(bold, input_ids, attention_mask, mask_mode=mask_mode)
