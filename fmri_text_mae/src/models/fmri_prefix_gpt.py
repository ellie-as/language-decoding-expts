from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class FmriPrefixGPT(nn.Module):
    """Minimal scaffold for milestone 3: map fMRI windows into GPT soft-prefix tokens."""

    def __init__(self, n_features: int, fmri_window_len: int, decoder_model: str = "gpt2", n_prefix_tokens: int = 16, freeze_gpt: bool = True):
        super().__init__()
        self.gpt = AutoModelForCausalLM.from_pretrained(decoder_model)
        hidden = self.gpt.config.n_embd
        self.n_prefix_tokens = int(n_prefix_tokens)
        self.brain_encoder = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Flatten(start_dim=1),
            nn.Linear(fmri_window_len * hidden, self.n_prefix_tokens * hidden),
        )
        if freeze_gpt:
            for p in self.gpt.parameters():
                p.requires_grad = False

    def forward(self, bold: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        token_emb = self.gpt.get_input_embeddings()(input_ids)
        prefix = self.brain_encoder(bold).view(bold.shape[0], self.n_prefix_tokens, -1)
        inputs_embeds = torch.cat([prefix, token_emb], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones((bold.shape[0], self.n_prefix_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return self.gpt(inputs_embeds=inputs_embeds, attention_mask=full_mask)
