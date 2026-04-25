from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_text_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.bool()
    if active.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[active], targets[active])


def masked_bold_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.bool()
    if active.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[active], target[active])


def info_nce_loss(z_brain: torch.Tensor, z_text: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z_brain = F.normalize(z_brain, dim=-1)
    z_text = F.normalize(z_text, dim=-1)
    logits = z_brain @ z_text.T / temperature
    target = torch.arange(z_brain.shape[0], device=z_brain.device)
    return F.cross_entropy(logits, target)
