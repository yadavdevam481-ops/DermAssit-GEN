from __future__ import annotations
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([], dtype=torch.float32))

    def forward(self, logits, targets):
        weight = self.weight if self.weight.numel() else None
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=weight)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()
