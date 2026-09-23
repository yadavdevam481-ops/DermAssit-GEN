from __future__ import annotations

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.30):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
