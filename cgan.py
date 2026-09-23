from __future__ import annotations

import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes=7, z_dim=128, embed_dim=64, img_channels=3, base=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.project = nn.Sequential(
            nn.Linear(z_dim + embed_dim, base * 8 * 4 * 4),
            nn.BatchNorm1d(base * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.project(x).view(x.size(0), -1, 4, 4)
        return self.blocks(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=7, embed_dim=64, img_channels=3, base=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, base*8, 4, 2, 1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.class_embed = nn.Linear(embed_dim, base * 8)
        self.head = nn.Conv2d(base*8, 1, 4, 1, 0)

    def forward(self, x, labels):
        f = self.features(x)
        e = self.class_embed(self.label_emb(labels)).view(labels.size(0), -1, 1, 1)
        f = f + e
        return self.head(f).view(-1)
