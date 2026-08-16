from __future__ import annotations

import torch
import torch.nn as nn


class FractureCNN(nn.Module):
    def __init__(self, n_classes: int = 7) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.cls = nn.Linear(64 * 4 * 4, n_classes)
        self.box = nn.Linear(64 * 4 * 4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).flatten(1)
        return self.cls(feat), torch.sigmoid(self.box(feat))
