"""Card encoder: content vector ([features ; text embedding]) -> shared MLP -> card embedding.

See docs/architecture.md component 1; design-decisions.md DD-001.

The encoder is ID-free: it sees only content, so an unseen card is encoded like any other. The
same encoder is applied to pool cards and pack cards.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CardEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256,
                 layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        blocks: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x[..., in_dim] -> [..., out_dim]; arbitrary leading dims are preserved."""
        return self.proj(self.body(x))
