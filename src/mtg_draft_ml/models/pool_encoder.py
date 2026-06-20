"""Permutation-invariant pool encoder.

See docs/architecture.md component 2; research/findings/set-architectures.md.

Phase 1 uses masked mean pooling (Deep Sets). Phase 2 will add a Set Transformer (SAB + PMA)
to model synergy; the interface (card_embs[B,L,d], mask[B,L]) -> context[B,d] stays the same.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolEncoder(nn.Module):
    """Masked mean over the pool. Empty pool (pick 0) -> zero vector."""

    def forward(self, card_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(card_embs.dtype)          # [B, L, 1]
        summed = (card_embs * m).sum(dim=1)                 # [B, d]
        denom = m.sum(dim=1).clamp_min(1.0)                 # [B, 1]
        return summed / denom
