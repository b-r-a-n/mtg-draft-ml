"""Phase-0 baseline: one-hot MLP over the collection, scoring the pack (masked softmax).

See docs/roadmap.md Phase 0 and design-decisions.md DD-001 (this is the *baseline* we measure
against — a fixed-vocabulary model that CANNOT generalize to unseen cards; the content encoder
in Phase 1 replaces it).

Input is the pool (collection) as a fixed-width count vector. The MLP outputs one logit per card
in the set's vocabulary; logits for cards not in the current pack are masked to -inf before the
softmax. This is the Statistical-Drafting / Draftsim "NNetBot" style baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OneHotMLP(nn.Module):
    def __init__(self, n_cards: int, hidden: int = 512, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_cards = n_cards
        blocks: list[nn.Module] = []
        in_dim = n_cards
        for _ in range(layers):
            blocks += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(in_dim, n_cards)

    def forward(self, pool: torch.Tensor, pack_mask: torch.Tensor) -> torch.Tensor:
        """pool[B, n_cards] float, pack_mask[B, n_cards] bool -> logits[B, n_cards] (masked)."""
        logits = self.head(self.body(pool))
        return logits.masked_fill(~pack_mask, float("-inf"))
