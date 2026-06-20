"""Pointer / masked-softmax head scoring each card in the variable pack.

See docs/architecture.md component 3; design-decisions.md DD-002.

logit_i = <query(context), card_i> / sqrt(d), masked to the pack. Output is logits over pack
*positions* [B, P]; padded positions are set to -inf so softmax ignores them. This natively
handles any pack size / contents, including unseen cards.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DotPickHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, context: torch.Tensor, pack_embs: torch.Tensor,
                pack_mask: torch.Tensor) -> torch.Tensor:
        """context[B,d], pack_embs[B,P,d], pack_mask[B,P] bool -> logits[B,P] (masked)."""
        q = self.query(context).unsqueeze(1)               # [B, 1, d]
        logits = (q * pack_embs).sum(dim=-1) * self.scale  # [B, P]
        return logits.masked_fill(~pack_mask, float("-inf"))

    def score_global(self, context: torch.Tensor, card_embs: torch.Tensor) -> torch.Tensor:
        """Score a shared set of negative cards against each context. card_embs[K,d] -> [B,K].

        Used for contextual InfoNCE: extra negatives drawn from the whole vocab (beyond the pack).
        """
        q = self.query(context)                            # [B, d]
        return (q @ card_embs.t()) * self.scale            # [B, K]
