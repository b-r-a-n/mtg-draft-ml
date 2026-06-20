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


class _SAB(nn.Module):
    """Set Attention Block: masked multi-head self-attention over the set + FFN (residual)."""

    def __init__(self, d: int, heads: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.ReLU(), nn.Linear(d * ff_mult, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))


class _PMA(nn.Module):
    """Pooling by Multihead Attention: one learned seed query attends over the set -> [B, d]."""

    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        q = self.seed.expand(x.shape[0], 1, -1)
        a, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm(a)
        h = h + self.ff(h)
        return h.squeeze(1)


class SetTransformerEncoder(nn.Module):
    """Set Transformer pool encoder: SAB(s) for synergy + PMA pooling -> context [B, d].

    A learned token is prepended to the pool so every row has >=1 valid key — this both gives an
    empty pool (pick 0) a well-defined representation and avoids all-masked-row NaNs in attention.
    """

    def __init__(self, d: int, heads: int = 4, n_sab: int = 1, dropout: float = 0.1):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, d))
        self.sabs = nn.ModuleList([_SAB(d, heads, dropout=dropout) for _ in range(n_sab)])
        self.pma = _PMA(d, heads, dropout=dropout)

    def forward(self, card_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = card_embs.shape[0]
        tok = self.token.expand(B, 1, -1)
        x = torch.cat([tok, card_embs], dim=1)              # [B, 1+L, d]
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        kpm = ~valid                                        # MultiheadAttention: True = ignore
        for sab in self.sabs:
            x = sab(x, kpm)
        return self.pma(x, kpm)
