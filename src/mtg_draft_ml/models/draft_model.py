"""Content draft model: shared card encoder + pool encoder + pointer pick head.

See docs/architecture.md.

The content feature matrix [n_cards, D] is held as a frozen buffer; cards are encoded by index.
Because the card encoder is content-based (DD-001), swapping in a matrix that includes unseen
cards lets the model score them with no retraining — the basis of the Phase-1 generalization test.

`card_embeddings()` exposes the per-card embedding for reuse by other heads (e.g. the card-value
tool, DD-005).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .card_encoder import CardEncoder
from .pick_head import DotPickHead
from .pool_encoder import MeanPoolEncoder


class ContentDraftModel(nn.Module):
    def __init__(self, content_matrix: torch.Tensor, emb_dim: int = 256,
                 enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("content", content_matrix.float())  # [n_cards, D], frozen
        self.card_encoder = CardEncoder(content_matrix.shape[1], hidden=enc_hidden,
                                        out_dim=emb_dim, layers=enc_layers, dropout=dropout)
        self.pool_encoder = MeanPoolEncoder()
        self.pick_head = DotPickHead(emb_dim)

    def set_content(self, content_matrix: torch.Tensor):
        """Swap the card table (e.g. to a new set's cards) for zero-shot evaluation."""
        self.content = content_matrix.float().to(self.content.device)

    def card_embeddings(self) -> torch.Tensor:
        """Encode every card in the current table -> [n_cards, emb_dim]."""
        return self.card_encoder(self.content)

    def forward(self, pool: torch.Tensor, pool_mask: torch.Tensor,
                pack: torch.Tensor, pack_mask: torch.Tensor) -> torch.Tensor:
        """pool[B,L], pool_mask[B,L], pack[B,P], pack_mask[B,P] -> logits[B,P] over pack positions."""
        all_emb = self.card_embeddings()          # [n_cards, d]
        pool_emb = all_emb[pool]                   # [B, L, d]
        pack_emb = all_emb[pack]                   # [B, P, d]
        ctx = self.pool_encoder(pool_emb, pool_mask)
        return self.pick_head(ctx, pack_emb, pack_mask)
