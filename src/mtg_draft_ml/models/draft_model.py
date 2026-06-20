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
from .pool_encoder import MeanPoolEncoder, SetTransformerEncoder


class ContentDraftModel(nn.Module):
    def __init__(self, content_matrix: torch.Tensor, emb_dim: int = 256,
                 enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
                 pool: str = "mean", n_heads: int = 4, n_sab: int = 1):
        super().__init__()
        self.register_buffer("content", content_matrix.float())  # [n_cards, D], frozen
        self.card_encoder = CardEncoder(content_matrix.shape[1], hidden=enc_hidden,
                                        out_dim=emb_dim, layers=enc_layers, dropout=dropout)
        if pool == "set_transformer":
            self.pool_encoder = SetTransformerEncoder(emb_dim, heads=n_heads, n_sab=n_sab,
                                                      dropout=dropout)
        elif pool == "mean":
            self.pool_encoder = MeanPoolEncoder()
        else:
            raise ValueError(f"unknown pool encoder: {pool}")
        self.pick_head = DotPickHead(emb_dim)

    def set_content(self, content_matrix: torch.Tensor):
        """Swap the card table (e.g. to a new set's cards) for zero-shot evaluation."""
        self.content = content_matrix.float().to(self.content.device)

    def card_embeddings(self) -> torch.Tensor:
        """Encode every card in the current table -> [n_cards, emb_dim]."""
        return self.card_encoder(self.content)

    def forward(self, pool: torch.Tensor, pool_mask: torch.Tensor,
                pack: torch.Tensor, pack_mask: torch.Tensor,
                neg_idx: torch.Tensor | None = None) -> torch.Tensor:
        """pool[B,L], pool_mask[B,L], pack[B,P], pack_mask[B,P] -> logits[B,P] over pack positions.

        If neg_idx[K] is given (contextual InfoNCE), K shared global negatives are scored and
        concatenated after the pack: returns [B, P+K]; the positive stays at its pack position, so
        the same pick_pos label is the cross-entropy target.
        """
        all_emb = self.card_embeddings()          # [n_cards, d]
        ctx = self.pool_encoder(all_emb[pool], pool_mask)
        pack_logits = self.pick_head(ctx, all_emb[pack], pack_mask)
        if neg_idx is None:
            return pack_logits
        neg_logits = self.pick_head.score_global(ctx, all_emb[neg_idx])  # [B, K]
        return torch.cat([pack_logits, neg_logits], dim=1)
