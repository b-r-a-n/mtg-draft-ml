"""Phase-5 sequence draft model: causal transformer over (pack-seen, pick-made) steps.

See docs/phase5-sequence-modeling.md.

Replaces the memoryless pool encoder with a causal transformer over the draft's history, so the
context for pick t includes what you SAW and PASSED at every earlier pick (signal reading), not just
the cards you kept. Shares the content card encoder (DD-001) and the pointer head (DD-002), so it
still generalizes to unseen cards.

Causality: a step token z_t = f(pooled pack_t, card picked_t, position_t) — it contains the pick, so
it must NOT be visible when predicting pick_t. We prepend a learned BOS token and run a causal
transformer over [BOS, z_1..z_T]; the context for predicting pick_t is the output at position t-1
(it attends to BOS..z_{t-1} only). That context, concatenated with the *current* pooled pack, queries
the pointer head over pack_t's card embeddings.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .card_encoder import CardEncoder


class SequenceDraftModel(nn.Module):
    def __init__(self, content_matrix: torch.Tensor, emb_dim: int = 256,
                 enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
                 n_layers: int = 2, n_heads: int = 4, max_steps: int = 48, aux_wr: bool = False):
        super().__init__()
        self.register_buffer("content", content_matrix.float())
        self.card_encoder = CardEncoder(content_matrix.shape[1], hidden=enc_hidden,
                                        out_dim=emb_dim, layers=enc_layers, dropout=dropout)
        # step token = MLP([pooled_pack ; picked_card]) -> emb_dim
        self.step_proj = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.ReLU(),
                                       nn.Linear(emb_dim, emb_dim))
        self.bos = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos = nn.Embedding(max_steps + 1, emb_dim)  # +1 for BOS position
        layer = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward=emb_dim * 2,
                                           dropout=dropout, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        # query for the pointer head = MLP([history_context ; current pooled pack])
        self.query_proj = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.ReLU(),
                                        nn.Linear(emb_dim, emb_dim))
        self.scale = 1.0 / math.sqrt(emb_dim)
        from .draft_model import WRHead
        self.wr_head = WRHead(emb_dim) if aux_wr else None
        self.max_steps = max_steps

    def set_content(self, content_matrix: torch.Tensor):
        self.content = content_matrix.float().to(self.content.device)

    def card_embeddings(self) -> torch.Tensor:
        return self.card_encoder(self.content)

    def card_quality(self) -> torch.Tensor:
        if self.wr_head is None:
            raise RuntimeError("model built without aux_wr head")
        return self.wr_head(self.card_embeddings())

    @staticmethod
    def _masked_mean(emb, mask):  # emb[...,P,d], mask[...,P] -> [...,d]
        m = mask.unsqueeze(-1).to(emb.dtype)
        return (emb * m).sum(-2) / m.sum(-2).clamp_min(1.0)

    def forward(self, pack, pack_mask, step_mask, pick_idx):
        """pack[B,T,P], pack_mask[B,T,P], step_mask[B,T], pick_idx[B,T] -> logits[B,T,P] over packs.

        Per-step pick logits; padded steps/cards are -inf-masked. Loss is masked CE over real steps.
        """
        B, T, P = pack.shape
        all_emb = self.card_embeddings()                 # [n_cards, d]
        d = all_emb.shape[1]
        pack_emb = all_emb[pack]                          # [B,T,P,d]
        pooled_pack = self._masked_mean(pack_emb, pack_mask)          # [B,T,d] (what was seen)
        picked_emb = all_emb[pick_idx]                               # [B,T,d] (what was taken)

        z = self.step_proj(torch.cat([pooled_pack, picked_emb], dim=-1))  # [B,T,d] step tokens
        bos = self.bos.expand(B, 1, d)
        seq = torch.cat([bos, z], dim=1)                             # [B,T+1,d]
        pos_ids = torch.arange(T + 1, device=pack.device).clamp_max(self.max_steps)
        seq = seq + self.pos(pos_ids).unsqueeze(0)

        causal = torch.triu(torch.ones(T + 1, T + 1, device=pack.device, dtype=torch.bool), 1)
        # key padding: BOS always valid; step i valid per step_mask
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pack.device),
                         ~step_mask], dim=1)              # True = ignore
        out = self.transformer(seq, mask=causal, src_key_padding_mask=kpm)  # [B,T+1,d]
        # context for predicting pick at step t (0-indexed) = output at position t (= BOS..z_{t-1})
        ctx = out[:, :T, :]                              # [B,T,d], excludes z_t for each t

        query = self.query_proj(torch.cat([ctx, pooled_pack], dim=-1))     # [B,T,d]
        # pointer logits per step: dot(query_t, pack_emb_t) over P, masked
        logits = (query.unsqueeze(2) * pack_emb).sum(-1) * self.scale       # [B,T,P]
        return logits.masked_fill(~pack_mask, float("-inf"))
