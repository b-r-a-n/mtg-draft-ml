"""Training losses for draft pick models.

See docs/architecture.md component 4; design-decisions.md DD-004.

Phase 0 uses `pick_cross_entropy` (masked-softmax CE). The InfoNCE / distillation variants for
the Phase-2 contextual model live alongside it but are not used by the baseline yet.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pick_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                       weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy over (already pack-masked) logits.

    logits[B, V] with -inf at non-pack cards; target[B] is the global index of the picked card
    (guaranteed present in the pack). `weights[B]` optionally scales each example (e.g. by drafter
    win-rate — DD-004 / Phase 3).
    """
    loss = F.cross_entropy(logits, target, reduction="none")
    if weights is not None:
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1e-8)
    return loss.mean()


def win_rate_weights(wins: torch.Tensor, scheme: str = "exp", beta: float = 0.3,
                     baseline: float = 3.0, min_wins: int = 0) -> torch.Tensor:
    """Per-example weights from a draft's event_match_wins (advantage-weighted BC — DD-004).

    scheme: 'none' (all 1), 'linear' (wins+1), or 'exp' (exp(beta*(wins-baseline))). `min_wins`
    zeroes examples below a win threshold (a soft filter to high-win drafters). Higher wins ->
    higher weight, biasing imitation toward decks that actually won.
    """
    w = wins.float()
    if scheme == "none":
        out = torch.ones_like(w)
    elif scheme == "linear":
        out = w + 1.0
    elif scheme == "exp":
        out = torch.exp(beta * (w - baseline))
    else:
        raise ValueError(f"unknown win-weight scheme: {scheme}")
    if min_wins > 0:
        out = out * (w >= min_wins).float()
    return out


def pick_advantage_weights(pick_idx, pack, pack_mask, card_wr, card_wr_mask,
                           tau: float = 0.05, clamp: float = 5.0) -> torch.Tensor:
    """Per-example weights = how good the human's pick was, by win-rate, RELATIVE to its pack.

    advantage = card_wr[pick] - mean(card_wr over rated pack cards); weight = exp(advantage / tau),
    clamped to [1/clamp, clamp], normalized to mean 1 over the batch. Examples where the picked card
    or the pack has no win-rate rating get weight 1 (neutral). This upweights imitation of high-WR
    (winning) human picks and downweights weak ones — WITHOUT ever changing the target, so the
    contextual policy (pool fit / signals) is preserved. card_wr is a per-card array (global vocab),
    card_wr_mask marks which cards are rated (DD-004; less-confounded with IWD as the WR field).
    """
    wr = torch.as_tensor(card_wr, dtype=torch.float32, device=pack.device)
    rated = torch.as_tensor(card_wr_mask, dtype=torch.bool, device=pack.device)
    pick_wr = wr[pick_idx]
    pick_rated = rated[pick_idx]
    pack_wr = wr[pack]
    pack_rated = rated[pack] & pack_mask
    denom = pack_rated.sum(-1).clamp_min(1)
    pack_mean = (torch.where(pack_rated, pack_wr, torch.zeros_like(pack_wr)).sum(-1)) / denom
    adv = pick_wr - pack_mean
    w = torch.exp(adv / tau).clamp(1.0 / clamp, clamp)
    usable = pick_rated & (pack_rated.sum(-1) > 0)
    w = torch.where(usable, w, torch.ones_like(w))
    return w / w.mean().clamp_min(1e-8)  # normalize so loss scale is stable
