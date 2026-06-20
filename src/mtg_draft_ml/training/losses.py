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
