"""Evaluation metrics: top-1, mean pick distance (MTPD), per-pick-position accuracy.

See docs/roadmap.md Phase 0/1; research/findings/data-and-eval.md.

The new-set generalization protocol (leave-one-set-out, release-day feature-zeroing,
WR-agreement) is Phase 1+ and will extend this evaluator.
"""
from __future__ import annotations

import torch


class PickEvaluator:
    """Accumulate pick metrics over batches of (pack-masked) logits.

    - top1: fraction where argmax == the human pick.
    - top3 / top5: fraction where the human pick is among the model's k highest-ranked pack cards
      (a fairer metric under genuine human disagreement — did the model rate the pick as reasonable?).
      NOTE: when a pack has <=k cards (late picks), top-k is trivially ~1.0 — read per-region.
    - mtpd: mean number of pack cards the model ranks strictly above the human pick
      (0 = model agreed; lower is better).
    - acc_by_pick: top-1 accuracy bucketed by pick_number (exposes the hard mid-pack region).
    """

    def __init__(self):
        self.n = 0
        self.correct = 0
        self.top3 = 0
        self.top5 = 0
        self.dist_sum = 0.0
        self._by_pick: dict[int, list[int]] = {}  # pick_number -> [correct, total]

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor,
               pack_mask: torch.Tensor, pick_number: torch.Tensor | None = None):
        pred = logits.argmax(dim=-1)
        hit = pred == target
        self.correct += int(hit.sum())
        self.n += logits.shape[0]

        true_score = logits.gather(1, target[:, None]).squeeze(1)
        # cards within the pack scored strictly above the human pick
        better = ((logits > true_score[:, None]) & pack_mask).sum(dim=-1)
        self.dist_sum += float(better.sum())
        # human pick is in top-k iff fewer than k cards rank strictly above it
        self.top3 += int((better < 3).sum())
        self.top5 += int((better < 5).sum())

        if pick_number is not None:
            pn = pick_number.tolist()
            hl = hit.tolist()
            for p, h in zip(pn, hl):
                b = self._by_pick.setdefault(int(p), [0, 0])
                b[0] += int(h)
                b[1] += 1

    def compute(self) -> dict:
        n = max(self.n, 1)
        acc_by_pick = {p: c / t for p, (c, t) in sorted(self._by_pick.items()) if t}
        return {"top1": self.correct / n, "top3": self.top3 / n, "top5": self.top5 / n,
                "mtpd": self.dist_sum / n, "n": self.n, "acc_by_pick": acc_by_pick}
