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
    - mtpd: mean number of pack cards the model ranks strictly above the human pick
      (0 = model agreed; lower is better).
    - acc_by_pick: top-1 accuracy bucketed by pick_number (exposes the hard mid-pack region).
    """

    def __init__(self):
        self.n = 0
        self.correct = 0
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

        if pick_number is not None:
            pn = pick_number.tolist()
            hl = hit.tolist()
            for p, h in zip(pn, hl):
                b = self._by_pick.setdefault(int(p), [0, 0])
                b[0] += int(h)
                b[1] += 1

    def compute(self) -> dict:
        top1 = self.correct / max(self.n, 1)
        mtpd = self.dist_sum / max(self.n, 1)
        acc_by_pick = {p: c / t for p, (c, t) in sorted(self._by_pick.items()) if t}
        return {"top1": top1, "mtpd": mtpd, "n": self.n, "acc_by_pick": acc_by_pick}
