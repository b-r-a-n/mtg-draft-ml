"""Card win-rate signal for 'good, not just human' evaluation (Phase 3).

See docs/roadmap.md Phase 3; research/findings/training-objectives.md.

Loads 17lands aggregate card ratings (default field: ever_drawn_win_rate = GIH WR) and aligns them
to a set's manifest vocabulary, so we can ask: does the model take the highest-win-rate card in the
pack — not just the human's pick? GIH WR is a confounded proxy (favors controlling decks); it's a
first-pass signal, with confounder-adjusted WR a future upgrade.
"""
from __future__ import annotations

import json

import numpy as np

DEFAULT_FIELD = "ever_drawn_win_rate"  # GIH WR


def load_card_winrates(ratings_path, field: str = DEFAULT_FIELD) -> dict[str, float]:
    data = json.load(open(ratings_path))
    out: dict[str, float] = {}
    for c in data:
        name, v = c.get("name"), c.get(field)
        if name and v is not None:
            out[name] = float(v)
    return out


def align_winrates(manifest_path, ratings_path, field: str = DEFAULT_FIELD) -> np.ndarray:
    """Return a float32 array [n_cards] of win rates in manifest-index order (NaN where missing)."""
    wr = load_card_winrates(ratings_path, field)
    cards = json.load(open(manifest_path))["cards"]
    arr = np.full(len(cards), np.nan, dtype=np.float32)
    for c in cards:
        v = wr.get(c["name"])
        if v is not None:
            arr[c["index"]] = v
    return arr


class WRMeter:
    """Accumulate 'good-not-just-human' metrics over batches, given per-card win rates.

    Operates on pack-position logits [B,P], the human label [B], pack_mask [B,P], and the win rate
    of each pack card [B,P] (NaN where unknown). Reports, over packs with >=2 rated cards:
      - wr_agreement_model / wr_agreement_human: fraction taking the highest-WR card in the pack
      - avg_pick_wr_model / avg_pick_wr_human:    mean WR of the chosen card
    """

    def __init__(self):
        self.n = 0
        self.model_best = 0
        self.human_best = 0
        self.model_wr_sum = 0.0
        self.human_wr_sum = 0.0
        self.model_pick_n = 0
        self.human_pick_n = 0

    def update(self, logits, label, pack_wr, pack_mask):
        import torch

        valid = pack_mask & ~torch.isnan(pack_wr)
        wr = torch.where(valid, pack_wr, torch.full_like(pack_wr, float("-inf")))
        enough = valid.sum(dim=-1) >= 2                       # need a real choice among rated cards
        if not bool(enough.any()):
            return
        best_pos = wr.argmax(dim=-1)
        model_pos = logits.argmax(dim=-1)
        rows = torch.arange(logits.shape[0], device=logits.device)

        e = enough
        self.n += int(e.sum())
        self.model_best += int((model_pos[e] == best_pos[e]).sum())
        self.human_best += int((label[e] == best_pos[e]).sum())

        # average WR of the chosen card, only over packs in `e` where the chosen card is rated
        mwr = pack_wr[rows, model_pos]
        hwr = pack_wr[rows, label]
        mvalid = e & ~torch.isnan(mwr)
        hvalid = e & ~torch.isnan(hwr)
        self.model_wr_sum += float(mwr[mvalid].sum())
        self.human_wr_sum += float(hwr[hvalid].sum())
        self.model_pick_n += int(mvalid.sum())
        self.human_pick_n += int(hvalid.sum())

    def compute(self) -> dict:
        n = max(self.n, 1)
        return {
            "wr_agreement_model": self.model_best / n,
            "wr_agreement_human": self.human_best / n,
            "avg_pick_wr_model": self.model_wr_sum / max(self.model_pick_n, 1),
            "avg_pick_wr_human": self.human_wr_sum / max(self.human_pick_n, 1),
            "n_wr_packs": self.n,
        }
