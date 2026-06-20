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


def zscore_ignore_nan(a: np.ndarray) -> np.ndarray:
    """Standardize an array (mean 0, std 1) ignoring NaNs; NaNs are preserved."""
    m = np.nanmean(a)
    s = np.nanstd(a)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return (a - m) / s


def build_global_wr_targets(rating_specs, local_to_global, n_global: int,
                            field: str = DEFAULT_FIELD, standardize: bool = True):
    """Build a global-vocab win-rate regression target for the aux head (Phase 3 / DD-004).

    rating_specs: list of {"manifest", "ratings"} (parallel to local_to_global). Each set's win
    rates are optionally **standardized within the set** (z-score) — a light confounder adjustment
    that removes per-format win-rate baselines so the target is *relative card quality*, comparable
    across sets. Reprints (same global card) are averaged.

    Returns (target[n_global] float32 (0 where unknown), mask[n_global] bool).
    """
    acc = np.zeros(n_global, dtype=np.float32)
    cnt = np.zeros(n_global, dtype=np.float32)
    for spec, l2g in zip(rating_specs, local_to_global):
        local = align_winrates(spec["manifest"], spec["ratings"], field)
        if standardize:
            local = zscore_ignore_nan(local)
        for i, gv in enumerate(l2g):
            if not np.isnan(local[i]):
                acc[gv] += local[i]
                cnt[gv] += 1.0
    mask = cnt > 0
    acc[mask] /= cnt[mask]
    return acc, mask


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
