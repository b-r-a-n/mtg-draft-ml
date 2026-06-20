"""Tests for the adjusted-WR auxiliary head + global target construction."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.eval.winrate import build_global_wr_targets, zscore_ignore_nan  # noqa: E402
from mtg_draft_ml.models.draft_model import ContentDraftModel  # noqa: E402


def test_zscore_ignore_nan():
    a = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float32)
    z = zscore_ignore_nan(a)
    assert np.isnan(z[3])
    assert abs(np.nanmean(z)) < 1e-6
    assert abs(np.nanstd(z) - 1.0) < 1e-5


def test_build_global_wr_targets(tmp_path):
    # one set, 3 cards (A,B,C); C has no rating -> masked out
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"cards": [
        {"index": 0, "name": "A", "oracle_id": "a"},
        {"index": 1, "name": "B", "oracle_id": "b"},
        {"index": 2, "name": "C", "oracle_id": "c"},
    ]}))
    rat = tmp_path / "r.json"
    rat.write_text(json.dumps([
        {"name": "A", "ever_drawn_win_rate": 0.60},
        {"name": "B", "ever_drawn_win_rate": 0.50},
    ]))
    l2g = np.array([0, 1, 2])  # identity map (single set)
    target, mask = build_global_wr_targets([{"manifest": str(man), "ratings": str(rat)}],
                                           [l2g], n_global=3)
    assert mask.tolist() == [True, True, False]
    # standardized within set: A (higher WR) > B; mean ~0 over the two rated
    assert target[0] > target[1]
    assert abs(target[0] + target[1]) < 1e-5  # two-point z-scores are +/- equal


def test_card_quality_head():
    matrix = torch.randn(6, 10)
    m = ContentDraftModel(matrix, emb_dim=8, enc_hidden=16, enc_layers=2, aux_wr=True)
    q = m.card_quality()
    assert q.shape == (6,)               # one predicted WR per card
    assert torch.isfinite(q).all()
    # without aux head it should raise
    m2 = ContentDraftModel(matrix, emb_dim=8, enc_hidden=16, enc_layers=2, aux_wr=False)
    assert m2.wr_head is None
    with pytest.raises(RuntimeError):
        m2.card_quality()


def test_pick_time_blend_follows_quality():
    # with a large blend alpha, the pick should follow the predicted card quality
    matrix = torch.randn(6, 10)
    m = ContentDraftModel(matrix, emb_dim=8, enc_hidden=16, enc_layers=2, pool="mean",
                          aux_wr=True).eval()
    pool = torch.tensor([[0, 1]]); pool_mask = torch.tensor([[True, True]])
    pack = torch.tensor([[2, 3, 4]]); pack_mask = torch.tensor([[True, True, True]])
    with torch.no_grad():
        logits = m(pool, pool_mask, pack, pack_mask)        # [1,3]
        q = m.card_quality()                                # [6]
        blended = logits + 1e4 * q[pack]                    # huge alpha
    assert int(blended.argmax(dim=-1)) == int(q[pack].argmax(dim=-1))  # picks highest-quality card
    # alpha=0 leaves the pick unchanged
    assert int((logits + 0.0 * q[pack]).argmax(-1)) == int(logits.argmax(-1))
