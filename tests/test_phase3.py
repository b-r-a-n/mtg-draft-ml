"""Phase-3 tests: win-rate weighting + WR-agreement metric."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.eval.winrate import WRMeter, align_winrates  # noqa: E402
from mtg_draft_ml.training.losses import pick_advantage_weights, win_rate_weights  # noqa: E402


def test_win_rate_weights_schemes():
    wins = torch.tensor([0, 3, 7])
    assert torch.allclose(win_rate_weights(wins, "none"), torch.ones(3))
    assert torch.allclose(win_rate_weights(wins, "linear"), torch.tensor([1.0, 4.0, 8.0]))
    exp = win_rate_weights(wins, "exp", beta=0.3, baseline=3.0)
    assert exp[1].item() == pytest.approx(1.0, abs=1e-5)        # baseline -> 1
    assert exp[2] > exp[1] > exp[0]                              # monotonic in wins
    # min_wins zeroes low-win examples
    filt = win_rate_weights(wins, "linear", min_wins=3)
    assert filt[0].item() == 0.0 and filt[1] > 0


def test_align_winrates(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"cards": [
        {"index": 0, "name": "A", "oracle_id": "a"},
        {"index": 1, "name": "B", "oracle_id": "b"},
        {"index": 2, "name": "C", "oracle_id": "c"},  # missing from ratings
    ]}))
    ratings = tmp_path / "r.json"
    ratings.write_text(json.dumps([
        {"name": "A", "ever_drawn_win_rate": 0.60},
        {"name": "B", "ever_drawn_win_rate": 0.52},
    ]))
    arr = align_winrates(man, ratings)
    assert arr[0] == pytest.approx(0.60) and arr[1] == pytest.approx(0.52)
    assert np.isnan(arr[2])


def test_wr_meter_agreement_and_avg():
    # one pack of 3 cards; WRs: pos0=0.55, pos1=0.60(best), pos2=0.50
    logits = torch.tensor([[10.0, 0.0, 0.0]])     # model picks pos0 (not the best-WR pos1)
    label = torch.tensor([1])                      # human picks pos1 (the best-WR card)
    pack_wr = torch.tensor([[0.55, 0.60, 0.50]])
    pack_mask = torch.tensor([[True, True, True]])
    m = WRMeter()
    m.update(logits, label, pack_wr, pack_mask)
    out = m.compute()
    assert out["n_wr_packs"] == 1
    assert out["wr_agreement_model"] == 0.0        # model missed the best-WR card
    assert out["wr_agreement_human"] == 1.0        # human took it
    assert out["avg_pick_wr_model"] == pytest.approx(0.55)
    assert out["avg_pick_wr_human"] == pytest.approx(0.60)


def test_pick_advantage_weights():
    # 3 cards, WR = [0.5, 0.6, 0.4] (card1 best). Weights are batch-normalized to mean 1, so the
    # meaningful test is WITHIN a batch: best-pick example upweighted vs worst-pick example.
    wr = [0.5, 0.6, 0.4]; mask = [True, True, True]
    pk2 = torch.tensor([[0, 1, 2], [0, 1, 2]])
    pm2 = torch.tensor([[True, True, True], [True, True, True]])
    w = pick_advantage_weights(torch.tensor([1, 2]), pk2, pm2, wr, mask, tau=0.05)  # picks: best, worst
    assert abs(w.mean().item() - 1.0) < 1e-5      # normalized to mean 1
    assert w[0] > 1.0 > w[1]                       # best-pick upweighted, worst-pick downweighted


def test_pick_advantage_unrated_is_neutral():
    wr = [0.5, float("nan"), 0.4]; mask = [True, False, True]
    pack = torch.tensor([[0, 1, 2], [0, 1, 2]]); pm = torch.tensor([[True]*3, [True]*3])
    # picking the unrated card (1) -> neutral weight 1 (pre-normalization); after norm still finite
    w = pick_advantage_weights(torch.tensor([1, 0]), pack, pm, wr, mask, tau=0.05)
    assert torch.isfinite(w).all()


def test_wr_meter_requires_two_rated_cards():
    # only one rated card -> pack excluded (no real WR choice)
    logits = torch.tensor([[1.0, 0.0]])
    label = torch.tensor([0])
    pack_wr = torch.tensor([[0.55, float("nan")]])
    pack_mask = torch.tensor([[True, True]])
    m = WRMeter()
    m.update(logits, label, pack_wr, pack_mask)
    assert m.compute()["n_wr_packs"] == 0
