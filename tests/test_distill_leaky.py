"""Tests for leaky-feature -> release-day distillation (DD-004 #3) + CompositeTeacher."""
import json

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.distill.ensemble import CompositeTeacher  # noqa: E402
from mtg_draft_ml.distill.leaky import augment_with_winrate, run_leaky_distill  # noqa: E402

CSV_A = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_C,pack_card_D,pool_A,pool_B,pool_C,pool_D
d1,0,0,A,3,0,1,1,1,1,0,0,0,0
d1,0,1,B,3,0,0,1,1,1,1,0,0,0
d2,0,0,C,1,2,1,1,1,0,0,0,0,0
d2,0,1,A,1,2,1,1,0,1,0,0,1,0
d3,0,0,D,2,1,1,1,0,1,0,0,0,0
d3,0,1,B,2,1,0,1,1,1,0,0,0,1
"""
CSV_B = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_E,pack_card_F,pool_A,pool_B,pool_E,pool_F
e1,0,0,E,3,0,1,1,1,1,0,0,0,0
e1,0,1,A,3,0,1,1,0,1,0,0,1,0
e2,0,0,F,1,2,0,1,1,1,0,0,0,0
e2,0,1,B,1,2,1,1,0,1,0,0,1,0
"""
SCRY = [{"name": n, "oracle_id": n.lower(), "cmc": float(i), "type_line": "Creature",
         "colors": ["R"], "color_identity": ["R"], "rarity": "common", "oracle_text": f"text {n}"}
        for i, n in enumerate(["A", "B", "C", "D", "E", "F"])]


def _prep(tmp_path, name, csv):
    p = tmp_path / f"{name}.csv"; p.write_text(csv)
    pq = tmp_path / f"{name}.parquet"; man = tmp_path / f"{name}.json"
    scry = tmp_path / "scry.json"; scry.write_text(json.dumps(SCRY))
    preprocess_set(p, pq, man, scryfall_path=scry, set_code=name)
    return str(pq), str(man), str(scry)


def _ratings(tmp_path, name, wr):
    p = tmp_path / f"ratings_{name}.json"
    p.write_text(json.dumps([{"name": n, "ever_drawn_win_rate": v} for n, v in wr.items()]))
    return str(p)


def test_augment_with_winrate_appends_two_columns():
    base = np.zeros((3, 4), dtype=np.float32)
    wr_z = np.array([1.5, np.nan, -0.5])
    wr_mask = np.array([True, False, True])
    aug = augment_with_winrate(base, wr_z, wr_mask)
    assert aug.shape == (3, 6)                          # +2 columns
    assert aug[0, 4] == pytest.approx(1.5) and aug[0, 5] == 1.0     # rated card: WR + flag
    assert aug[1, 4] == 0.0 and aug[1, 5] == 0.0                    # unrated: 0-filled, flag off


def test_composite_teacher_averages_distributions():
    class _Fixed:
        def __init__(self, p):
            self.p = p

        def mean_probs(self, pool, pool_mask, pack, pack_mask, temp=2.0):
            return self.p

    a = _Fixed(torch.tensor([[0.8, 0.2]]))
    b = _Fixed(torch.tensor([[0.0, 1.0]]))
    out = CompositeTeacher([a, b]).mean_probs(None, None, None, None)
    assert torch.allclose(out, torch.tensor([[0.4, 0.6]]))
    # weighted: 2*a + 1*b normalized
    wout = CompositeTeacher([a, b], weights=[2.0, 1.0]).mean_probs(None, None, None, None)
    assert torch.allclose(wout, torch.tensor([[(1.6 + 0.0) / 3, (0.4 + 1.0) / 3]]))
    assert out.sum(-1).item() == pytest.approx(1.0)


def test_run_leaky_distill_end_to_end(tmp_path):
    pq_a, man_a, scry = _prep(tmp_path, "A", CSV_A)
    pq_b, man_b, _ = _prep(tmp_path, "B", CSV_B)
    rat_b = _ratings(tmp_path, "B", {"A": .58, "B": .55, "E": .53, "F": .48})
    rat_a = _ratings(tmp_path, "A", {"A": .58, "B": .55, "C": .50, "D": .47})
    res = run_leaky_distill(
        [{"parquet": pq_b, "manifest": man_b, "scryfall": scry, "ratings": rat_b}],
        {"parquet": pq_a, "manifest": man_a, "scryfall": scry},
        holdout_ratings=rat_a, distill_lambda=0.5, distill_temp=2.0,
        embedder="hash", emb_dim=16, enc_hidden=32, enc_layers=2, pool="set_transformer",
        epochs=2, batch_size=4, val_frac=0.5, warmup_frac=0.0, grad_clip=0.0, device="cpu",
        checkpoint_dir=str(tmp_path / "ck"), seed=0, teacher_seed=1,
    )
    assert res["mode"] == "leaky_distill"
    assert res["teacher_dim"] == res["base_dim"] + 2          # WR + rated-flag columns
    for key in ("baseline", "distilled", "teacher"):
        m = res[key]
        assert 0.0 <= m["top1"] <= 1.0
        assert m["wr_agreement_model"] is not None
    assert res["frac_cards_novel"] == 0.5
