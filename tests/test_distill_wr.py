"""Tests for WR-softmax soft-label distillation (DD-004 #1, good-not-just-human dense target)."""
import json

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.distill.wr import WRSoftmaxTeacher, run_wr_distill  # noqa: E402

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


def _rich_ratings(tmp_path, name, rows):
    """rows: {card: {field: value, ...}} with the richer 17lands fields."""
    p = tmp_path / f"rich_{name}.json"
    p.write_text(json.dumps([{"name": n, **fields} for n, fields in rows.items()]))
    return str(p)


# ---- the teacher distribution ----

def _teacher():
    # 4-card global vocab: card 0 best, 2 worst, 3 unrated
    score = torch.tensor([2.0, 1.0, -1.0, 0.0])
    mask = torch.tensor([True, True, True, False])
    return WRSoftmaxTeacher(score, mask, tau=1.0)


def test_wr_teacher_ranks_by_winrate():
    t = _teacher()
    pool = torch.tensor([[0]]); pool_mask = torch.tensor([[True]])
    pack = torch.tensor([[0, 1, 2]]); pack_mask = torch.ones(1, 3, dtype=torch.bool)
    p = t.mean_probs(pool, pool_mask, pack, pack_mask)
    assert p.sum(-1).item() == pytest.approx(1.0)
    assert p[0, 0] > p[0, 1] > p[0, 2]                       # WR order 0 > 1 > 2


def test_wr_teacher_neutral_fill_for_unrated():
    t = _teacher()
    # pack = [card0 (rated, hi), card2 (rated, lo), card3 (unrated)]: card3 sits at the pack mean
    pack = torch.tensor([[0, 2, 3]]); pack_mask = torch.ones(1, 3, dtype=torch.bool)
    p = t.mean_probs(torch.tensor([[0]]), torch.tensor([[True]]), pack, pack_mask)
    assert p[0, 0] > p[0, 2] > p[0, 1]                       # hi > neutral(unrated) > lo
    assert p.sum(-1).item() == pytest.approx(1.0)


def test_wr_teacher_pad_gets_zero_mass():
    t = _teacher()
    pack = torch.tensor([[0, 1, 2]]); pack_mask = torch.tensor([[True, True, False]])
    p = t.mean_probs(torch.tensor([[0]]), torch.tensor([[True]]), pack, pack_mask)
    assert p[0, 2].item() == pytest.approx(0.0)
    assert p.sum(-1).item() == pytest.approx(1.0)


def test_wr_teacher_uniform_when_no_signal():
    t = _teacher()
    # only one rated card in the pack -> no real WR choice -> uniform over the pack
    pack = torch.tensor([[0, 3]]); pack_mask = torch.ones(1, 2, dtype=torch.bool)
    p = t.mean_probs(torch.tensor([[0]]), torch.tensor([[True]]), pack, pack_mask)
    assert p[0, 0].item() == pytest.approx(0.5)
    assert p[0, 1].item() == pytest.approx(0.5)


# ---- composite card-quality from richer fields ----

def test_composite_card_quality(tmp_path):
    from mtg_draft_ml.cards.content_table import build_multiset_content
    from mtg_draft_ml.eval.winrate import composite_card_quality

    _, man, scry = _prep(tmp_path, "A", CSV_A)
    # equal (high) counts so shrinkage is uniform — isolates the multi-field combination ordering.
    # GIH-WR and ALSA both rank A>B>C>D; the composite should preserve that.
    rows = {
        "A": {"ever_drawn_win_rate": .60, "drawn_improvement_win_rate": .04, "avg_pick": 1.0,
              "ever_drawn_game_count": 5000, "drawn_game_count": 5000, "pick_count": 5000},
        "B": {"ever_drawn_win_rate": .55, "drawn_improvement_win_rate": .02, "avg_pick": 4.0,
              "ever_drawn_game_count": 5000, "drawn_game_count": 5000, "pick_count": 5000},
        "C": {"ever_drawn_win_rate": .52, "drawn_improvement_win_rate": .00, "avg_pick": 8.0,
              "ever_drawn_game_count": 5000, "drawn_game_count": 5000, "pick_count": 5000},
        "D": {"ever_drawn_win_rate": .48, "drawn_improvement_win_rate": -.03, "avg_pick": 13.0,
              "ever_drawn_game_count": 5000, "drawn_game_count": 5000, "pick_count": 5000},
    }
    rat = _rich_ratings(tmp_path, "A", rows)
    _, _, _, l2gs = build_multiset_content([{"manifest": man, "scryfall": scry}], embedder=None)
    score, mask = composite_card_quality([{"manifest": man, "ratings": rat}], l2gs, n_global=4,
                                         fields=("ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"))
    assert mask.all() and score.shape == (4,)
    idx = {c["name"]: c["index"] for c in json.load(open(man))["cards"]}
    # ALSA inverted + WR both point the same way: A (strong) > D (weak)
    assert score[idx["A"]] > score[idx["B"]] > score[idx["C"]] > score[idx["D"]]


def test_composite_confidence_shrinks_low_sample(tmp_path):
    from mtg_draft_ml.eval.winrate import _set_quality

    man = tmp_path / "m.json"; rat = tmp_path / "r.json"
    man.write_text(json.dumps({"cards": [{"index": 0, "name": "Hi", "oracle_id": "hi"},
                                         {"index": 1, "name": "Lo", "oracle_id": "lo"}]}))
    # identical-magnitude extreme GIH, but Lo has a tiny sample -> shrunk toward 0
    rat.write_text(json.dumps([{"name": "Hi", "ever_drawn_win_rate": .62, "ever_drawn_game_count": 20000},
                               {"name": "Lo", "ever_drawn_win_rate": .38, "ever_drawn_game_count": 20}]))
    shrunk = _set_quality(str(man), str(rat), ("ever_drawn_win_rate",), [1.0], shrink=True)
    unshrunk = _set_quality(str(man), str(rat), ("ever_drawn_win_rate",), [1.0], shrink=False)
    assert abs(shrunk[1]) < abs(unshrunk[1])     # low-sample card pulled toward neutral


# ---- end-to-end experiment ----

def test_run_wr_distill_end_to_end(tmp_path):
    pq_a, man_a, scry = _prep(tmp_path, "A", CSV_A)
    pq_b, man_b, _ = _prep(tmp_path, "B", CSV_B)
    # richer train ratings so the composite arm has multiple fields to combine
    rat_b = _rich_ratings(tmp_path, "B", {
        "A": {"ever_drawn_win_rate": .58, "drawn_improvement_win_rate": .03, "avg_pick": 1.5,
              "ever_drawn_game_count": 4000, "drawn_game_count": 4000, "pick_count": 4000},
        "B": {"ever_drawn_win_rate": .55, "drawn_improvement_win_rate": .01, "avg_pick": 4.0,
              "ever_drawn_game_count": 3000, "drawn_game_count": 3000, "pick_count": 3000},
        "E": {"ever_drawn_win_rate": .53, "drawn_improvement_win_rate": .00, "avg_pick": 6.0,
              "ever_drawn_game_count": 2000, "drawn_game_count": 2000, "pick_count": 2000},
        "F": {"ever_drawn_win_rate": .48, "drawn_improvement_win_rate": -.02, "avg_pick": 11.0,
              "ever_drawn_game_count": 1000, "drawn_game_count": 1000, "pick_count": 1000}})
    rat_a = _ratings(tmp_path, "A", {"A": .58, "B": .55, "C": .50, "D": .47})
    res = run_wr_distill(
        [{"parquet": pq_b, "manifest": man_b, "scryfall": scry, "ratings": rat_b}],
        {"parquet": pq_a, "manifest": man_a, "scryfall": scry},
        holdout_ratings=rat_a, wr_tau=1.0,
        quality_fields=["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"],
        distill_lambda=0.5, distill_temp=1.0, adv_tau=0.05,
        embedder="hash", emb_dim=16, enc_hidden=32, enc_layers=2, pool="set_transformer",
        epochs=2, batch_size=4, val_frac=0.5, warmup_frac=0.0, grad_clip=0.0, device="cpu",
        checkpoint_dir=str(tmp_path / "ck"), seed=0,
    )
    assert res["mode"] == "wr_distill"
    assert res["quality_fields"] == ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"]
    for key in ("baseline", "wr_kd", "adv", "wr_kd_comp"):   # composite arm present
        m = res[key]
        assert 0.0 <= m["top1"] <= 1.0
        assert m["wr_agreement_model"] is not None and m["wr_agreement_human"] is not None
