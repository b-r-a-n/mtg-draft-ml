"""Tests for the mechanistic castability model (eval/castability.py)."""
import pytest

from mtg_draft_ml.eval import castability as C


def test_hypergeom_exact():
    # P(>=2 of 17 lands in 8 cards from 40) — closed-form anchor
    assert C.hypergeom_at_least(2, 40, 17, 8) == pytest.approx(0.9394, abs=1e-3)
    assert C.hypergeom_at_least(0, 40, 17, 8) == 1.0          # >=0 always
    assert C.hypergeom_at_least(3, 40, 2, 8) == 0.0           # can't draw 3 of only 2


def test_parse_pips():
    assert {k: v for k, v in C.parse_pips("{2}{U}{U}").items() if k in C.COLORS} == \
        {"W": 0, "U": 2, "B": 0, "R": 0, "G": 0}
    assert C.parse_pips("{X}{R}")["R"] == 1                   # X contributes no color
    assert C.parse_pips("{5}") == dict.fromkeys(C.COLORS, 0) | {"_hybrid": []}
    assert C.parse_pips("{W/U}")["_hybrid"] == [("W", "U")]   # hybrid recorded, not a hard pip
    assert C.parse_pips("{W/U}")["W"] == 0


def test_mdfc_mana_cost_fallback():
    # Scryfall leaves top-level mana_cost empty on DFCs; the front face carries the cost
    mdfc = {"t": "creature", "mana_cost": "", "card_faces": [{"mana_cost": "{1}{G}"}, {"mana_cost": ""}]}
    assert C.card_mana_cost(mdfc) == "{1}{G}"
    assert C.card_mana_cost({"t": "creature", "mana_cost": "{2}{R}"}) == "{2}{R}"   # normal card unchanged
    assert C.card_mana_cost({"t": "land", "mana_cost": ""}) == ""                   # land: no cost


def test_joint_no_double_count():
    # a {U}{U} 2-drop in an ALL-blue manabase: every land is a U source, so P(castable) must equal
    # P(>=2 lands) exactly — the old P(lands)*P(sources) factorization under-counted this.
    seen = C.seen_cards(2, 40, on_play=True)
    joint = C._joint_castable(40, seen, 2, ((17, 2),), 0, 23)
    assert joint == pytest.approx(C.p_lands_ok(40, 17, 2, 2), abs=1e-9)
    # the old double-count is strictly lower (the bug we fixed)
    assert joint > C.p_lands_ok(40, 17, 2, 2) * C.p_color_ok(40, 17, 2, 2)


@pytest.mark.parametrize("turn,pips", sorted(C.KARSTEN_40))
def test_karsten_single_and_double_pip_anchor(turn, pips):
    # the min sources our model needs for >=90% should match Karsten's published table within ±2
    target = C.KARSTEN_40[(turn, pips)]
    need = next(s for s in range(1, 40) if C.p_color_ok(40, s, pips, turn) >= 0.90)
    assert abs(need - target) <= 2, f"turn {turn} {pips}-pip: model {need} vs Karsten {target}"


def _card(cmc, mc, t="creature"):
    return {"cmc": cmc, "t": t, "mana_cost": mc, "ci": ""}


def test_coherent_beats_rainbow():
    # same number of spells; a 2-color deck should be far more castable than a 5-color smear
    coherent = [_card(1, "{W}"), _card(2, "{1}{W}"), _card(2, "{W}{U}"), _card(3, "{2}{U}"),
                _card(3, "{1}{W}{U}"), _card(4, "{3}{W}"), _card(2, "{U}{U}"), _card(1, "{U}"),
                _card(4, "{2}{W}{U}"), _card(5, "{4}{U}")]
    rainbow = [_card(1, "{W}"), _card(2, "{1}{U}"), _card(2, "{B}{B}"), _card(3, "{2}{R}"),
               _card(3, "{1}{G}{G}"), _card(4, "{3}{B}"), _card(2, "{R}{W}"), _card(1, "{G}"),
               _card(4, "{2}{U}{B}"), _card(5, "{4}{R}")]
    idx = list(range(10))
    sc = C.castability_score(idx, coherent, n_lands=17)
    sr = C.castability_score(idx, rainbow, n_lands=17)
    assert 0.0 <= sr < sc <= 1.0
    assert sc - sr > 0.15                       # a large, real gap (not the compressed one)


def test_score_bounds_and_empty():
    assert C.castability_score([], [], n_lands=17) == 0.0
    one = [_card(2, "{1}{U}")]
    assert 0.0 < C.castability_score([0], one, n_lands=17) <= 1.0
