"""Unit tests for the deck doctor (eval/deck_doctor.py) — synthetic cards + a fake play model, no data."""
import numpy as np

from mtg_draft_ml.eval.deck_doctor import diagnose


class _FakePM:
    """Ranks pack cards by a fixed per-card score so the build is deterministic."""
    def __init__(self, score):
        self.score = score
    def probs(self, pool, pack):
        return np.array([self.score[i] for i in pack], dtype=float)


def _card(i, name, dv, cmc, ci, mc, t="creature"):
    return {"i": i, "name": name, "deck_value": dv, "cmc": cmc, "ci": ci, "mana_cost": mc, "t": t}


def _coherent_set():
    # 8 clean mono-U spells (high P(played)) + 4 off-color splashes (lower) + some lands
    cards, score = [], {}
    for i in range(8):
        cards.append(_card(i, f"U{i}", 0.2, 1 + i % 4, "U", "{%d}{U}" % (i % 4)))
        score[i] = 0.9
    for i in range(8, 12):
        cards.append(_card(i, f"R{i}", 0.25, 3, "R", "{2}{R}"))   # off-color bombs, model won't play
        score[i] = 0.2
    return cards, score


def test_report_shape_and_grade():
    cards, score = _coherent_set()
    pool = list(range(12))
    r = diagnose(pool, cards, _FakePM(score), n_spells_cap=8)
    for k in ("deck", "power_pct", "castability", "coherence", "grade", "advice", "per_turn_castability"):
        assert k in r
    assert r["grade"] in {"A", "B", "C", "D", "F"}
    assert 0.0 <= r["castability"] <= 1.0 and 0.0 <= r["coherence"] <= 1.0
    # the mono-U cards (high score) make the deck; the off-color R bombs are excluded
    assert all(n.startswith("U") for n in r["deck"])
    assert r["coherence"] == 1.0                       # single color -> fully concentrated


def test_advice_flags_offcolor_left_behind_only_if_oncolor():
    # an off-color bomb left in the sideboard must NOT be suggested (it isn't castable in-colors)
    cards, score = _coherent_set()
    r = diagnose(list(range(12)), cards, _FakePM(score), n_spells_cap=8)
    assert not any("R" in line and "consider running" in line for line in r["advice"])
