"""Deck doctor — "rate my deck" over a drafted pool, combining the three validated lenses:

  POWER        Σ/mean `deck_value` — the game_data card-power model.
  BUILDABILITY `P(played | pool)` (eval/play_prob) — which cards actually belong; also *builds* the deck.
  FUNCTION     castability (eval/castability) — does the mana/curve work (hypergeometric, Karsten ±2,
               and now outcome-validated to predict winning, [decensor-curve.md]).

`diagnose()` builds the recommended deck from the pool (P(played)), scores it on the three axes, and
emits specific advice (splash warnings, on-curve castability weak spots, borderline includes, cards left
in the sideboard, land count). The overall letter grade is an illustrative HEURISTIC blend — the three
components are each individually validated, but their weighting into one number is not.
"""
from __future__ import annotations

import numpy as np

from .castability import castability_score, per_turn_castability
from .play_prob import deck_from_play_model


def _land_count(deck_idx, cards):
    cmcs = [cards[i].get("cmc") or 0 for i in deck_idx]
    avg = sum(cmcs) / len(cmcs) if cmcs else 3.0
    return (16 if avg < 2.6 else 18 if avg > 3.3 else 17), avg


def _colors(idxs, cards, thresh=3):
    cc = {}
    for i in idxs:
        for c in (cards[i].get("ci") or ""):
            if c != "C":
                cc[c] = cc.get(c, 0) + 1
    real = sorted((c for c, n in cc.items() if n >= thresh), key=lambda c: -cc[c])
    return cc, real


def _grade(score):
    for cut, g in [(0.85, "A"), (0.72, "B"), (0.58, "C"), (0.45, "D")]:
        if score >= cut:
            return g
    return "F"


def diagnose(pool, cards, play_model, n_spells_cap: int = 23) -> dict:
    """`pool` = global card indices drafted; `cards` = per-index dicts (deck_value, cmc, ci, t,
    mana_cost, name); `play_model` = a PlayModel / PlayTreeModel. Returns a report dict."""
    types = [c.get("t") for c in cards]
    # 1. build the recommended deck (learned buildability), then trim to 40-lands spells
    ranked = deck_from_play_model(play_model, pool, types, n_spells_cap)
    lands, avg_cmc = _land_count(ranked[:n_spells_cap], cards)
    deck = ranked[: 40 - lands]

    # 2. scores ------------------------------------------------------------------------------------
    dv_deck = [cards[i].get("deck_value") for i in deck if cards[i].get("deck_value") is not None]
    power_mean = float(np.mean(dv_deck)) if dv_deck else 0.0
    set_dv = np.array([c["deck_value"] for c in cards
                       if c.get("t") != "land" and c.get("deck_value") is not None])
    power_pct = float((set_dv < power_mean).mean()) if len(set_dv) else 0.5   # percentile vs the set
    cast = castability_score(deck, cards, n_lands=lands)
    perturn = per_turn_castability(deck, cards, n_lands=lands)
    cc, real = _colors(deck, cards)
    coherence = (sum(sorted(cc.values(), reverse=True)[:2]) / sum(cc.values())) if cc else 1.0
    overall = 0.4 * power_pct + 0.35 * cast + 0.25 * coherence            # illustrative blend

    # 3. advice ------------------------------------------------------------------------------------
    nonland = [i for i in dict.fromkeys(pool) if types[i] != "land"]
    probs = dict(zip(nonland, play_model.probs(pool, nonland)))
    deck_set = set(deck)
    on_color = lambda i: set(cards[i].get("ci") or "") - {"C"} <= set(real)   # castable in the deck's colors
    weak_in_deck = sorted((i for i in deck if probs.get(i, 1.0) < 0.5), key=lambda i: probs[i])[:3]
    strong_left = sorted((i for i in nonland if i not in deck_set and on_color(i) and probs.get(i, 0.0) > 0.6),
                         key=lambda i: -probs[i])[:3]
    advice = []
    if len(real) > 2:
        advice.append(f"{len(real)} colors in the deck ({'/'.join(real)}) — consider cutting to the "
                      "best 2 for consistency.")
    weak_turns = sorted(((t, p) for t, p in perturn.items() if p < 0.7), key=lambda x: x[1])[:3]
    if weak_turns:
        advice.append("shaky on-curve castability at turn " + ", ".join(str(t) for t, _ in weak_turns)
                      + " — add colored sources or lower the curve there.")
    for i in weak_in_deck:
        advice.append(f"borderline include — {cards[i].get('name')} (P(played)={probs[i]:.2f}): the "
                      "build model would often cut it.")
    for i in strong_left:
        advice.append(f"consider running — {cards[i].get('name')} (P(played)={probs[i]:.2f}): left in "
                      "your sideboard.")
    if not advice:
        advice.append("clean build — coherent colors, on-curve, no obvious cuts/adds.")

    return {
        "deck": [cards[i].get("name") for i in deck],
        "n_spells": len(deck), "lands": lands, "avg_cmc": round(avg_cmc, 2),
        "colors": real,
        "power_mean": round(power_mean, 4), "power_pct": round(power_pct, 3),
        "castability": round(cast, 3), "coherence": round(coherence, 3),
        "per_turn_castability": {t: round(p, 3) for t, p in perturn.items()},
        "overall": round(overall, 3), "grade": _grade(overall),
        "advice": advice,
        "_note": "grade is an illustrative heuristic blend, not a validated win-rate metric",
    }
