"""Mechanistic castability / curve-FUNCTION model — Frank-Karsten-style hypergeometric mana math.

Given a built deck (spells with cmc + colored pip requirements) and a mana base (land count +
colored sources), compute P(the deck FUNCTIONS) = the probability you can pay each spell's mana on
or near its curve turn, from your *opening-region* card draws. NO outcome / win-rate data is used —
this is pure combinatorics on the decklist, which is exactly why it sidesteps the game_data
censoring (game_data only contains decks humans actually built, i.e. already curve-sane).

Reference: Frank Karsten, "How Many Lands / Colored Sources Do You Need" (2018/2022). The core is a
hypergeometric draw: in a `D`-card deck, having drawn `seen` cards by a given turn (on the play:
7 + turn-1 = 7 + (turn-1) for turn T, i.e. you see 7 + (T-1) cards), the chance of holding at least
`k` lands / at least `r` sources of a color is 1 - hypergeom_cdf(k-1; D, successes, seen).

We score each spell by P(castable on-curve) and aggregate. The deck-level score is the mean over
spells (creatures weighted slightly higher since tempo matters), in [0,1].
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

COLORS = ("W", "U", "B", "R", "G")
_PIP_RE = re.compile(r"\{([^}]+)\}")


def _comb(n: int, k: int) -> float:
    if k < 0 or k > n or n < 0:
        return 0.0
    return math.comb(n, k)


def hypergeom_at_least(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) drawing n from N with K successes (hypergeometric upper tail)."""
    if k <= 0:
        return 1.0
    if K < k or n < k:
        return 0.0
    denom = _comb(N, n)
    if denom == 0:
        return 0.0
    # sum P(X = x) for x in [k, min(K, n)]
    total = 0.0
    for x in range(k, min(K, n) + 1):
        total += _comb(K, x) * _comb(N - K, n - x)
    return total / denom


def parse_pips(mana_cost: str | None) -> dict[str, int]:
    """Parse a Scryfall mana-cost string ('{2}{U}{U}') -> colored-pip counts {'U':2,...}.

    Generic/numeric/X symbols contribute nothing to *color* requirements. Hybrid '{W/U}' counts as
    a 'soft' half-pip toward EACH of its colors (either source casts it) — handled by the caller via
    a reduced requirement; here we record it under a special key so the scorer can be lenient.
    """
    pips: dict[str, int] = dict.fromkeys(COLORS, 0)
    hybrid: list[tuple[str, ...]] = []
    if not mana_cost:
        return pips
    for sym in _PIP_RE.findall(mana_cost):
        s = sym.upper()
        if "/" in s:  # hybrid like W/U, 2/W, W/P (phyrexian)
            parts = [p for p in s.split("/") if p in COLORS]
            if parts:
                hybrid.append(tuple(parts))
            continue
        if s in COLORS:
            pips[s] += 1
    pips["_hybrid"] = hybrid  # type: ignore[assignment]
    return pips


def card_mana_cost(card: dict) -> str:
    """The castable (front) face's mana cost, MDFC/DFC-safe.

    Scryfall puts an EMPTY top-level `mana_cost` on double-faced cards; the spell's real cost lives in
    `card_faces[0].mana_cost`. Without this fallback an MDFC spell parses to no pips (treated as
    colorless/free), silently inflating its castability — a real bug on MDFC-heavy sets (MOM/MID/…),
    harmless on sets with none (DSK). Lands return '' (no colored requirement)."""
    mc = card.get("mana_cost")
    if mc:
        return mc
    if card.get("t") != "land":
        faces = card.get("card_faces")
        if faces:
            return faces[0].get("mana_cost") or ""
    return mc or ""


# Frank Karsten's recommended sources for "cast on curve ~90% of the time" in a 40-card deck,
# keyed by (turn the pip is needed, number of identical colored pips). Used only as a SANITY anchor
# in tests; the live model computes the hypergeometric value directly so it generalizes to any
# turn / source count rather than table-lookup.
KARSTEN_40 = {
    (1, 1): 14, (2, 1): 13, (3, 1): 12, (4, 1): 11, (5, 1): 10, (6, 1): 9, (7, 1): 9,
    (2, 2): 20, (3, 2): 18, (4, 2): 16, (5, 2): 15, (6, 2): 14, (7, 2): 13,
    (3, 3): 23, (4, 3): 22, (5, 3): 20, (6, 3): 19, (7, 3): 18,
}


# Karsten's published source tables are mulligan-adjusted: they target ~90% *conditional on keeping a
# castable hand*, which empirically corresponds to discounting ~2 cards from the raw "cards seen by
# turn T" figure. With MULL_DISCOUNT=2 the model reproduces Karsten's 40-card single-pip table to ±2
# sources (exact-to-±1 for turns 1-5; a 1-2 source under-estimate at the turn 6-7 tail — see tests).
# Set to 0 for the raw, un-mulliganed hypergeometric (a slight over-estimate).
MULL_DISCOUNT = 2


def seen_cards(turn: int, deck_size: int = 40, on_play: bool = True,
               mull_discount: int = MULL_DISCOUNT) -> int:
    """Cards effectively seen by `turn`: 7 opening + 1/turn drawn, minus a mulligan discount.

    On the play you skip the turn-1 draw. `mull_discount` (~2) calibrates to Karsten's published
    'cast on curve ~90%' tables, which assume you mulligan unkeepable hands."""
    drawn = (turn - 1) if on_play else turn
    return max(1, min(deck_size, 7 + drawn - mull_discount))


def p_color_ok(deck_size: int, sources: int, needed: int, turn: int, on_play: bool = True) -> float:
    """P(>= `needed` sources of a color in hand by `turn`)."""
    return hypergeom_at_least(needed, deck_size, sources, seen_cards(turn, deck_size, on_play))


def p_lands_ok(deck_size: int, lands: int, needed: int, turn: int, on_play: bool = True) -> float:
    """P(>= `needed` lands by `turn`) — the generic mana / land-drop requirement."""
    return hypergeom_at_least(needed, deck_size, lands, seen_cards(turn, deck_size, on_play))


@lru_cache(maxsize=None)
def _joint_castable(deck_size: int, seen: int, need_lands: int, needed: tuple,
                    other_lands: int, nonlands: int) -> float:
    """EXACT multivariate-hypergeometric P(>= `need_lands` total lands AND >= m_i of each colored-source
    group i) within `seen` draws. `needed` = tuple of (group_size, min_count) per required color.

    This is the fix for the old `P(lands>=cmc) * prod P(sources>=pips)` factorization: a colored source
    IS a land, so the two requirements are positively correlated and multiplying them as independent
    DOUBLE-COUNTS the land draw and systematically under-states castability (worst for color-concentrated
    decks, which compressed the coherent-vs-rainbow gap). The deck is partitioned into the colored-source
    groups + an 'other lands' group + nonlands; we sum the joint upper tail exactly. Cached because (curve,
    pips, manabase) combos repeat heavily across spells/decks."""
    denom = _comb(deck_size, seen)
    if denom == 0.0:
        return 0.0

    def rec(i: int, drawn_land: int, ways: float) -> float:
        if i == len(needed):
            total = 0.0
            for y in range(0, min(other_lands, seen - drawn_land) + 1):
                rest = seen - drawn_land - y                  # nonland draws fill the remainder
                if 0 <= rest <= nonlands and drawn_land + y >= need_lands:
                    total += ways * _comb(other_lands, y) * _comb(nonlands, rest)
            return total
        size, need = needed[i]
        s = 0.0
        for x in range(need, min(size, seen - drawn_land) + 1):
            s += rec(i + 1, drawn_land + x, ways * _comb(size, x))
        return s

    return rec(0, 0, 1.0) / denom


def spell_castability(cmc, pips, deck_size, lands, sources, on_play=True, max_turn=7):
    """P(this spell is castable on/near its curve turn).

    turn = its CMC (capped at max_turn; a 7+ drop is judged at turn 7's draws). Requirement:
      - >= turn total lands in play (generic mana / land drops), AND
      - for each color, >= (colored pips of that color) sources of that color.
    Computed as the EXACT joint hypergeometric over {colored-source groups, other lands, nonlands}
    (see _joint_castable) — colored sources double as lands, so this does NOT multiply a separate
    P(lands>=cmc) against the colored terms. Hybrid pips ('{W/U}') are satisfied if EITHER color is
    present; we fold them in with a lenient max() factor (a minor independence approx for the rarer
    hybrid case only).
    """
    turn = min(int(round(cmc or 0)) or 1, max_turn)
    seen = seen_cards(turn, deck_size, on_play)
    needed = tuple(sorted((int(sources.get(col, 0)), int(pips.get(col, 0)))
                          for col in COLORS if pips.get(col, 0) > 0))
    other_lands = max(0, int(lands) - sum(s for s, _ in needed))
    nonlands = max(0, int(deck_size) - int(lands))
    p = _joint_castable(int(deck_size), int(seen), turn, needed, other_lands, nonlands)
    for hy in pips.get("_hybrid", []):  # one of these colors suffices -> best available source prob
        p *= max(p_color_ok(deck_size, sources.get(c, 0), 1, turn, on_play) for c in hy)
    return p


def infer_manabase(deck_idx, cards, n_lands):
    """Heuristic mana base from the spells' colored requirements when no lands are chosen.

    Counts colored pips across nonland spells, allocates `n_lands` proportionally to pip demand
    (Karsten's 'sources proportional to pips' rule), min 0. Returns {color: sources}. A dual/any
    land would add to multiple colors; here basics only -> each land is one color.
    """
    demand = dict.fromkeys(COLORS, 0)
    for i in deck_idx:
        c = cards[i]
        if c.get("t") == "land":
            continue
        for col, k in parse_pips(card_mana_cost(c)).items():
            if col in COLORS:
                demand[col] += k
    tot = sum(demand.values())
    if tot == 0:
        return {col: 0 for col in COLORS}
    # largest-remainder allocation of n_lands across colors by demand share
    raw = {col: n_lands * demand[col] / tot for col in COLORS}
    alloc = {col: int(math.floor(raw[col])) for col in COLORS}
    rem = n_lands - sum(alloc.values())
    for col in sorted(COLORS, key=lambda c: -(raw[c] - alloc[c]))[:rem]:
        alloc[col] += 1
    return alloc


def castability_score(deck_idx, cards, n_lands=17, sources=None, deck_size=40,
                      on_play=True, creature_weight=1.0):
    """Deck-level P(functions) in [0,1]: mean per-spell on-curve castability.

    `deck_idx`  global indices of the NONLAND spells in the built deck (lands excluded).
    `cards`     per-index metadata dicts with keys: cmc, t, mana_cost (Scryfall string), ci.
    `n_lands`   land count of the mana base; `sources` optional {color: count}; if None it's
                inferred from the deck's color demand (basics, proportional to pips).
    `deck_size` total cards (spells + lands), default 40 for limited.
    Returns the (optionally creature-weighted) mean spell castability.
    """
    spells = [i for i in deck_idx if cards[i].get("t") != "land"]
    if not spells:
        return 0.0
    if sources is None:
        sources = infer_manabase(deck_idx, cards, n_lands)
    num = den = 0.0
    for i in spells:
        c = cards[i]
        pips = parse_pips(card_mana_cost(c))
        w = creature_weight if c.get("t") == "creature" else 1.0
        num += w * spell_castability(c.get("cmc") or 0, pips, deck_size, n_lands, sources, on_play)
        den += w
    return num / den if den else 0.0


def per_turn_castability(deck_idx, cards, n_lands=17, sources=None, deck_size=40, on_play=True):
    """Helper: average castability of the spells at each CMC bucket (1..7) — a curve diagnostic."""
    if sources is None:
        sources = infer_manabase(deck_idx, cards, n_lands)
    out = {}
    for t in range(1, 8):
        bucket = [i for i in deck_idx if cards[i].get("t") != "land"
                  and min(int(round(cards[i].get("cmc") or 0)) or 1, 7) == t]
        if not bucket:
            continue
        out[t] = sum(spell_castability(cards[i].get("cmc") or 0, parse_pips(card_mana_cost(cards[i])),
                     deck_size, n_lands, sources, on_play) for i in bucket) / len(bucket)
    return out
