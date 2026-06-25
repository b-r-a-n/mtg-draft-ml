"""Outcome eval — score a drafted POOL by the game_data deck-value model (estimated deck win rate).

The honest adjudicator the WR-agreement metrics couldn't be (see docs/game-data-plan.md): instead of
asking "does the bot agree with a card rating", replay held-out drafts where a policy drafts a seat,
then score the resulting pool by the game_data logistic deck-value model — sigma(intercept + sum of
the best-N spell betas) ≈ the deck's win probability. Compares the model's drafted decks to the
*humans who actually drafted those seats*.

It is **non-circular** when the drafting model was trained on a target OTHER than deck_value (e.g. the
GIH-composite): a higher estimated deck-WR then means imitation-style drafting yields outcome-good
decks, not that the model was trained toward the scorer. Caveat: packs are on-rails (the recorded
sequence), so a policy's own picks don't change what wheels — a standard counterfactual approximation.
"""
from __future__ import annotations

import numpy as np

# A 23-nonland limited deck's CMC-bucket targets (1..6+), and the 10 two-color pairs.
DECK_CURVE = {1: 2, 2: 6, 3: 5, 4: 4, 5: 3, 6: 3}
PAIRS = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]


def _bucket(cmc: float) -> int:
    return min(max(int(round(cmc or 0)), 1), 6)


def build_best_deck(pool, beta, ci, cmc, types, n_spells: int = 23) -> list[int]:
    """Best legal **2-color, curve-respecting** deck (nonland spells) from a pool, by sum of beta.

    For each of the 10 two-color pairs: take the on-color nonland cards (color identity ⊆ pair, plus
    colorless), fill the `DECK_CURVE` CMC buckets with the highest-beta cards, then backfill remaining
    slots with the best leftovers; keep the pair with the highest total beta. This accounts for **color
    distribution** (a 5-color bomb pile collapses to one pair's depth) and **mana curve** (a pool with
    no early drops can't fill the low buckets) — unlike a naive global top-N. `ci`/`cmc`/`types` are
    per-global-index (color-identity string, CMC, 'land'/'creature'/'spell').
    """
    best_deck, best_score = [], -1e18
    for pair in PAIRS:
        ps = set(pair)
        onc = sorted({i for i in pool if np.isfinite(beta[i]) and types[i] != "land"
                      and (set(ci[i]) - {"C"}) <= ps}, key=lambda i: -beta[i])
        deck, filled = [], dict.fromkeys(DECK_CURVE, 0)
        for i in onc:                                       # fill the curve buckets, best beta first
            b = _bucket(cmc[i])
            if len(deck) < n_spells and filled[b] < DECK_CURVE[b]:
                deck.append(i); filled[b] += 1
        for i in onc:                                       # backfill remaining slots with leftovers
            if len(deck) >= n_spells:
                break
            if i not in deck:
                deck.append(i)
        score = sum(float(beta[i]) for i in deck)
        if score > best_score:
            best_deck, best_score = deck, score
    return best_deck


def estimated_deck_wr(pool, beta, intercept: float = 0.0, n_spells: int = 23,
                      ci=None, cmc=None, types=None) -> float:
    """sigma(intercept + sum of the played spells' betas).

    With `ci`/`cmc`/`types` given, the deck is the constrained best 2-color + curve deck
    (`build_best_deck`); otherwise the cards are the naive global top-N by beta (lands fall out
    naturally since their beta is low). `pool` is a list of global card indices.
    """
    if ci is not None:
        s = sum(float(beta[i]) for i in build_best_deck(pool, beta, ci, cmc, types, n_spells))
    else:
        s = sum(sorted((float(beta[i]) for i in pool if np.isfinite(beta[i])), reverse=True)[:n_spells])
    return 1.0 / (1.0 + np.exp(-(intercept + s)))


def load_drafts(parquet, limit: int | None = None) -> list[list[dict]]:
    """Group a holdout draft parquet into ordered drafts: per draft, a list of {pack, human} per pick
    (pick order). `pack` is the list of global card indices offered; `human` is the recorded pick."""
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["draft_id", "pack_number", "pick_number",
                                        "pack_indices", "pick_idx"]).to_pylist()
    by_draft: dict[str, list[dict]] = {}
    for r in t:
        by_draft.setdefault(r["draft_id"], []).append(r)
    drafts = []
    for rows in by_draft.values():
        rows.sort(key=lambda r: (r["pack_number"], r["pick_number"]))
        drafts.append([{"pack": list(r["pack_indices"]), "human": r["pick_idx"]} for r in rows])
        if limit and len(drafts) >= limit:
            break
    return drafts


def replay(draft, pick_fn) -> list[int]:
    """Accumulate a pool by calling pick_fn(pool, pack, step) at each pick. Returns the pool indices."""
    pool: list[int] = []
    for step in draft:
        pool.append(int(pick_fn(pool, step["pack"], step)))
    return pool


def run_eval(drafts, policies: dict, beta, intercept: float = 0.0, n_spells: int = 23,
             ci=None, cmc=None, types=None) -> dict:
    """Score every policy's drafted pool by estimated deck-WR over `drafts`.

    `policies` maps name -> pick_fn(pool, pack, step). With `ci`/`cmc`/`types` given, each pool is
    scored via the constrained best 2-color + curve deck. Returns per-policy mean estimated deck-WR
    and, for each non-human policy, the fraction of drafts where it beats 'human' (if present).
    """
    pools = {name: [replay(d, fn) for d in drafts] for name, fn in policies.items()}
    wr = {name: np.array([estimated_deck_wr(p, beta, intercept, n_spells, ci, cmc, types) for p in ps])
          for name, ps in pools.items()}
    out = {"n_drafts": len(drafts), "n_spells": n_spells,
           "mean": {k: float(v.mean()) for k, v in wr.items()},
           "std": {k: float(v.std()) for k, v in wr.items()}}
    if "human" in wr:
        out["beats_human_frac"] = {k: float((v > wr["human"]).mean())
                                   for k, v in wr.items() if k != "human"}
        out["delta_vs_human"] = {k: float((v - wr["human"]).mean())
                                 for k, v in wr.items() if k != "human"}
    return out


# ---- reference pick functions (model pick_fn lives with the ONNX glue in the runner) ------------
def human_pick(pool, pack, step):
    return step["human"]


def random_pick(rng):
    def fn(pool, pack, step):
        return pack[rng.integers(len(pack))]
    return fn


def greedy_pick(score):
    """Pick the highest-`score` card in the pack (score = per-global-idx array, NaN = avoid)."""
    def fn(pool, pack, step):
        vals = [(score[c] if np.isfinite(score[c]) else -np.inf) for c in pack]
        return pack[int(np.argmax(vals))]
    return fn
