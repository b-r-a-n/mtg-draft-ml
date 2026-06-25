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


def estimated_deck_wr(pool, beta, intercept: float = 0.0, n_spells: int = 23) -> float:
    """sigma(intercept + sum of the top-n_spells card betas in `pool`).

    `pool` is a list of global card indices (repeats allowed = multiple copies); `beta` is the
    deck-value coefficient per global index (NaN where unknown — skipped). The top-N by beta proxy
    the spells you'd actually run (lands have low/negative beta and fall out naturally).
    """
    vals = sorted((float(beta[i]) for i in pool if np.isfinite(beta[i])), reverse=True)[:n_spells]
    return 1.0 / (1.0 + np.exp(-(intercept + sum(vals))))


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


def run_eval(drafts, policies: dict, beta, intercept: float = 0.0, n_spells: int = 23) -> dict:
    """Score every policy's drafted pool by estimated deck-WR over `drafts`.

    `policies` maps name -> pick_fn(pool, pack, step). Returns per-policy mean estimated deck-WR and,
    for each non-human policy, the fraction of drafts where it beats 'human' (if present).
    """
    pools = {name: [replay(d, fn) for d in drafts] for name, fn in policies.items()}
    wr = {name: np.array([estimated_deck_wr(p, beta, intercept, n_spells) for p in ps])
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
