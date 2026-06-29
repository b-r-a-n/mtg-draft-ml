"""P(card is played | pool) — a learned buildability model from the deck-vs-sideboard build decision.

17lands game_data has deck_<card> (played) and sideboard_<card> (cut), so for every card in a drafter's
pool we know whether it made the deck. P(played | card, pool composition) is strongly pool-dependent
(see docs/results/play-prob.md) — a contextual "will this card make my deck" signal that captures color
commitment / curve / castability without a simulator or hand-coded features. This module trains it and
exposes per-pick inference so a drafting policy can discount picks that won't make the deck.
"""
from __future__ import annotations

import json

import numpy as np

COLORS = ["W", "U", "B", "R", "G"]


def _card_arrays(cards, n):
    cmc = np.zeros(n); beta = np.full(n, np.nan); cmask = np.zeros((5, n)); creat = np.zeros(n); land = np.zeros(n)
    for c in cards:
        i = c["i"]; cmc[i] = c["cmc"] or 0
        if c["deck_value"] is not None:
            beta[i] = c["deck_value"]
        for k, col in enumerate(COLORS):
            cmask[k, i] = col in (c["ci"] or "")
        creat[i] = c["t"] == "creature"; land[i] = c["t"] == "land"
    cmcb = np.clip(np.round(cmc).astype(int), 1, 6)
    return cmc, beta, cmask, cmcb, creat, land


class PlayModel:
    """Fitted P(played | pool). `.probs(pool, pack)` -> P(played) for each pack card given the pool."""

    def __init__(self, gbm, cmc, beta, cmask, cmcb, creat, land):
        self.gbm = gbm; self.cmc = cmc; self.beta = beta; self.cmask = cmask
        self.cmcb = cmcb; self.creat = creat; self.land = land
        self.cardfeat = np.column_stack([cmc, beta, cmask.T, creat, land])  # [n, 9]

    def pool_feats(self, pool) -> np.ndarray:
        v = np.zeros(len(self.cmc))
        for i in pool:
            v[i] += 1
        poolcolor = self.cmask @ v
        poolcmc = np.array([v[self.cmcb == b].sum() for b in range(1, 7)])
        return np.concatenate([poolcolor, poolcmc, [v.sum()]])             # [12]

    def probs(self, pool, pack) -> np.ndarray:
        pf = self.pool_feats(pool)
        F = np.hstack([self.cardfeat[list(pack)], np.tile(pf, (len(pack), 1))])
        return self.gbm.predict_proba(F)[:, 1]


class PlayTreeModel(PlayModel):
    """PlayModel-compatible inference from the EXPORTED JSON trees (webapp/model/<SET>.playprob.json,
    written by scripts/export_playprob.py) — no game_data / no retrain. Same 21-feature layout and the
    same `.probs(pool, pack)` interface, so `deck_from_play_model` and callers work unchanged; the tree
    walk mirrors the browser's (and reproduces the source HGB's `predict_proba` exactly)."""

    def __init__(self, bundle: dict, cards, n: int):
        cmc, beta, cmask, cmcb, creat, land = _card_arrays(cards, n)
        super().__init__(None, cmc, beta, cmask, cmcb, creat, land)
        self.base = float(bundle["base"])
        self.trees = bundle["trees"]

    def _margin(self, row) -> float:
        s = self.base
        for t in self.trees:
            i = 0
            while not t["leaf"][i]:
                x = row[t["f"][i]]
                i = (t["l"][i] if t["ml"][i] else t["r"][i]) if x != x \
                    else (t["l"][i] if x <= t["thr"][i] else t["r"][i])
            s += t["val"][i]
        return s

    def probs(self, pool, pack) -> np.ndarray:
        pf = self.pool_feats(pool)
        F = np.hstack([self.cardfeat[list(pack)], np.tile(pf, (len(pack), 1))])
        m = np.array([self._margin(r) for r in F])
        return 1.0 / (1.0 + np.exp(-m))


def load_play_tree_model(set_code: str, cards, n: int, webapp_dir: str = "webapp") -> "PlayTreeModel":
    """Load the exported `<SET>.playprob.json` trees into a PlayTreeModel (local, no game_data)."""
    bundle = json.load(open(f"{webapp_dir}/model/{set_code}.playprob.json"))
    return PlayTreeModel(bundle, cards, n)


def deck_from_play_model(pm, pool, types, n_spells: int = 23) -> list[int]:
    """LEARNED deckbuilder: fill n_spells nonland slots with the most-likely-played cards, by P(played | pool),
    KEEPING MULTIPLES — a pool with 2x Murder yields up to 2x Murder in the deck.

    At build time the pool is full (~45 cards), so P(played|pool) is in-distribution — it picks the deck
    a good player would build (color-coherent, on-curve) without any hand-coded color/curve heuristic.
    Both copies of a card share the same feature vector (same P(played)), so we rank UNIQUE cards and then
    take each one's pool-count copies in ranked order, up to n_spells (a strong 2-of can crowd out a singleton).
    """
    from collections import Counter

    nonland = [c for c in dict.fromkeys(pool) if types[c] != "land"]   # unique pool cards, no lands
    if not nonland:
        return []
    probs = pm.probs(pool, nonland)
    order = [nonland[i] for i in np.argsort(-probs)]                   # unique, ranked desc by P(played)
    counts = Counter(c for c in pool if types[c] != "land")           # real copies in the pool
    deck: list[int] = []
    for c in order:
        deck.extend([c] * min(counts[c], n_spells - len(deck)))       # take its copies, clamp at the cutoff
        if len(deck) >= n_spells:
            break
    return deck


def _load_decks(csv, manifest_path, cards):
    """Returns (n, card_arrays, cardfeat, gidx, deckM, sideM) for the dedup'd built decks in a CSV."""
    import pandas as pd
    name_to_idx = {c["name"]: c["index"] for c in json.load(open(manifest_path))["cards"]}
    n = len(name_to_idx)
    arrs = _card_arrays(cards, n)
    cmc, beta, cmask, cmcb, creat, land = arrs
    cardfeat = np.column_stack([cmc, beta, cmask.T, creat, land])
    head = pd.read_csv(csv, nrows=0).columns.tolist()
    deck_cols = [c for c in head if c.startswith("deck_")]
    side_cols = ["sideboard_" + c[len("deck_"):] for c in deck_cols]
    df = (pd.read_csv(csv, usecols=["draft_id"] + deck_cols + side_cols)
          .drop_duplicates("draft_id").head(50_000))
    gidx = np.array([name_to_idx[c[len("deck_"):]] for c in deck_cols])
    return n, arrs, cardfeat, gidx, df[deck_cols].fillna(0).to_numpy(np.int16), df[side_cols].fillna(0).to_numpy(np.int16)


def _fit(Xc, Xp, Y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.hstack([np.vstack(Xc), np.vstack(Xp)]); Y = np.concatenate(Y).astype(int)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0,
                                          early_stopping=True).fit(X, Y)


def _pool_feats(pv, cmask, cmcb):
    return np.concatenate([cmask @ pv, [pv[cmcb == b].sum() for b in range(1, 7)], [pv.sum()]])


def train_play_model(csv, manifest_path, cards, max_drafts: int = 12000) -> PlayModel:
    """Fit P(played | FULL pool) — one row per (built deck, card in pool). For BUILD-time use (the pool
    is full and in-distribution): score a finished pool / build the deck."""
    n, (cmc, beta, cmask, cmcb, creat, land), cardfeat, gidx, deckM, sideM = _load_decks(csv, manifest_path, cards)
    Xc, Xp, Y = [], [], []
    for r in range(min(len(deckM), max_drafts)):
        pool = np.zeros(n); played = np.zeros(n, bool)
        pool[gidx] = deckM[r] + sideM[r]; played[gidx] = deckM[r] > 0
        pres = np.flatnonzero(pool > 0)
        if len(pres) < 15:
            continue
        pf = _pool_feats(pool, cmask, cmcb)
        Xc.append(cardfeat[pres]); Xp.append(np.tile(pf, (len(pres), 1))); Y.append(played[pres])
    return PlayModel(_fit(Xc, Xp, Y), cmc, beta, cmask, cmcb, creat, land)


def train_play_model_partial(csv, manifest_path, cards, max_drafts: int = 6000,
                             samples: int = 5, seed: int = 0) -> PlayModel:
    """Fit P(card eventually played | PARTIAL pool, candidate not yet in pool) — for PICK-time use.

    Subsample approximation of "current picks": for each built deck, draw `samples` random partial
    pools (k-subsets of the pool), and for each, the CANDIDATES are the pool cards NOT in the subset
    (cards you might still take), labelled by whether they made the deck. This is in-distribution for
    mid-draft pools (the full-pool model is OOD there — see docs/results/play-prob-pick-time.md)."""
    n, (cmc, beta, cmask, cmcb, creat, land), cardfeat, gidx, deckM, sideM = _load_decks(csv, manifest_path, cards)
    rng = np.random.default_rng(seed)
    Xc, Xp, Y = [], [], []
    for r in range(min(len(deckM), max_drafts)):
        pool = np.zeros(n); played = np.zeros(n, bool)
        pool[gidx] = deckM[r] + sideM[r]; played[gidx] = deckM[r] > 0
        pres = np.flatnonzero(pool > 0)
        if len(pres) < 20:
            continue
        for _ in range(samples):
            k = int(rng.integers(6, min(len(pres), 40)))
            sub = rng.choice(pres, size=k, replace=False)
            insub = np.zeros(n, bool); insub[sub] = True
            cand = pres[~insub[pres]]                                   # candidates: not yet in pool
            if len(cand) == 0:
                continue
            pv = np.zeros(n); pv[sub] = pool[sub]                       # partial-pool composition
            pf = _pool_feats(pv, cmask, cmcb)
            Xc.append(cardfeat[cand]); Xp.append(np.tile(pf, (len(cand), 1))); Y.append(played[cand])
    return PlayModel(_fit(Xc, Xp, Y), cmc, beta, cmask, cmcb, creat, land)
