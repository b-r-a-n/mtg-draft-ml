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


def train_play_model(csv, manifest_path, cards, max_drafts: int = 12000) -> PlayModel:
    """Fit P(played | pool) on a set's game_data (deck vs sideboard, one row per built deck)."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    name_to_idx = {c["name"]: c["index"] for c in json.load(open(manifest_path))["cards"]}
    n = len(name_to_idx)
    cmc, beta, cmask, cmcb, creat, land = _card_arrays(cards, n)
    cardfeat = np.column_stack([cmc, beta, cmask.T, creat, land])
    head = pd.read_csv(csv, nrows=0).columns.tolist()
    deck_cols = [c for c in head if c.startswith("deck_")]
    side_cols = ["sideboard_" + c[len("deck_"):] for c in deck_cols]
    df = (pd.read_csv(csv, usecols=["draft_id"] + deck_cols + side_cols)
          .drop_duplicates("draft_id").head(max_drafts))
    gidx = np.array([name_to_idx[c[len("deck_"):]] for c in deck_cols])
    deckM = df[deck_cols].fillna(0).to_numpy(np.int16); sideM = df[side_cols].fillna(0).to_numpy(np.int16)

    Xc, Xp, Y = [], [], []
    for r in range(len(df)):
        pool = np.zeros(n); played = np.zeros(n, bool)
        pool[gidx] = deckM[r] + sideM[r]; played[gidx] = deckM[r] > 0
        pres = np.flatnonzero(pool > 0)
        if len(pres) < 15:
            continue
        poolcolor = cmask @ pool; poolcmc = np.array([pool[cmcb == b].sum() for b in range(1, 7)])
        pf = np.concatenate([poolcolor, poolcmc, [pool.sum()]])
        Xc.append(cardfeat[pres]); Xp.append(np.tile(pf, (len(pres), 1))); Y.append(played[pres])
    X = np.hstack([np.vstack(Xc), np.vstack(Xp)]); Y = np.concatenate(Y).astype(int)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0,
                                         early_stopping=True).fit(X, Y)
    return PlayModel(gbm, cmc, beta, cmask, cmcb, creat, land)
