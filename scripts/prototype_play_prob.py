"""Prototype: P(card is played | pool) — can the build-decision signal teach the pool-dependent dynamic
(color commitment / curve / castability) that `won`-over-composition couldn't?

17lands game_data has deck_<card> (played) AND sideboard_<card> (in pool, cut). So pool = deck+side and
played = (deck>0): the CUT decision is the human's build judgment, and it's *context-dependent* (a white
card gets cut from a green deck), not range-restricted, not single-game-noisy. We fit P(played) two
ways and ask whether the pool CONTEXT adds predictive power, then inspect whether the learned model
shows the castability dynamic (a white card's play-prob rises with the pool's white commitment).

  baseline : card features only (intrinsic playability)
  +context : card features + POOL composition (color counts, CMC histogram, size)

    uv run python scripts/prototype_play_prob.py --set DSK
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

COLORS = ["W", "U", "B", "R", "G"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-drafts", type=int, default=12000)
    ap.add_argument("--webapp-dir", default="webapp")
    ap.add_argument("--hf-dir", default="data/hf")
    a = ap.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score

    csv = a.csv or str(next(iter(pathlib.Path("data/raw").glob(f"game.{a.set_code}.*.csv"))))
    man = next(iter(pathlib.Path(f"{a.hf_dir}/manifests").glob(f"{a.set_code}.PremierDraft.sample*.json")))
    name_to_idx = {c["name"]: c["index"] for c in json.load(open(man))["cards"]}
    n = len(name_to_idx)
    cards = json.load(open(f"{a.webapp_dir}/data/{a.set_code}.cards.json"))["cards"]
    cmc = np.zeros(n); beta = np.full(n, np.nan); cmask = np.zeros((5, n)); creat = np.zeros(n); land = np.zeros(n)
    for c in cards:
        i = c["i"]; cmc[i] = c["cmc"] or 0
        if c["deck_value"] is not None:
            beta[i] = c["deck_value"]
        for k, col in enumerate(COLORS):
            cmask[k, i] = col in (c["ci"] or "")
        creat[i] = c["t"] == "creature"; land[i] = c["t"] == "land"
    cmcb = np.clip(np.round(cmc).astype(int), 1, 6)            # CMC bucket 1..6 per card

    # one row per built deck (dedup by draft_id, first build); pool = deck+sideboard, played = deck>0
    head = pd.read_csv(csv, nrows=0).columns.tolist()
    deck_cols = [c for c in head if c.startswith("deck_")]
    side_cols = ["sideboard_" + c[len("deck_"):] for c in deck_cols]
    use = ["draft_id"] + deck_cols + side_cols
    df = pd.read_csv(csv, usecols=use).drop_duplicates("draft_id").head(a.max_drafts)
    gidx = np.array([name_to_idx[c[len("deck_"):]] for c in deck_cols])  # deck-col -> global idx
    deckM = df[deck_cols].fillna(0).to_numpy(np.int16)
    sideM = df[side_cols].fillna(0).to_numpy(np.int16)

    cardfeat = np.column_stack([cmc, beta, cmask.T, creat, land])       # [n, 9]  intrinsic card features
    Xc, Xp, Y, draft_id = [], [], [], []
    for r in range(len(df)):
        pool = np.zeros(n); played = np.zeros(n, bool)
        pool[gidx] = deckM[r] + sideM[r]; played[gidx] = deckM[r] > 0
        present = np.flatnonzero(pool > 0)
        if len(present) < 15:
            continue
        poolcolor = cmask @ pool                                        # [5] pool cards per color
        poolcmc = np.array([pool[cmcb == b].sum() for b in range(1, 7)])  # [6] CMC histogram
        pf = np.concatenate([poolcolor, poolcmc, [pool.sum()]])         # [12] pool context
        Xc.append(cardfeat[present]); Xp.append(np.tile(pf, (len(present), 1)))
        Y.append(played[present]); draft_id.append(np.full(len(present), r))
    Xc = np.vstack(Xc); Xp = np.vstack(Xp); Y = np.concatenate(Y).astype(int); did = np.concatenate(draft_id)
    print(f"{a.set_code}: {len(df)} decks, {len(Y)} (deck,card-in-pool) rows, played rate={Y.mean():.3f}")

    # split by DRAFT (no leakage)
    rng = np.random.default_rng(0); drafts = np.unique(did); rng.shuffle(drafts)
    test_d = set(drafts[: len(drafts) // 5]); te = np.array([d in test_d for d in did]); tr = ~te
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0,
                                          early_stopping=True)
    ctx = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0,
                                         early_stopping=True)
    base.fit(Xc[tr], Y[tr]); ctx.fit(np.hstack([Xc, Xp])[tr], Y[tr])
    pb = base.predict_proba(Xc[te])[:, 1]; pc = ctx.predict_proba(np.hstack([Xc, Xp])[te])[:, 1]
    print("\nP(played) — does the POOL context add predictive power?")
    print(f"  baseline (card only)      log-loss={log_loss(Y[te], pb):.4f}  AUC={roc_auc_score(Y[te], pb):.4f}")
    print(f"  +context (card + pool)    log-loss={log_loss(Y[te], pc):.4f}  AUC={roc_auc_score(Y[te], pc):.4f}")

    # inspection: a median-beta WHITE 2-drop creature — does its play-prob rise with the pool's W count?
    wbeta = np.nanmedian([beta[i] for i in range(n) if cmask[0, i] and cmcb[i] == 2 and np.isfinite(beta[i])])
    card = np.array([2.0, wbeta, 1, 0, 0, 0, 0, 1, 0])                  # cmc2, white, creature
    print("\ninspection — P(play a median white 2-drop) vs the pool's WHITE commitment (deck size ~23):")
    for w in [2, 5, 9, 13, 17]:
        pf = np.array([w, 0, 0, 0, 23 - w, 2, 7, 6, 4, 2, 2, 40], float)  # poolW=w, poolG=23-w, a normal curve
        p = ctx.predict_proba(np.hstack([card, pf])[None])[0, 1]
        print(f"    pool has {w:2d} white cards (rest green):  P(played)={p:.3f}")


if __name__ == "__main__":
    main()
