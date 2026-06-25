"""Is there pool-dependent structure (curve/castability/synergy) that predicts WINNING beyond the
linear per-card deck-value model? This bounds how much any pool-conditioned drafting objective could buy.

Two tests on a set's game_data (decks -> won):
  1. NONLINEARITY bound — gradient boosting (can use any card interaction) vs the linear deck_value
     model, held-out log-loss/AUC. If GBM doesn't beat linear, no interaction is broadly predictive.
  2. SYNERGY (targeted) — more sensitive than the aggregate GBM to *specific* card pairs: fit linear,
     rank card pairs by how much their co-presence makes the linear model UNDER-predict wins (positive
     residual), add the top pairs as interaction features, and check whether that improves HELD-OUT
     prediction (selection on train, validation on test = honest). Reports the top synergy pairs.

Caveat: the game_data only has decks humans actually BUILT (castable, curve-sane) — so castability/curve
are range-censored (the bad case never appears) and can't show up here; synergy is NOT censored.

    uv run python scripts/probe_deck_outcome.py --set DSK
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--top-cards", type=int, default=90, help="restrict synergy pairs to the N most-played cards")
    ap.add_argument("--top-pairs", type=int, default=60, help="how many candidate synergy pairs to add")
    ap.add_argument("--min-support", type=int, default=150, help="min decks with both cards (train)")
    a = ap.parse_args(argv)
    import pathlib

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    npz = a.npz or str(next(iter(pathlib.Path("data/game").glob(f"game.{a.set_code}.*.npz"))))
    z = np.load(npz, allow_pickle=True)
    X, y, C = z["X"].astype(np.float32), z["y"].astype(int), z["C"].astype(np.float32)
    names = list(z["card_names"])
    F = np.hstack([X, C])
    n = len(y); rng = np.random.default_rng(0); idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    print(f"{a.set_code}: games={n}, deck-card features={X.shape[1]}, base win rate={y.mean():.3f}\n")

    # 1. nonlinearity bound -----------------------------------------------------------------------
    lin = LogisticRegression(C=0.1, max_iter=400).fit(F[tr], y[tr])
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, early_stopping=True).fit(F[tr], y[tr])
    p_lin = lin.predict_proba(F[te])[:, 1]
    print("[1] nonlinearity bound (held-out):")
    print(f"    linear (deck_value family)   log-loss={log_loss(y[te], p_lin):.4f}  AUC={roc_auc_score(y[te], p_lin):.4f}")
    pg = gbm.predict_proba(F[te])[:, 1]
    print(f"    gradient boosting (any inter) log-loss={log_loss(y[te], pg):.4f}  AUC={roc_auc_score(y[te], pg):.4f}")

    # 2. targeted synergy -------------------------------------------------------------------------
    resid = y[tr] - lin.predict_proba(F[tr])[:, 1]                 # linear residual on train
    freq = (X[tr] > 0).sum(0)
    top = np.argsort(-freq)[: a.top_cards]
    pres = {c: (X[tr][:, c] > 0) for c in top}
    cand = []
    for ci_, cj in itertools.combinations(top, 2):
        both = pres[ci_] & pres[cj]
        s = int(both.sum())
        if s < a.min_support:
            continue
        syn = resid[both].mean() - resid[~both].mean()            # >0 => pair over-performs linear
        cand.append((syn, s, ci_, cj))
    cand.sort(reverse=True)
    pairs = [(ci_, cj) for _, _, ci_, cj in cand[: a.top_pairs]]
    print(f"\n[2] synergy: {len(cand)} candidate pairs (support>={a.min_support}); adding top {len(pairs)} interactions")

    def inter(rowsX):
        return np.stack([((rowsX[:, ci_] > 0) & (rowsX[:, cj] > 0)).astype(np.float32) for ci_, cj in pairs], 1)
    Ftr2 = np.hstack([F[tr], inter(X[tr])]); Fte2 = np.hstack([F[te], inter(X[te])])
    lin2 = LogisticRegression(C=0.1, max_iter=400).fit(Ftr2, y[tr])
    p2 = lin2.predict_proba(Fte2)[:, 1]
    print(f"    linear + synergy interactions log-loss={log_loss(y[te], p2):.4f}  AUC={roc_auc_score(y[te], p2):.4f}")
    print(f"    Δ held-out log-loss vs linear: {log_loss(y[te], p2) - log_loss(y[te], p_lin):+.4f} "
          f"(negative = synergy helps)")
    print("    top synergy pairs (linear under-predicts when both are in the deck):")
    for syn, s, ci_, cj in cand[:8]:
        print(f"      +{syn:.3f}  (n={s})  {names[ci_][:26]:26s} + {names[cj][:26]}")


if __name__ == "__main__":
    main()
