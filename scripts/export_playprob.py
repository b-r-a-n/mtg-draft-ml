"""Export the BUILD-time P(played|pool) buildability model per webapp set, as compact JSON trees the
browser can run (a ~12-line tree-walker), so the webapp deckbuild can use the learned buildability
deckbuilder (deck_from_play_model, the validated +0.049 deck-WR / ~2-color path) instead of the
context-free deck_value sort.

The deployed pick-model ONNX is unchanged — this only adds webapp/model/<SET>.playprob.json + flips a
meta flag, so NO re-export / NO pod. Reads local game_data (data/raw/game.<SET>.*.csv).

Why JSON trees and not ONNX: the model is a sklearn HistGradientBoostingClassifier; the deckbuilder only
needs the candidate RANKING, so we dump the tree ensemble and rank by raw margin in JS (no onnxruntime-web
ML-op dependency). Verified: the dumped trees reproduce gbm.decision_function exactly.

    uv run python scripts/export_playprob.py                 # all webapp sets
    uv run python scripts/export_playprob.py --sets DSK,OTJ  # a subset
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
import sklearn

from mtg_draft_ml.eval.play_prob import deck_from_play_model, train_play_model

WEB = pathlib.Path("webapp")
NODE_FIELDS = ("is_leaf", "value", "feature_idx", "num_threshold", "left", "right", "missing_go_to_left")


def _manifest(set_code):
    g = sorted(glob.glob(f"data/hf/manifests/{set_code}.PremierDraft.sample*.json"))
    return g[0] if g else None


def _game_csv(set_code):
    # prefer the largest sample (most decks)
    g = sorted(glob.glob(f"data/raw/game.{set_code}.*.csv"),
               key=lambda p: pathlib.Path(p).stat().st_size)
    return g[-1] if g else None


def _extract_trees(gbm):
    """HistGradientBoostingClassifier -> {base, trees:[{leaf,val,f,thr,l,r,ml}]} (binary)."""
    base = float(np.ravel(gbm._baseline_prediction)[0])
    trees = []
    for stage in gbm._predictors:
        nd = stage[0].nodes
        for f in NODE_FIELDS:
            assert f in nd.dtype.names, f"sklearn node dtype missing {f!r} (sklearn {sklearn.__version__})"
        trees.append({
            "leaf": nd["is_leaf"].astype(bool).astype(int).tolist(),
            "val": nd["value"].astype(float).tolist(),
            "f": nd["feature_idx"].astype(int).tolist(),
            "thr": nd["num_threshold"].astype(float).tolist(),
            "l": nd["left"].astype(int).tolist(),
            "r": nd["right"].astype(int).tolist(),
            "ml": nd["missing_go_to_left"].astype(bool).astype(int).tolist(),
        })
    return base, trees


def _margin(row, base, trees):
    """Pure-python mirror of the JS tree-walker — used for the parity self-check only."""
    s = base
    for t in trees:
        i = 0
        while not t["leaf"][i]:
            x = row[t["f"][i]]
            if x != x:  # NaN
                i = t["l"][i] if t["ml"][i] else t["r"][i]
            else:
                i = t["l"][i] if x <= t["thr"][i] else t["r"][i]
        s += t["val"][i]
    return s


def _deck_from_trees(pm, base, trees, pool, types, n_spells):
    """Build the deck from the DUMPED trees (mirrors deck_from_play_model via the JS-equivalent walker
    + the exact 21-feature builder), for parity vs the real sklearn path."""
    nonland = [c for c in dict.fromkeys(pool) if types[c] != "land"]
    if not nonland:
        return []
    pf = pm.pool_feats(pool)                                  # [12]
    feats = np.hstack([pm.cardfeat[nonland], np.tile(pf, (len(nonland), 1))])  # [n,21]
    margins = np.array([_margin(r, base, trees) for r in feats])
    order = np.argsort(-margins, kind="stable")              # stable tie-break (mirror in JS)
    return [nonland[i] for i in order[:n_spells]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", default=None, help="comma list; default = all webapp sets")
    ap.add_argument("--max-drafts", type=int, default=12000)
    ap.add_argument("--n-spells", type=int, default=23)
    a = ap.parse_args(argv)

    sets = ([s.strip() for s in a.sets.split(",")] if a.sets
            else sorted(p.name.split(".")[0] for p in (WEB / "data").glob("*.meta.json")))
    print(f"exporting playprob for: {sets}  (sklearn {sklearn.__version__})")
    ok = []
    for s in sets:
        man, csv = _manifest(s), _game_csv(s)
        if not man or not csv:
            print(f"  {s}: SKIP (manifest={bool(man)} csv={bool(csv)})")
            continue
        cards = json.load(open(WEB / "data" / f"{s}.cards.json"))["cards"]
        pm = train_play_model(csv, man, cards, max_drafts=a.max_drafts)
        base, trees = _extract_trees(pm.gbm)

        # --- parity self-check: dumped trees must reproduce gbm + the deck builder ---
        rng = np.random.default_rng(0)
        types = [c["t"] for c in sorted(cards, key=lambda c: c["i"])]
        present = [c["i"] for c in cards if c["t"] != "land"]
        max_prob_diff, deck_mismatch = 0.0, 0
        for _ in range(20):
            pool = list(rng.choice(present, size=min(45, len(present)), replace=False))
            cand = [c for c in dict.fromkeys(pool) if types[c] != "land"]
            ref = pm.probs(pool, cand)                                    # sklearn proba
            pf = pm.pool_feats(pool)
            feats = np.hstack([pm.cardfeat[cand], np.tile(pf, (len(cand), 1))])
            mine = 1 / (1 + np.exp(-np.array([_margin(r, base, trees) for r in feats])))
            max_prob_diff = max(max_prob_diff, float(np.max(np.abs(mine - ref))))
            # stable-sorted sklearn reference (deck_from_play_model uses a NON-stable sort, so it can
            # differ from any stable walker by a behaviorally-irrelevant tie-break at the cutoff)
            nl = [c for c in dict.fromkeys(pool) if types[c] != "land"]
            d_ref = [nl[i] for i in np.argsort(-pm.probs(pool, nl), kind="stable")[:a.n_spells]]
            d_mine = _deck_from_trees(pm, base, trees, pool, types, a.n_spells)
            deck_mismatch += (set(d_ref) != set(d_mine))

        bundle = {
            "set": s, "base": base, "trees": trees, "n_features": 21, "n_spells": a.n_spells,
            "feature_spec": ("per-card[cmc,deck_value,W,U,B,R,G,creature,land]"
                             "+pool[W,U,B,R,G,cmc1..6,size]"),
            "colors": ["W", "U", "B", "R", "G"], "sklearn_version": sklearn.__version__,
            "n_trees": len(trees),
        }
        (WEB / "model" / f"{s}.playprob.json").write_text(json.dumps(bundle, separators=(",", ":")))
        size = (WEB / "model" / f"{s}.playprob.json").stat().st_size / 1e6
        print(f"  {s}: {len(trees)} trees, {size:.2f} MB · parity max|Δp|={max_prob_diff:.2e} "
              f"deck_mismatch={deck_mismatch}/20")
        assert max_prob_diff < 1e-9 and deck_mismatch == 0, f"{s}: parity FAILED"
        ok.append(s)
    print(f"\nwrote playprob.json for {len(ok)}/{len(sets)} sets: {ok}")


if __name__ == "__main__":
    main()
