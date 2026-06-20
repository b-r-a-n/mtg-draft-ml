#!/usr/bin/env python3
"""Data-scaling curve: does training on MORE sets improve held-out generalization?

Fix the holdout (DSK) and the best recipe (set_transformer + ce + aux-WR, MiniLM, raw features);
train on the first N sets of a fixed pool for N = 1..len(pool); report held-out top-1 per N.
Rising curve at the largest N => the full-corpus GPU run is justified; flat => it isn't.

Single seed (seed variance measured at ~±0.002, negligible). Usage: python scripts/scaling_curve.py
"""
import json
import sys

from mtg_draft_ml.eval.generalization import run_loso

P = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1proc"
SIZE = "60000"
POOL = ["BLB", "OTJ", "WOE", "MKM", "LCI", "MOM", "MH3"]   # added in this order
HOLDOUT = "DSK"


def spec(s: str) -> dict:
    return {
        "parquet": f"{P}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
        "manifest": f"{P}/manifests/{s}.PremierDraft.sample{SIZE}.json",
        "scryfall": f"{P}/scryfall_{s.lower()}.json",
        "ratings": f"{P}/ratings/{s}.PremierDraft.ratings.json",
    }


hold = spec(HOLDOUT)
rows = []
for n in range(1, len(POOL) + 1):
    train = [spec(s) for s in POOL[:n]]
    r = run_loso(
        train,
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce",
        aux_wr=1.0, holdout_ratings=hold["ratings"], epochs=10, seed=0,
        val_frac=0.05, checkpoint_dir="/tmp/scale_ck",
    )
    h = r["holdout"]
    rows.append({"n_sets": n, "train_picks": n * int(SIZE), "train_cards": r["train_cards"],
                 "holdout_top1": h["top1"], "novel_top1": h.get("novel_top1"),
                 "wr_agreement": h.get("wr_agreement_model")})
    print(f">>> n_sets={n} picks={n * int(SIZE)} cards={r['train_cards']} "
          f"top1={h['top1']:.4f} novel={h.get('novel_top1', 0):.4f}")

print(f"\n=== data-scaling curve (holdout {HOLDOUT}, 1 seed) ===")
print("  n_sets  picks    cards   holdout_top1   novel_top1")
for x in rows:
    print(f"  {x['n_sets']:>5}  {x['train_picks']:>7}  {x['train_cards']:>5}   "
          f"{x['holdout_top1']:.4f}        {(x['novel_top1'] or 0):.4f}")
json.dump(rows, open(f"{P}/scaling_curve.json", "w"), indent=2, default=float)
print(f"wrote {P}/scaling_curve.json")
