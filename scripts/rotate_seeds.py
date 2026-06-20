#!/usr/bin/env python3
"""Robustness study: rotate the held-out set x multiple seeds (best config).

Runs LOSO (set_transformer + ce + aux-WR=1.0, MiniLM) holding out each set in turn, across
several seeds, and reports per-holdout and overall mean +/- std. This tightens the single-seed,
single-holdout Phase-1/2/3 numbers. Results -> docs/results/phase4-robustness.md.

Usage: python scripts/rotate_seeds.py [data_dir]
"""
import json
import statistics
import sys

from mtg_draft_ml.eval.generalization import run_loso

P = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1proc"
SIZE = "60000"
SETS = ["BLB", "OTJ", "WOE", "MKM", "DSK"]
SEEDS = [0, 1, 2]


def spec(s: str) -> dict:
    return {
        "parquet": f"{P}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
        "manifest": f"{P}/manifests/{s}.PremierDraft.sample{SIZE}.json",
        "scryfall": f"{P}/scryfall_{s.lower()}.json",
        "ratings": f"{P}/ratings/{s}.PremierDraft.ratings.json",
    }


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), 0.0)
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


rows = []
for seed in SEEDS:
    for ho in SETS:
        train = [spec(s) for s in SETS if s != ho]
        hold = spec(ho)
        r = run_loso(
            train,
            {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
            embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce",
            aux_wr=1.0, holdout_ratings=hold["ratings"], epochs=10, seed=seed,
            val_frac=0.05, checkpoint_dir="/tmp/rot_ck",
        )
        h = r["holdout"]
        rows.append({"seed": seed, "holdout": ho, "top1": h["top1"],
                     "novel_top1": h.get("novel_top1"), "wr_model": h.get("wr_agreement_model"),
                     "wr_human": h.get("wr_agreement_human")})
        print(f">>> [seed {seed} | holdout {ho}] top1={h['top1']:.4f} "
              f"novel={h.get('novel_top1', 0):.4f} wr-agr={h.get('wr_agreement_model', 0):.4f}")

print("\n=== per-holdout (mean +/- std over seeds) ===")
for ho in SETS:
    t = ms([r["top1"] for r in rows if r["holdout"] == ho])
    nv = ms([r["novel_top1"] for r in rows if r["holdout"] == ho])
    wr = ms([r["wr_model"] for r in rows if r["holdout"] == ho])
    print(f"  {ho}: top1={t[0]:.4f}+/-{t[1]:.4f}  novel={nv[0]:.4f}  wr-agr={wr[0]:.4f}")

allt = ms([r["top1"] for r in rows])
alln = ms([r["novel_top1"] for r in rows])
allwr = ms([r["wr_model"] for r in rows])
allwrh = ms([r["wr_human"] for r in rows])
print(f"\n=== OVERALL (n={len(rows)} runs) ===")
print(f"  held-out top1 : {allt[0]:.4f} +/- {allt[1]:.4f}")
print(f"  novel-only    : {alln[0]:.4f} +/- {alln[1]:.4f}")
print(f"  WR-agreement  : model {allwr[0]:.4f}  human {allwrh[0]:.4f}")

json.dump(rows, open(f"{P}/rotate_results.json", "w"), indent=2, default=float)
print(f"wrote {P}/rotate_results.json")
