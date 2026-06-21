#!/usr/bin/env python3
"""Capacity probe: at the largest data point (7 sets), does a BIGGER model break the plateau?

The data-scaling curve saturated at ~3-4 sets for the small (~10M-param) model -> data-saturated,
not data-limited. If accuracy is capacity-limited, scaling the model UP at fixed (max) data should
help; if it's flat, we're at a task/noise ceiling and neither more data nor more model helps (here).

Fixed: 7 training sets -> holdout DSK, best recipe, single seed. Vary model size.
"""
import json
import os
import sys

from mtg_draft_ml.eval.generalization import run_loso

P = "/tmp/p1proc"
SIZE = "60000"
POOL = ["BLB", "OTJ", "WOE", "MKM", "LCI", "MOM", "MH3"]
HOLDOUT = "DSK"

CONFIGS = [
    {"name": "S (current)", "emb_dim": 256, "enc_hidden": 512,  "enc_layers": 3, "n_sab": 1, "n_heads": 4},
    {"name": "M",           "emb_dim": 384, "enc_hidden": 1024, "enc_layers": 4, "n_sab": 2, "n_heads": 6},
    {"name": "L",           "emb_dim": 512, "enc_hidden": 1536, "enc_layers": 4, "n_sab": 2, "n_heads": 8},
    {"name": "XL",          "emb_dim": 768, "enc_hidden": 2048, "enc_layers": 5, "n_sab": 3, "n_heads": 8},
]


def spec(s: str) -> dict:
    return {
        "parquet": f"{P}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
        "manifest": f"{P}/manifests/{s}.PremierDraft.sample{SIZE}.json",
        "scryfall": f"{P}/scryfall_{s.lower()}.json",
        "ratings": f"{P}/ratings/{s}.PremierDraft.ratings.json",
    }


train = [spec(s) for s in POOL]
hold = spec(HOLDOUT)
OUT = f"{P}/capacity_probe.json"


def _emit(rows):
    """Write results so far + flush stdout — makes the run peekable mid-flight."""
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    sys.stdout.flush()


# Resume: keep any configs already recorded in a prior (interrupted) run.
rows = []
if os.path.exists(OUT):
    rows = json.load(open(OUT))
    print(f"resuming: {len(rows)} config(s) already done: {[r['name'] for r in rows]}")
done = {r["name"] for r in rows}

for c in CONFIGS:
    if c["name"] in done:
        continue
    r = run_loso(
        train,
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce", aux_wr=1.0,
        holdout_ratings=hold["ratings"], emb_dim=c["emb_dim"], enc_hidden=c["enc_hidden"],
        enc_layers=c["enc_layers"], n_sab=c["n_sab"], n_heads=c["n_heads"],
        epochs=12, seed=0, val_frac=0.05,
        checkpoint_dir=f"/tmp/cap_ck/{c['name'].split()[0]}",  # per-config dir; no overwrite
    )
    h = r["holdout"]
    rows.append({"name": c["name"], **c, "in_set_top1": r["train_in_set_top1"],
                 "holdout_top1": h["top1"], "novel_top1": h.get("novel_top1")})
    print(f">>> {c['name']:>12}  emb={c['emb_dim']} hid={c['enc_hidden']} L={c['enc_layers']} "
          f"sab={c['n_sab']}  in_set={r['train_in_set_top1']:.4f}  holdout={h['top1']:.4f}",
          flush=True)
    _emit(rows)  # persist after each config so partial results survive / are peekable

print(f"\n=== capacity probe (7 sets -> {HOLDOUT}, 12 epochs, 1 seed) ===")
print("  config        emb   hid   L  sab   in-set   held-out")
for x in rows:
    print(f"  {x['name']:>12}  {x['emb_dim']:>3}  {x['enc_hidden']:>4}  {x['enc_layers']}  "
          f"{x['n_sab']}    {x['in_set_top1']:.4f}   {x['holdout_top1']:.4f}")
print(f"wrote {OUT}")
