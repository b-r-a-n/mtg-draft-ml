"""Tuned capacity sweep: per-size LR + LR warmup + grad clip + more epochs.

Resolves the fixed-LR collapse from the naive sweep (M's in-set fell 0.62->0.41 = optimization
failure, not a capacity verdict). Here each rung gets LR warmup, gradient clipping, scaled-down LR
for bigger models, and 20 epochs. Question: with proper optimization, does a bigger model fit train
as well as S AND beat S's held-out ~0.576? Canonical data layout (data/hf), 7 train sets -> DSK.
"""
import json
from mtg_draft_ml.eval.generalization import run_loso

D = "data/hf"; SIZE = "60000"
TRAIN_SETS = ["BLB", "OTJ", "WOE", "MKM", "LCI", "MOM", "MH3"]
HOLDOUT = "DSK"


def spec(s):
    lc = s.lower()
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
            "scryfall": f"{D}/scryfall/{lc}.json",
            "ratings": f"{D}/ratings/{s}.PremierDraft.ratings.json"}


train = [spec(s) for s in TRAIN_SETS]
hold = spec(HOLDOUT)
# name, emb, hid, layers, sab, heads, lr  (bigger -> lower LR)
CONFIGS = [
    ("S",  256, 512,  3, 1, 4, 1e-3),
    ("M",  384, 1024, 4, 2, 6, 5e-4),
    ("L",  512, 1536, 4, 2, 8, 3e-4),
]
rows = []
for name, emb, hid, lyr, sab, heads, lr in CONFIGS:
    print(f">>> config {name} (emb={emb} hid={hid} L={lyr} sab={sab} heads={heads} lr={lr})", flush=True)
    r = run_loso(
        train, {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce", aux_wr=1.0,
        holdout_ratings=hold["ratings"], emb_dim=emb, enc_hidden=hid, enc_layers=lyr,
        n_sab=sab, n_heads=heads, lr=lr, warmup_frac=0.1, grad_clip=1.0,
        epochs=20, seed=0, val_frac=0.05, checkpoint_dir=f"/workspace/tuned_ck/{name}")
    h = r["holdout"]
    rows.append({"name": name, "lr": lr, "in_set": r["train_in_set_top1"],
                 "holdout": h["top1"], "novel": h.get("novel_top1")})
    json.dump(rows, open("/workspace/tuned_sweep_results.json", "w"), indent=2)
    print(f">>> {name}: in_set={r['train_in_set_top1']:.4f} holdout={h['top1']:.4f} "
          f"novel={h.get('novel_top1', 0):.4f}", flush=True)

print("=== TUNED SWEEP DONE ===")
for x in rows:
    print(f"  {x['name']:>3} lr={x['lr']:<6} in_set={x['in_set']:.4f} holdout={x['holdout']:.4f} novel={x['novel']:.4f}")
