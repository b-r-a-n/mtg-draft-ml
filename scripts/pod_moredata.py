"""Depth-scaling test: train the LANDED recipe on ~8x more picks/set and compare held-out.

Every prior run used 60k picks/set; this preprocesses the same sets at 500k picks/set (depth axis,
untested — set-count already saturated in Phase 4). Final recipe: content encoder + Set Transformer
+ in-pack CE + IWD advantage-weighting, MiniLM. 4 sets -> DSK. Reuses Scryfall/ratings from HF;
re-downloads draft data fresh from 17lands at the larger size.
"""
import json
import sys

from mtg_draft_ml.data.download import download_17lands_draft
from mtg_draft_ml.data.preprocess import preprocess_set
from mtg_draft_ml.eval.generalization import run_loso

HF = "data/hf"          # scryfall/ + ratings/ pulled from HF
OUT = "data/big"        # fresh large parquet here
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLD = "DSK"


def prep(s):
    scry = f"{HF}/scryfall/{s.lower()}.json"
    csv = download_17lands_draft(s, "PremierDraft", dest_dir=f"{OUT}/raw", sample_rows=ROWS)
    pqp = f"{OUT}/draft/{s}.PremierDraft.r{ROWS}.parquet"
    man = f"{OUT}/manifests/{s}.PremierDraft.r{ROWS}.json"
    m = preprocess_set(csv, pqp, man, scryfall_path=scry, set_code=s, event_type="PremierDraft")
    print(f"  {s}: {m['n_rows']} picks, {m['n_cards']} cards", flush=True)
    return {"parquet": pqp, "manifest": man, "scryfall": scry,
            "ratings": f"{HF}/ratings/{s}.PremierDraft.ratings.json"}


print(f"=== preprocessing {len(TRAIN)+1} sets at {ROWS} picks each ===", flush=True)
train = [prep(s) for s in TRAIN]
hold = prep(HOLD)

print("=== training landed recipe (set_transformer + ce + IWD adv-weighting) ===", flush=True)
r = run_loso(
    train, {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
    embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce",
    adv_tau=0.03, adv_field="drawn_improvement_win_rate",
    holdout_ratings=hold["ratings"], wr_metric_field="drawn_improvement_win_rate",
    warmup_frac=0.1, grad_clip=1.0, epochs=10, seed=0, val_frac=0.05,
    checkpoint_dir="/workspace/big_ck",
)
h = r["holdout"]
res = {"rows_per_set": ROWS, "train_picks_total": sum(json.load(open(s["manifest"]))["n_rows"] for s in train),
       "holdout_top1": h["top1"], "wr_agreement": h.get("wr_agreement_model"),
       "avg_pick_iwd": h.get("avg_pick_wr_model"), "human_wr_agreement": h.get("wr_agreement_human")}
json.dump(res, open("/workspace/moredata_result.json", "w"), indent=2, default=float)
print("\n=== MORE-DATA RESULT ===")
print(f"  rows/set={ROWS}  total train picks={res['train_picks_total']:,}")
print(f"  held-out top1={res['holdout_top1']:.4f}  WR-agree(IWD)={res['wr_agreement']:.4f}  "
      f"avg_pick_IWD={res['avg_pick_iwd']:.4f}")
print("  (60k/set baseline: top1~0.55-0.57, WR-agree~0.295)")
