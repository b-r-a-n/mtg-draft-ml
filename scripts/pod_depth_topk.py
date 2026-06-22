"""Depth comparison ON TOP-K: landed recipe at 60k vs 500k picks/set, 4 sets -> DSK.

Reuses the 500k parquet already preprocessed by pod_moredata.py (data/big) and the 60k parquet from
HF (data/hf). Reports top-1/3/5 + WR-agreement for both depths — the fair metric given top-1 is
human-noise-capped. Run after pod_moredata.py (so data/big exists) and after pulling HF draft 60k.
"""
import json

from mtg_draft_ml.eval.generalization import run_loso

TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLD = "DSK"


def specs(root, tag):
    def s(x):
        return {"parquet": f"{root}/draft/{x}.PremierDraft.{tag}.parquet",
                "manifest": f"{root}/manifests/{x}.PremierDraft.{tag}.json",
                "scryfall": f"data/hf/scryfall/{x.lower()}.json",
                "ratings": f"data/hf/ratings/{x}.PremierDraft.ratings.json"}
    return [s(x) for x in TRAIN], s(HOLD)


out = {}
for label, root, tag in [("60k", "data/hf", "sample60000"), ("500k", "data/big", "r500000")]:
    tr, ho = specs(root, tag)
    print(f"\n######## depth={label} ########", flush=True)
    r = run_loso(tr, {k: ho[k] for k in ("parquet", "manifest", "scryfall")},
                 embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce",
                 adv_tau=0.03, adv_field="drawn_improvement_win_rate",
                 holdout_ratings=ho["ratings"], wr_metric_field="drawn_improvement_win_rate",
                 warmup_frac=0.1, grad_clip=1.0, epochs=10, seed=0, val_frac=0.05,
                 checkpoint_dir=f"/workspace/depth_ck/{label}")
    h = r["holdout"]
    out[label] = {"top1": h["top1"], "top3": h.get("top3"), "top5": h.get("top5"),
                  "wr_agreement": h.get("wr_agreement_model"), "n": h["n"]}
    json.dump(out, open("/workspace/depth_topk_results.json", "w"), indent=2, default=float)

print("\n=== DEPTH x TOP-K (4 sets -> DSK, landed recipe) ===")
print(f"{'depth':<6} {'top1':>7} {'top3':>7} {'top5':>7} {'WR-agree':>9}")
for label, v in out.items():
    print(f"{label:<6} {v['top1']:>7.4f} {v['top3']:>7.4f} {v['top5']:>7.4f} {v['wr_agreement']:>9.4f}")
