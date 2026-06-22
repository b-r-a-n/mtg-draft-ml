"""Test the IWD advantage-weighted objective: does it make 'good, not just human' picks?

Baseline (plain imitation) vs advantage-weighted (upweight imitation of high-IWD human picks).
Same 4-set -> DSK, set_transformer, MiniLM. WR-agreement measured against IWD (the signal we optimize).
Expectation: WR-agreement + avg-pick-IWD up, top-1 down modestly (deviating from human on purpose).
"""
import json
from mtg_draft_ml.eval.generalization import run_loso

D = "data/hf"; SIZE = "60000"
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLDOUT = "DSK"
IWD = "drawn_improvement_win_rate"


def spec(s):
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
            "scryfall": f"{D}/scryfall/{s.lower()}.json",
            "ratings": f"{D}/ratings/{s}.PremierDraft.ratings.json"}


train = [spec(s) for s in TRAIN]
hold = spec(HOLDOUT)
out = {}
for name, tau in [("baseline", 0.0), ("adv_tau0.03", 0.03), ("adv_tau0.05", 0.05)]:
    print(f"\n######## {name} ########", flush=True)
    r = run_loso(
        train, {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce",
        adv_tau=tau, adv_field=IWD, holdout_ratings=hold["ratings"], wr_metric_field=IWD,
        warmup_frac=0.1, grad_clip=1.0, epochs=12, seed=0, val_frac=0.05,
        checkpoint_dir=f"/workspace/adv_ck/{name}",
    )
    h = r["holdout"]
    out[name] = {"top1": h["top1"], "wr_agreement": h.get("wr_agreement_model"),
                 "avg_pick_iwd": h.get("avg_pick_wr_model"), "human_wr_agree": h.get("wr_agreement_human")}
    json.dump(out, open("/workspace/adv_objective_results.json", "w"), indent=2, default=float)

print("\n=== ADV OBJECTIVE DONE (WR-agreement vs IWD) ===")
print(f"{'config':<14} {'top1':>7} {'wr_agree':>9} {'avg_pick_iwd':>13}")
for name, v in out.items():
    print(f"{name:<14} {v['top1']:>7.4f} {v['wr_agreement']:>9.4f} {v['avg_pick_iwd']:>13.4f}")
print(f"(human WR-agreement vs IWD: {next(iter(out.values()))['human_wr_agree']:.4f})")
