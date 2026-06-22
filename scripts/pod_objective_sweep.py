"""'Good, not just human' experiment: win-rate objective (blend) x cleaner target (IWD vs GIH).

Top-1 human-pick accuracy is at its ~0.58 ceiling (data/capacity/inputs all flat). The lever with
headroom is the win-rate objective. This tests two things at once:
  (1) the pick-time quality blend (sweep alpha: trade human-likeness for picking winning cards), and
  (2) a cleaner win-rate target — IWD (drawn_improvement_win_rate, less deck/archetype-confounded)
      vs GIH (ever_drawn_win_rate, favors control decks).
Same 4-set -> DSK, set_transformer + aux-WR, MiniLM. WR-agreement is measured against the trained
target's field (so each variant is judged on its own signal); avg-pick-WR shows pick quality.
"""
import json
from mtg_draft_ml.eval.generalization import run_loso

D = "data/hf"; SIZE = "60000"
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLDOUT = "DSK"
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0]


def spec(s):
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
            "scryfall": f"{D}/scryfall/{s.lower()}.json",
            "ratings": f"{D}/ratings/{s}.PremierDraft.ratings.json"}


train = [spec(s) for s in TRAIN]
hold = spec(HOLDOUT)
out = {}
for name, field in [("GIH", "ever_drawn_win_rate"), ("IWD", "drawn_improvement_win_rate")]:
    print(f"\n######## variant {name} (aux target = {field}) ########", flush=True)
    r = run_loso(
        train, {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        embedder="all-MiniLM-L6-v2", pool="set_transformer", loss="ce",
        aux_wr=1.0, aux_wr_field=field, holdout_ratings=hold["ratings"],
        blend_alphas=ALPHAS, warmup_frac=0.1, grad_clip=1.0,
        epochs=12, seed=0, val_frac=0.05, checkpoint_dir=f"/workspace/obj_ck/{name}",
    )
    out[name] = {"holdout_top1": r["holdout"]["top1"],
                 "blend_sweep": r.get("blend_sweep")}
    json.dump(out, open("/workspace/objective_sweep_results.json", "w"), indent=2, default=float)

print("\n=== OBJECTIVE SWEEP DONE ===")
for name, v in out.items():
    print(f"-- {name}: base top1={v['holdout_top1']:.4f}")
    for s in (v["blend_sweep"] or []):
        print(f"   alpha={s['alpha']:<4} top1={s['top1']:.4f} "
              f"wr_agr={s.get('wr_agreement_model'):.4f} avg_pick_wr={s.get('avg_pick_wr_model'):.4f}")
