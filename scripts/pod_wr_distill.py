"""WR-softmax distillation (DD-004 #1): good-not-just-human as a DENSE target vs a scalar reweight.

Compare on a held-out set: baseline CE vs CE + WR-softmax KD (reshapes the target into the win-rate
ranking over the pack) vs CE + IWD advantage-weighting (the existing scalar example-reweighting). The
question: does reshaping the target beat reweighting the example for taking winning picks on an unseen
set? Read WR-agreement + avg-pick-WR (not top-1).

    uv run python scripts/pod_wr_distill.py
    uv run python scripts/pod_wr_distill.py --wr-field drawn_improvement_win_rate --distill-topk 5
"""
import argparse

from mtg_draft_ml.distill.wr import run_wr_distill

D = "data/hf"
SIZE = "60000"
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]   # default 4-set; --train-sets to use more (all 8 downloaded)
HOLDOUT = "DSK"


def spec(s):
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
            "scryfall": f"{D}/scryfall/{s.lower()}.json",
            "ratings": f"{D}/ratings/{s}.PremierDraft.ratings.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-field", default="ever_drawn_win_rate")
    ap.add_argument("--wr-tau", type=float, default=1.0)
    ap.add_argument("--quality-fields",
                    default="ever_drawn_win_rate,drawn_improvement_win_rate,avg_pick",
                    help="comma list of 17lands fields for the composite-target arm ('' to skip)")
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=1.0)
    ap.add_argument("--distill-topk", type=int, default=0)
    ap.add_argument("--adv-tau", type=float, default=0.03)
    ap.add_argument("--train-sets", default=",".join(TRAIN),
                    help="comma list of train sets (more sets = the proven multi-set axis)")
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    train_sets = [s.strip() for s in a.train_sets.split(",")]
    hold = spec(a.holdout)
    print(f"train={train_sets} holdout={a.holdout}")
    run_wr_distill(
        [spec(s) for s in train_sets],
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        holdout_ratings=hold["ratings"], wr_field=a.wr_field, wr_tau=a.wr_tau,
        quality_fields=[s.strip() for s in a.quality_fields.split(",")] if a.quality_fields else None,
        distill_lambda=a.distill_lambda, distill_temp=a.distill_temp, distill_topk=a.distill_topk,
        adv_tau=a.adv_tau, embedder="all-MiniLM-L6-v2", pool="set_transformer",
        epochs=a.epochs, device=a.device, seed=0,
        out_json=f"data/wr_distill_{a.holdout}_{len(train_sets)}sets_{a.wr_field}.json",
    )


if __name__ == "__main__":
    main()
