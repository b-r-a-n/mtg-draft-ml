"""Leaky-feature -> release-day distillation (DD-004 #3): smuggle win-rate knowledge into a student.

Train a teacher whose card features include per-card win rate (a post-hoc aggregate absent on a new
set's release day), then distill its rankings into a student that sees only release-day inputs. The
student needs no win-rate features at inference. Compare on a held-out set: baseline CE (floor) vs
CE+KD (candidate) vs the leaky teacher (ceiling). Read WR-agreement / avg-pick-WR.

    uv run python scripts/pod_leaky_distill.py
    uv run python scripts/pod_leaky_distill.py --wr-field drawn_improvement_win_rate --distill-topk 5
"""
import argparse

from mtg_draft_ml.distill.leaky import run_leaky_distill

D = "data/hf"
SIZE = "60000"
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLDOUT = "DSK"


def spec(s):
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
            "scryfall": f"{D}/scryfall/{s.lower()}.json",
            "ratings": f"{D}/ratings/{s}.PremierDraft.ratings.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-field", default="ever_drawn_win_rate")
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-topk", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    hold = spec(HOLDOUT)
    run_leaky_distill(
        [spec(s) for s in TRAIN],
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        holdout_ratings=hold["ratings"], wr_field=a.wr_field,
        distill_lambda=a.distill_lambda, distill_temp=a.distill_temp, distill_topk=a.distill_topk,
        embedder="all-MiniLM-L6-v2", pool="set_transformer", epochs=a.epochs, device=a.device, seed=0,
        out_json=f"data/leaky_distill_{HOLDOUT}_{a.wr_field}.json",
    )


if __name__ == "__main__":
    main()
