"""Ensemble-of-seeds soft-label distillation (DD-004 #1): more value from the data we already have.

Train K seed teachers on 4 sets, distill their averaged pack ranking into one student, and compare
on a held-out set: baseline CE vs CE+KD vs the ensemble itself. The point isn't top-1 (likely
noise-ceilinged at ~0.58) — it's WR-agreement and ranking quality (top-3/5, MTPD), and the
sample-efficiency test via --train-frac (does KD on a fraction match plain CE on all the data?).

    uv run python scripts/pod_distill.py                       # full-data KD vs CE
    uv run python scripts/pod_distill.py --train-frac 0.25      # sample-efficiency: KD on 1/4 the data
"""
import argparse

from mtg_draft_ml.distill.ensemble import run_ensemble_distill

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
    ap.add_argument("--teachers", type=int, default=3)
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-topk", type=int, default=0, help="0=full pack; else top-k ranking distill")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    hold = spec(HOLDOUT)
    run_ensemble_distill(
        [spec(s) for s in TRAIN],
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        n_teachers=a.teachers, distill_lambda=a.distill_lambda, distill_temp=a.distill_temp,
        distill_topk=a.distill_topk, train_frac=a.train_frac, holdout_ratings=hold["ratings"],
        embedder="all-MiniLM-L6-v2", pool="set_transformer",
        epochs=a.epochs, device=a.device, seed=0,
        out_json=f"data/distill_{HOLDOUT}_l{a.distill_lambda}_f{a.train_frac}.json",
    )


if __name__ == "__main__":
    main()
