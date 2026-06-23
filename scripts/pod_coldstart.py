"""Cold-start distillation experiment (DD-004 #4): does an LLM teacher help a brand-new set?

Train on 4 sets, hold out a 5th as "the new set on release day". Build an LLM-teacher card-quality
ratings file for the held-out set (text+stats only — no 17lands data), then compare pick policies:
no-data baseline vs LLM cold-start blend vs real-17lands blend (upper bound). Headline: what
fraction of the (oracle - baseline) WR-agreement gap does the LLM close before human data exists?

Run the heuristic (free, offline) teacher first to smoke-test the whole pipeline, then swap in the
real Claude teacher:
    uv run python scripts/pod_coldstart.py --teacher heuristic
    uv run python scripts/pod_coldstart.py --teacher anthropic   # needs ANTHROPIC_API_KEY + [distill]
"""
import argparse

from mtg_draft_ml.distill.coldstart import run_coldstart
from mtg_draft_ml.distill.teacher import build_teacher_ratings, get_teacher

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
    ap.add_argument("--teacher", default="heuristic", help="'heuristic' or 'anthropic[:model]'")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    hold = spec(HOLDOUT)
    # 1. Build the teacher ratings for the held-out set (cached on disk; cheap to re-run).
    ratings_out = f"data/teacher_cache/{HOLDOUT}.{a.teacher.replace(':', '_')}.ratings.json"
    build_teacher_ratings(hold["manifest"], hold["scryfall"], get_teacher(a.teacher),
                          out_path=ratings_out, set_code=HOLDOUT)

    # 2. Train + compare baseline / LLM cold-start / real-data oracle.
    run_coldstart(
        [spec(s) for s in TRAIN],
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        coldstart_ratings=ratings_out, holdout_ratings=hold["ratings"],
        embedder="all-MiniLM-L6-v2", pool="set_transformer",
        epochs=a.epochs, device=a.device, seed=0,
        out_json=f"data/coldstart_{HOLDOUT}_{a.teacher.replace(':', '_')}.json",
    )


if __name__ == "__main__":
    main()
