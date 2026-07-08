"""Cold-start distillation experiment (DD-004 #4): does an LLM teacher help a brand-new set?

Train on 4 sets, hold out a 5th as "the new set on release day". Build an LLM-teacher card-quality
ratings file for the held-out set (text+stats only — no 17lands data), then compare pick policies:
no-data baseline vs LLM cold-start blend vs real-17lands blend (upper bound). Headline: what
fraction of the (oracle - baseline) WR-agreement gap does the LLM close before human data exists?

Run the heuristic (free, offline) teacher first to smoke-test the whole pipeline, then swap in the
real Claude teacher:
    uv run python scripts/pod_coldstart.py --teacher heuristic
    uv run python scripts/pod_coldstart.py --teacher anthropic   # needs ANTHROPIC_API_KEY + [distill]

Pass multiple teachers as a comma-separated list to evaluate them on the SAME trained model:
    uv run python scripts/pod_coldstart.py --teacher heuristic,cached:agent-claude --seed 0
"""
import argparse
import pathlib

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
    ap.add_argument("--teacher", default="heuristic",
                    help="Teacher spec or comma-separated list, e.g. 'heuristic,cached:agent-claude'")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: data/coldstart_<teacher-ids>_seed<seed>.json)")
    a = ap.parse_args()

    teacher_names = [t.strip() for t in a.teacher.split(",")]
    hold = spec(HOLDOUT)

    # 1. Build a teacher ratings file for the held-out set per teacher (cached; cheap to re-run).
    ratings_paths: list[str] = []
    for tname in teacher_names:
        teacher = get_teacher(tname)
        ratings_out = f"data/teacher_cache/{HOLDOUT}.{tname.replace(':', '_')}.ratings.json"
        pathlib.Path(ratings_out).parent.mkdir(parents=True, exist_ok=True)
        build_teacher_ratings(hold["manifest"], hold["scryfall"], teacher,
                              out_path=ratings_out, set_code=HOLDOUT)
        ratings_paths.append(ratings_out)

    # Resolve output path.
    if a.out is None:
        ids = "_".join(t.replace(":", "_") for t in teacher_names)
        out_json = f"data/coldstart_{ids}_seed{a.seed}.json"
    else:
        out_json = a.out

    # 2. Train + compare baseline / teacher arm(s) / real-data oracle.
    #    Pass a single str for single-teacher (backward compat) or a list for multi-teacher.
    coldstart_arg = ratings_paths[0] if len(ratings_paths) == 1 else ratings_paths

    run_coldstart(
        [spec(s) for s in TRAIN],
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        coldstart_ratings=coldstart_arg, holdout_ratings=hold["ratings"],
        embedder="all-MiniLM-L6-v2", pool="set_transformer",
        epochs=a.epochs, device=a.device, seed=a.seed,
        out_json=out_json,
    )


if __name__ == "__main__":
    main()
