"""Good-players-vs-all experiment: train on high-skill drafters instead of the average one.

17lands tags every pick with the drafter's skill (already in our parquets). This trains 2x2 —
{all players, good players} x {CE, CE + composite-WR KD} — holding out DSK, and reports whether
filtering to good players raises WR-agreement (cleaner, better picks) and top-1 *on good-player
holdout* (less label noise). The depth null means we can afford to drop the volume.

    uv run python scripts/pod_skill.py                          # winrate>=0.55 & games>=50
    uv run python scripts/pod_skill.py --min-winrate 0.58 --ranks mythic,diamond,platinum
"""
import argparse

from mtg_draft_ml.distill.skill import run_skill_experiment

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
    ap.add_argument("--train-sets", default=",".join(TRAIN))
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--min-games", type=float, default=50)
    ap.add_argument("--ranks", default=None, help="comma list e.g. mythic,diamond,platinum")
    ap.add_argument("--volume-control", action="store_true",
                    help="add a random-subsample arm at the good-player fraction (quality vs quantity)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    train_sets = [s.strip() for s in a.train_sets.split(",")]
    hold = spec(a.holdout)
    print(f"train={train_sets} holdout={a.holdout}")
    run_skill_experiment(
        [spec(s) for s in train_sets],
        {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]},
        holdout_ratings=hold["ratings"], min_winrate=a.min_winrate, min_games=a.min_games,
        ranks=set(s.strip() for s in a.ranks.split(",")) if a.ranks else None,
        volume_control=a.volume_control,
        embedder="all-MiniLM-L6-v2", pool="set_transformer", epochs=a.epochs, device=a.device, seed=0,
        out_json=f"data/skill_{a.holdout}_wr{a.min_winrate}{'_vc' if a.volume_control else ''}.json",
    )


if __name__ == "__main__":
    main()
