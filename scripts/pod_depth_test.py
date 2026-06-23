"""Depth test under the composite WR objective: does MORE picks/set help now?

The original depth-scaling null (60k->500k flat) was measured under hard-label CE imitation. This
re-checks it under the new dense composite-WR target (the lever that actually moved WR-agreement):
train the SAME 4 sets at increasing picks/set, hold out a FIXED DSK (so only train-data volume
changes), and compare WR-agreement. If depth still saturates here too, that's a second confirmation
the bottleneck is the signal, not the pick count; if it climbs, more picks help the WR policy.

Larger samples are built on the fly (download a bigger first-N-rows sample from 17lands, preprocess
into data/hf/) reusing the already-pulled per-set Scryfall + ratings. Run on a GPU pod after the
data is present:  uv run python scripts/pod_depth_test.py --sizes 60000,240000 --epochs 8
"""
import argparse
import pathlib

from mtg_draft_ml.data.download import download_17lands_draft
from mtg_draft_ml.data.preprocess import preprocess_set
from mtg_draft_ml.distill.wr import run_wr_distill

D = "data/hf"
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLDOUT = "DSK"
HOLD_SIZE = "60000"                       # fixed holdout — vary only TRAIN volume
QUALITY = ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"]


def spec(s, size):
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{size}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{size}.json",
            "scryfall": f"{D}/scryfall/{s.lower()}.json",
            "ratings": f"{D}/ratings/{s}.PremierDraft.ratings.json"}


def ensure_sample(s, size):
    """Build data/hf/draft/<s>...sample<size>.{parquet,json} if missing (download + preprocess)."""
    p = spec(s, size)
    if pathlib.Path(p["parquet"]).exists() and pathlib.Path(p["manifest"]).exists():
        return
    print(f">>> building {s} sample{size} (download + preprocess)…", flush=True)
    csv = download_17lands_draft(s, sample_rows=int(size), dest_dir="data/raw")
    preprocess_set(csv, p["parquet"], p["manifest"], scryfall_path=p["scryfall"], set_code=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="60000,240000", help="train picks/set to compare")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    hold = spec(HOLDOUT, HOLD_SIZE)
    hold_eval = {"parquet": hold["parquet"], "manifest": hold["manifest"], "scryfall": hold["scryfall"]}
    summary = {}
    for size in [s.strip() for s in a.sizes.split(",")]:
        print(f"\n######## DEPTH: train picks/set = {size} ########", flush=True)
        for s in TRAIN:
            ensure_sample(s, size)
        res = run_wr_distill(
            [spec(s, size) for s in TRAIN], hold_eval, holdout_ratings=hold["ratings"],
            quality_fields=QUALITY, embedder="all-MiniLM-L6-v2", pool="set_transformer",
            epochs=a.epochs, device=a.device, seed=0,
            out_json=f"data/wr_depth_{HOLDOUT}_train{size}.json",
        )
        comp = res.get("wr_kd_comp") or res["wr_kd"]
        summary[size] = {"baseline_wr": res["baseline"]["wr_agreement_model"],
                         "comp_wr": comp["wr_agreement_model"], "comp_top1": comp["top1"]}

    print("\n=== DEPTH SUMMARY (does more picks/set help under the composite WR target?) ===")
    print(f"  {'picks/set':>10} {'baseline WR':>12} {'composite WR':>13} {'comp top1':>10}")
    for size, v in summary.items():
        print(f"  {size:>10} {v['baseline_wr']:>12.4f} {v['comp_wr']:>13.4f} {v['comp_top1']:>10.4f}")
    sizes = list(summary)
    if len(sizes) >= 2:
        d = summary[sizes[-1]]["comp_wr"] - summary[sizes[0]]["comp_wr"]
        print(f"  composite WR  {sizes[0]} -> {sizes[-1]}:  {d:+.4f}  "
              f"({'depth helps the WR policy' if d > 0.003 else 'flat — depth still saturates'})")


if __name__ == "__main__":
    main()
