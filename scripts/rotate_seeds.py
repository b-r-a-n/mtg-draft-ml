#!/usr/bin/env python3
"""Robustness study: rotate the held-out set × multiple seeds (best config).

Runs LOSO (set_transformer + ce + aux-WR=1.0) holding out each set in turn,
across several seeds, and reports per-holdout and overall mean ± std.

This is the WS2.1 gate harness.  Baseline: rotated top-1 0.5405±0.0229, 3 seeds × 5 holdouts.

Usage examples
--------------
# Full run (default: MiniLM, 5 LOSO sets, seeds 0-2, 10 epochs):
    PYTHONPATH=src .venv/bin/python scripts/rotate_seeds.py

# Smoke (1 holdout, 1 seed, 1 epoch):
    PYTHONPATH=src .venv/bin/python scripts/rotate_seeds.py \\
        --sets DSK --seeds 0 --epochs 1 --out /tmp/smoke.json

# Embedder sweep arm (env-var style, alternative to --embedder):
    MTG_EMBED_MODEL=BAAI/bge-large-en-v1.5 \\
        PYTHONPATH=src .venv/bin/python scripts/rotate_seeds.py --out /tmp/bge.json

# Test embedder plumbing without downloading (hash embedder):
    PYTHONPATH=src .venv/bin/python scripts/rotate_seeds.py \\
        --embedder hash --sets DSK --seeds 0 --epochs 1 --out /tmp/hash_smoke.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

from mtg_draft_ml.eval.generalization import run_loso

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = "data/hf"
_DEFAULT_SETS = ["BLB", "OTJ", "WOE", "MKM", "DSK"]
_DEFAULT_SEEDS = [0, 1, 2]
_DEFAULT_EPOCHS = 10


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Rotated-LOSO harness (WS2.1 gate): train on all-but-one LOSO sets, "
            "evaluate zero-shot on the held-out set, repeat for each set and seed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--data-dir", default=_DEFAULT_DATA_DIR,
        help="Root of the HF data layout (default: %(default)s). "
             "Expects sub-dirs: draft/, manifests/, scryfall/, ratings/.",
    )
    ap.add_argument(
        "--sets", default=None,
        help="Comma-separated list of LOSO sets (default: BLB,OTJ,WOE,MKM,DSK).",
    )
    ap.add_argument(
        "--seeds", default=None,
        help="Comma-separated integer seeds (default: 0,1,2).",
    )
    ap.add_argument(
        "--embedder", default=None,
        help="Sentence-transformer model name, 'hash' (dev/test), or None (default: "
             "MTG_EMBED_MODEL env var, else all-MiniLM-L6-v2).",
    )
    ap.add_argument(
        "--epochs", type=int, default=_DEFAULT_EPOCHS,
        help="Training epochs per run (default: %(default)s; use 1 for a smoke test).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Path to write the results JSON (default: "
             "<data-dir>/rotate_results[_<embedder-slug>].json).",
    )
    ap.add_argument(
        "--checkpoint-dir", default="/tmp/rot_ck",
        help="Directory for training checkpoints (default: %(default)s).",
    )
    ap.add_argument(
        "--standardize", action="store_true",
        help="Per-column z-score the content matrix.",
    )
    ap.add_argument(
        "--tags-dir", default=None,
        help="Directory containing <SET>.tags.json files (default: None = no tags). "
             "When set, each set's tags file is loaded and the 14 tag dims are appended "
             "to the structured feature block for BOTH train and holdout content matrices.",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_sample_size(data_dir: str, set_code: str) -> str:
    """Glob for the first sample<N> parquet in draft/ and return the size string."""
    draft_dir = pathlib.Path(data_dir) / "draft"
    matches = sorted(draft_dir.glob(f"{set_code}.PremierDraft.sample*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No draft parquet found for {set_code} in {draft_dir}. "
            "Available files: " + ", ".join(str(p.name) for p in draft_dir.glob("*.parquet"))
        )
    # Extract the sample-size string from the first match
    name = matches[0].stem  # e.g. "BLB.PremierDraft.sample60000"
    parts = name.split(".")
    sample_part = next((p for p in parts if p.startswith("sample")), None)
    if sample_part is None:
        raise ValueError(f"Cannot parse sample size from filename: {matches[0].name}")
    return sample_part[len("sample"):]  # e.g. "60000"


def spec(data_dir: str, s: str, size: str, tags_dir: str | None = None) -> dict:
    """Build a file-spec dict for set *s* given the resolved sample *size* string."""
    p = pathlib.Path(data_dir)
    d: dict = {
        "parquet":  str(p / "draft"     / f"{s}.PremierDraft.sample{size}.parquet"),
        "manifest": str(p / "manifests" / f"{s}.PremierDraft.sample{size}.json"),
        "scryfall": str(p / "scryfall"  / f"{s.lower()}.json"),
        "ratings":  str(p / "ratings"   / f"{s}.PremierDraft.ratings.json"),
    }
    if tags_dir is not None:
        tags_path = pathlib.Path(tags_dir) / f"{s}.tags.json"
        if tags_path.exists():
            d["tags"] = str(tags_path)
    return d


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), 0.0)
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    a = _parse_args(argv)

    sets  = [s.strip().upper() for s in a.sets.split(",")] if a.sets else _DEFAULT_SETS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else _DEFAULT_SEEDS

    # Resolve sample size once (all 5 LOSO sets should share it, but we check per-set)
    sizes = {s: _resolve_sample_size(a.data_dir, s) for s in sets}
    print(f"Resolved sample sizes: { {s: sizes[s] for s in sets} }")

    # Determine output path
    if a.out:
        out_path = a.out
    else:
        slug = ""
        if a.embedder:
            import re
            slug = "_" + re.sub(r"[^a-zA-Z0-9_-]", "_", a.embedder).strip("_")
        suffix = "_std" if a.standardize else ""
        out_path = str(pathlib.Path(a.data_dir) / f"rotate_results{slug}{suffix}.json")

    print(f"Sets: {sets}  Seeds: {seeds}  Epochs: {a.epochs}  Embedder: {a.embedder!r}")
    print(f"Output: {out_path}")

    rows = []
    for seed in seeds:
        for ho in sets:
            train_specs = [spec(a.data_dir, s, sizes[s], tags_dir=a.tags_dir)
                           for s in sets if s != ho]
            hold = spec(a.data_dir, ho, sizes[ho], tags_dir=a.tags_dir)
            r = run_loso(
                train_specs,
                {"parquet": hold["parquet"], "manifest": hold["manifest"],
                 "scryfall": hold["scryfall"], **({} if "tags" not in hold else {"tags": hold["tags"]})},
                embedder=a.embedder,        # None → env/MiniLM; "hash" → hashing; else explicit
                pool="set_transformer", loss="ce",
                aux_wr=1.0, holdout_ratings=hold["ratings"],
                standardize_features=a.standardize,
                epochs=a.epochs, seed=seed, val_frac=0.05,
                checkpoint_dir=a.checkpoint_dir,
            )
            h = r["holdout"]
            resolved_embedder = r.get("embedder")
            row: dict = {
                "seed": seed, "holdout": ho,
                "top1": h["top1"],
                "novel_top1": h.get("novel_top1"),
                "wr_model": h.get("wr_agreement_model"),
                "wr_human": h.get("wr_agreement_human"),
                "embedder": resolved_embedder,
                "files": {
                    "parquet":  hold["parquet"],
                    "manifest": hold["manifest"],
                    "scryfall": hold["scryfall"],
                },
            }
            if "tag_coverage_train" in r:
                row["tag_coverage_train"] = r["tag_coverage_train"]
            if "tag_coverage_holdout" in r:
                row["tag_coverage_holdout"] = r["tag_coverage_holdout"]
            rows.append(row)
            print(
                f">>> [seed {seed} | holdout {ho}] "
                f"top1={h['top1']:.4f} "
                f"novel={h.get('novel_top1', 0):.4f} "
                f"wr-agr={h.get('wr_agreement_model', 0):.4f} "
                f"embedder={resolved_embedder!r}"
            )

    print("\n=== per-holdout (mean +/- std over seeds) ===")
    for ho in sets:
        t  = ms([r["top1"]      for r in rows if r["holdout"] == ho])
        nv = ms([r["novel_top1"] for r in rows if r["holdout"] == ho])
        wr = ms([r["wr_model"]  for r in rows if r["holdout"] == ho])
        print(f"  {ho}: top1={t[0]:.4f}+/-{t[1]:.4f}  "
              f"novel={nv[0]:.4f}  wr-agr={wr[0]:.4f}")

    allt   = ms([r["top1"]      for r in rows])
    alln   = ms([r["novel_top1"] for r in rows])
    allwr  = ms([r["wr_model"]  for r in rows])
    allwrh = ms([r["wr_human"]  for r in rows])
    print(f"\n=== OVERALL (n={len(rows)} runs) ===")
    print(f"  held-out top1 : {allt[0]:.4f} +/- {allt[1]:.4f}")
    print(f"  novel-only    : {alln[0]:.4f} +/- {alln[1]:.4f}")
    print(f"  WR-agreement  : model {allwr[0]:.4f}  human {allwrh[0]:.4f}")
    print(f"  (standardize_features={a.standardize})")

    # Build summary record (richer than the old shape — adds embedder + resolved files)
    summary = {
        "embedder": rows[0]["embedder"] if rows else a.embedder,
        "sets": sets,
        "seeds": seeds,
        "epochs": a.epochs,
        "standardize": a.standardize,
        "tags_dir": a.tags_dir,
        "overall": {
            "n_runs": len(rows),
            "top1_mean": allt[0], "top1_std": allt[1],
            "novel_top1_mean": alln[0], "novel_top1_std": alln[1],
            "wr_agreement_model_mean": allwr[0],
            "wr_agreement_human_mean": allwrh[0],
        },
        "per_holdout": {
            ho: {
                "top1_mean": ms([r["top1"] for r in rows if r["holdout"] == ho])[0],
                "top1_std":  ms([r["top1"] for r in rows if r["holdout"] == ho])[1],
            }
            for ho in sets
        },
        "runs": rows,
    }
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_path, "w"), indent=2, default=float)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
