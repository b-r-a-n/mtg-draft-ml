#!/usr/bin/env python3
"""Stage exactly the files a training pod needs into a clean canonical-layout dir for HF push.

Copies (never moves — safe to run while other jobs read the source dir) the per-set
draft parquet, manifest, Scryfall records, and 17lands ratings into:

    <out>/draft/<SET>.<EVENT>.sample<N>.parquet
    <out>/manifests/<SET>.<EVENT>.sample<N>.json
    <out>/scryfall/<set>.json
    <out>/ratings/<SET>.<EVENT>.ratings.json

This avoids pushing experiment-output cruft (loso_*.json etc.) and gives the pod a known layout.
Then: huggingface-cli login && python -m mtg_draft_ml.data.hf push --repo <you>/mtg-draft \\
        --processed-dir <out>

Usage: python scripts/stage_for_push.py [src_dir] [out_dir] [size] [SET ...]
"""
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/p1proc")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "hf_push"
size = sys.argv[3] if len(sys.argv) > 3 else "60000"
sets = sys.argv[4:] or ["BLB", "OTJ", "WOE", "MKM", "LCI", "MOM", "MH3", "DSK"]
event = "PremierDraft"

for sub in ("draft", "manifests", "scryfall", "ratings"):
    (out / sub).mkdir(parents=True, exist_ok=True)

missing, staged = [], 0
for s in sets:
    tag = f"{s}.{event}.sample{size}"
    plan = [
        (src / "draft" / f"{tag}.parquet",            out / "draft" / f"{tag}.parquet"),
        (src / "manifests" / f"{tag}.json",           out / "manifests" / f"{tag}.json"),
        (src / f"scryfall_{s.lower()}.json",          out / "scryfall" / f"{s.lower()}.json"),
        (src / "ratings" / f"{s}.{event}.ratings.json", out / "ratings" / f"{s}.{event}.ratings.json"),
    ]
    for srcf, dstf in plan:
        if srcf.exists():
            shutil.copy2(srcf, dstf)
            staged += 1
        else:
            missing.append(str(srcf))

print(f"staged {staged} files for {len(sets)} sets -> {out}")
for sub in ("draft", "manifests", "scryfall", "ratings"):
    n = len(list((out / sub).glob("*")))
    print(f"  {sub}/: {n} files")
if missing:
    print(f"\nWARNING: {len(missing)} expected files missing:")
    for m in missing:
        print("  -", m)
