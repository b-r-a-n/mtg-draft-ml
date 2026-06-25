"""Ingest new 17lands sets into the canonical HF dataset layout (data/hf), end to end.

For each set: download a sample of draft data → compact parquet + manifest; filter the global Scryfall
oracle bulk to that set's cards → per-set scryfall json; download 17lands ratings; build the
game_data `deck_value` field. Result is the exact `data/hf/{draft,manifests,scryfall,ratings,
gamevalue}` layout the training pipeline + webapp expect. `--push` uploads to the HF dataset.

This is the prerequisite for expanding the training corpus (e.g. the WR-agreement scaling curve):

    uv run python scripts/ingest_set.py --sets FDN,DFT,TDM,FIN,EOE,ONE,BRO,DMU --sample-rows 60000 --push
"""
from __future__ import annotations

import argparse
import json
import pathlib

from mtg_draft_ml.data.download import (
    download_17lands_draft,
    download_card_ratings,
    download_scryfall_oracle,
)
from mtg_draft_ml.data.preprocess import preprocess_set

HF = pathlib.Path("data/hf")
EVENT = "PremierDraft"


def _per_set_scryfall(manifest_path, oracle_path, out_path):
    """Filter the global oracle bulk to the cards in a set's manifest → per-set scryfall records."""
    names = {c["name"] for c in json.load(open(manifest_path))["cards"]}
    fronts = {n.split(" // ", 1)[0] for n in names}
    keep = [c for c in json.load(open(oracle_path))
            if c.get("name") in names or (c.get("name") or "").split(" // ", 1)[0] in fronts]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(keep))
    return len(keep)


def ingest(set_code, sample_rows, oracle_path, l2=30.0, force=False):
    from mtg_draft_ml.eval.game_value import build_value_ratings, fit_card_values
    from mtg_draft_ml.data.game_preprocess import preprocess_game_set

    tag = f"{set_code}.{EVENT}.sample{sample_rows}"
    parquet = HF / "draft" / f"{tag}.parquet"
    manifest = HF / "manifests" / f"{tag}.json"
    scry = HF / "scryfall" / f"{set_code.lower()}.json"
    gv = HF / "gamevalue" / f"{set_code}.{EVENT}.gamevalue.json"

    # 1. draft → parquet + manifest (oracle_ids resolved via the global bulk)
    csv = download_17lands_draft(set_code, EVENT, "data/raw", sample_rows=sample_rows, force=force)
    m = preprocess_set(csv, parquet, manifest, scryfall_path=oracle_path,
                       set_code=set_code, event_type=EVENT)
    # 2. per-set scryfall
    n_scry = _per_set_scryfall(manifest, oracle_path, scry)
    # 3. ratings
    download_card_ratings(set_code, EVENT, str(HF / "ratings"), force=force)
    # 4. deck_value from game_data
    from mtg_draft_ml.data.download import download_17lands_game
    gcsv = download_17lands_game(set_code, EVENT, "data/raw", sample_rows=sample_rows, force=force)
    npz = pathlib.Path("data/game") / f"game.{set_code}.{EVENT}.sample{sample_rows}.npz"
    gdata = preprocess_game_set(gcsv, manifest, out_npz=npz)
    fit = fit_card_values(gdata["X"], gdata["y"], gdata["C"], l2=l2)
    support = gdata["X"].sum(axis=0)
    recs = build_value_ratings(fit["beta"], support, list(gdata["card_names"]), set_code=set_code)
    gv.parent.mkdir(parents=True, exist_ok=True)
    gv.write_text(json.dumps(recs))
    n_val = sum(r["deck_value"] is not None for r in recs)
    print(f"  {set_code}: {m['n_rows']} picks, {m['n_cards']} cards, {n_scry} scryfall recs, "
          f"ratings ok, deck_value for {n_val}/{m['n_cards']} cards (train_acc {fit['train_acc']:.3f})")
    return {"set": set_code, "n_cards": m["n_cards"], "n_rows": m["n_rows"], "n_valued": n_val}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", required=True, help="comma list, e.g. FDN,DFT,TDM,FIN,EOE")
    ap.add_argument("--sample-rows", type=int, default=60_000)
    ap.add_argument("--l2", type=float, default=30.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--push", action="store_true", help="upload to the HF dataset after ingest")
    ap.add_argument("--repo", default="b-r-a-n/mtg-draft")
    a = ap.parse_args(argv)

    sets = [s.strip() for s in a.sets.split(",")]
    oracle = download_scryfall_oracle(force=a.force)  # global bulk (one-time ~150MB)
    print(f"oracle bulk: {oracle}")
    done = []
    for s in sets:
        try:
            done.append(ingest(s, a.sample_rows, str(oracle), l2=a.l2, force=a.force))
        except Exception as e:
            print(f"  {s}: FAILED ({type(e).__name__}: {e})")

    print(f"\ningested {len(done)}/{len(sets)} sets")
    if a.push and done:
        from mtg_draft_ml.data.hf import push_dataset
        url = push_dataset(str(HF), a.repo, commit_message=f"add sets: {','.join(d['set'] for d in done)}")
        print(f"pushed → {url}")


if __name__ == "__main__":
    main()
