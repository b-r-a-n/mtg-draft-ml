"""Bake the webapp Deck-Doctor assets (local, no pod):
  1. add `mc` (Scryfall mana cost, MDFC-safe) to each card in webapp/data/<SET>.cards.json so the
     in-browser castability model can read colored pips (cards.json had only `ci`/`cmc`).
  2. write webapp/data/<SET>.sampledecks.json — a handful of REAL 17lands drafted pools (deck +
     sideboard card names) from the local game_data, so you can point the doctor at a real deck.

    uv run python scripts/export_doctor_assets.py                 # all webapp sets
    uv run python scripts/export_doctor_assets.py --sets DSK,OTJ --n-decks 12
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

from mtg_draft_ml.eval.castability import card_mana_cost, parse_pips

COLORS = ("W", "U", "B", "R", "G")
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
WEB = pathlib.Path("webapp")


def _scry_by_name(set_code, data_dir):
    out = {}
    p = pathlib.Path(f"{data_dir}/scryfall/{set_code.lower()}.json")
    if p.exists():
        for r in json.load(open(p)):
            nm = r.get("name")
            if nm:
                out.setdefault(nm, r)
                if " // " in nm:
                    out.setdefault(nm.split(" // ", 1)[0], r)
    return out


def add_mana_costs(set_code, data_dir):
    """Add `mc` to each card in cards.json (in place). Returns n with a real cost."""
    path = WEB / "data" / f"{set_code}.cards.json"
    blob = json.load(open(path))
    scry = _scry_by_name(set_code, data_dir)
    have = 0
    for c in blob["cards"]:
        rec = scry.get(c["name"], {})
        mc = card_mana_cost({"t": c.get("t"), "mana_cost": rec.get("mana_cost"),
                             "card_faces": rec.get("card_faces")})
        c["mc"] = mc
        have += bool(parse_pips(mc) and any(parse_pips(mc).get(k, 0) for k in COLORS))
    path.write_text(json.dumps(blob))
    return len(blob["cards"]), have


def sample_real_decks(set_code, raw_dir, n_decks, max_scan=4000):
    """Pull `n_decks` real drafted pools (deck + sideboard names, with multiples) from game_data."""
    import pandas as pd

    csvs = sorted(glob.glob(f"{raw_dir}/game.{set_code}.*.csv"),
                  key=lambda p: pathlib.Path(p).stat().st_size)
    if not csvs:
        return []
    csv = csvs[-1]
    head = pd.read_csv(csv, nrows=0).columns.tolist()
    deck_cols = [c for c in head if c.startswith("deck_")]
    side_cols = [("sideboard_" + c[len("deck_"):]) for c in deck_cols]
    side_cols = [c for c in side_cols if c in head]
    names = [c[len("deck_"):] for c in deck_cols]
    use = ["draft_id", "won"] + deck_cols + side_cols
    use = [c for c in use if c in head]

    decks, seen = [], set()
    for chunk in pd.read_csv(csv, usecols=use, chunksize=20_000):
        chunk = chunk[~chunk["draft_id"].isin(seen)].drop_duplicates("draft_id")
        seen.update(chunk["draft_id"].tolist())
        for _, row in chunk.iterrows():
            pool = []
            for j, nm in enumerate(names):
                if nm in BASICS:
                    continue                                   # skip basic lands in the pool
                cnt = int(row[deck_cols[j]] or 0) + (int(row[side_cols[j]] or 0) if side_cols else 0)
                pool += [nm] * cnt
            if len(pool) < 20:                                 # not a real pool
                continue
            decks.append({"label": f"real deck #{len(decks) + 1}", "pool": pool})
            if len(decks) >= n_decks:
                return decks
        if len(seen) >= max_scan:
            break
    return decks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None, help="comma list; default = all webapp sets")
    ap.add_argument("--n-decks", type=int, default=12)
    ap.add_argument("--data-dir", default="data/hf")
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args(argv)

    sets = ([s.strip() for s in a.sets.split(",")] if a.sets
            else sorted(p.name.split(".")[0] for p in (WEB / "data").glob("*.meta.json")))
    for s in sets:
        n, have = add_mana_costs(s, a.data_dir)
        decks = sample_real_decks(s, a.raw_dir, a.n_decks)
        (WEB / "data" / f"{s}.sampledecks.json").write_text(json.dumps(decks))
        print(f"  {s}: cards.json +mc ({have}/{n} colored) · {len(decks)} real decks "
              f"(avg pool {sum(len(d['pool']) for d in decks) // max(len(decks), 1)})")


if __name__ == "__main__":
    main()
