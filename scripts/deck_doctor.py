"""Deck doctor CLI — rate a drafted pool's recommended deck: power + buildability + castability + advice.

Standalone & local: webapp/data/<SET>.cards.json (deck_value/cmc/ci/t) + Scryfall (mana_cost) + the
exported webapp/model/<SET>.playprob.json buildability trees. No game_data / no pod.

    uv run python scripts/deck_doctor.py --set DSK --demo            # a value-pile pool (gets cut to 2c)
    uv run python scripts/deck_doctor.py --set DSK --pool "Murder, Valgavoth Terror Eater, ..."
    uv run python scripts/deck_doctor.py --set DSK --pool-file my_pool.txt
"""
from __future__ import annotations

import argparse
import json
import pathlib

from mtg_draft_ml.eval.deck_doctor import diagnose
from mtg_draft_ml.eval.play_prob import load_play_tree_model


def load_cards(set_code: str, webapp: str = "webapp", data_dir: str = "data/hf"):
    cj = json.load(open(f"{webapp}/data/{set_code}.cards.json"))["cards"]
    n = len(cj)
    by_name = {}
    scry = pathlib.Path(f"{data_dir}/scryfall/{set_code.lower()}.json")
    if scry.exists():
        for r in json.load(open(scry)):
            nm = r.get("name")
            if nm:
                by_name.setdefault(nm, r)
                if " // " in nm:
                    by_name.setdefault(nm.split(" // ", 1)[0], r)
    cards = [None] * n
    for c in cj:
        rec = by_name.get(c["name"], {})
        cards[c["i"]] = {**c, "mana_cost": rec.get("mana_cost"), "card_faces": rec.get("card_faces")}
    return cards, n


def resolve_pool(names, cards):
    lower = {c["name"].lower(): c["i"] for c in cards}
    pool, missing = [], []
    for nm in names:
        nm = nm.strip()
        if not nm:
            continue
        i = lower.get(nm.lower())
        if i is None:
            cand = [c["i"] for c in cards if c["name"].lower().startswith(nm.lower())]
            i = cand[0] if cand else None
        (pool.append(i) if i is not None else missing.append(nm))
    return pool, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--pool", default=None, help="comma-separated card names")
    ap.add_argument("--pool-file", default=None, help="one card name per line")
    ap.add_argument("--demo", action="store_true", help="use a top-45-by-deck_value value pile")
    a = ap.parse_args(argv)

    cards, n = load_cards(a.set_code)
    pm = load_play_tree_model(a.set_code, cards, n)

    if a.demo:
        pool = [c["i"] for c in sorted(
            (c for c in cards if c.get("t") != "land" and c.get("deck_value") is not None),
            key=lambda c: -c["deck_value"])[:45]]
        print(f"[demo] {a.set_code}: a 45-card 'value pile' = the top cards by deck_value (all colors)\n")
    else:
        if a.pool_file:
            names = pathlib.Path(a.pool_file).read_text().splitlines()
        elif a.pool:
            names = a.pool.split(",")
        else:
            raise SystemExit("give --pool, --pool-file, or --demo")
        pool, missing = resolve_pool(names, cards)
        if missing:
            print(f"(unmatched cards skipped: {missing})")

    r = diagnose(pool, cards, pm)
    bar = lambda x: "█" * int(round(x * 20)) + "·" * (20 - int(round(x * 20)))
    print(f"=== Deck doctor · {a.set_code} · grade {r['grade']} ({r['overall']:.2f}) ===")
    print(f"  deck: {r['n_spells']} spells + {r['lands']} lands · colors {'/'.join(r['colors']) or '—'} "
          f"· avg cmc {r['avg_cmc']}")
    print(f"  power      {bar(r['power_pct'])} {r['power_pct']*100:>3.0f}th pct  (mean deck_value {r['power_mean']})")
    print(f"  castability{bar(r['castability'])} {r['castability']:.2f}      function/mana")
    print(f"  coherence  {bar(r['coherence'])} {r['coherence']:.2f}      color concentration")
    pc = " ".join(f"T{t}:{p:.2f}" for t, p in sorted(r["per_turn_castability"].items()))
    print(f"  on-curve castability by turn: {pc}")
    print("\n  advice:")
    for line in r["advice"]:
        print(f"   • {line}")
    print(f"\n  ({r['_note']})")


if __name__ == "__main__":
    main()
