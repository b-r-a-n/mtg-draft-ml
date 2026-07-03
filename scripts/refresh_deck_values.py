"""Patch per-card `deck_value` (and the dial field `q`) in webapp/data/<SET>.cards.json IN PLACE
from the current data/hf/gamevalue/<SET>.PremierDraft.gamevalue.json — no retrain, no ONNX export.

The dial values are pure card metadata: export_webapp only bakes the *model* into ONNX; deck_value/q
come from the merged ratings at export time (scripts/export_webapp.py `_card_metadata`). So when the
gamevalue fit is redone (e.g. the WS1.1 full-data rebuild), the webapp dial can be refreshed by
regenerating data/ratings_merged/<SET>.merged.json and rewriting ONLY those two keys per card —
leaving every other key (notably `mc`, added later by scripts/export_doctor_assets.py) untouched.

    PYTHONPATH=src python scripts/refresh_deck_values.py                 # all webapp sets
    PYTHONPATH=src python scripts/refresh_deck_values.py --sets DSK,OTJ
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from mtg_draft_ml.eval.game_value import _rank_corr, merged_ratings_with_value
from mtg_draft_ml.eval.winrate import align_winrates


def _f(v):
    """Mirrors scripts/export_webapp.py:302 `_f` (None where NaN, round to 5)."""
    v = float(v)
    return None if not np.isfinite(v) else round(v, 5)


def refresh_set(s, hf_dir, web_dir, merged_dir):
    cards_path = pathlib.Path(web_dir) / "data" / f"{s}.cards.json"
    gamevalue = pathlib.Path(hf_dir) / "gamevalue" / f"{s}.PremierDraft.gamevalue.json"
    if not gamevalue.exists():
        print(f"  {s}: no gamevalue file ({gamevalue}) — skipped")
        return
    blob = json.load(open(cards_path))
    # cards.json records the manifest it was exported against (`"set"` = manifest path) plus the
    # quality_field, so we align against exactly the vocabulary the webapp uses.
    manifest = blob["set"]
    quality_field = blob.get("quality_field", "deck_value")

    ratings = merged_ratings_with_value(
        f"{hf_dir}/ratings/{s}.PremierDraft.ratings.json", gamevalue, f"{merged_dir}/{s}.merged.json")
    # Mirrors scripts/export_webapp.py:262-265 `_card_metadata` — deck_value/q in manifest-index order.
    dv = align_winrates(manifest, ratings, field="deck_value")
    q = align_winrates(manifest, ratings, field=quality_field)

    old = np.array([c["deck_value"] if c["deck_value"] is not None else np.nan
                    for c in sorted(blob["cards"], key=lambda c: c["i"])], dtype=np.float64)
    for c in blob["cards"]:  # update ONLY deck_value/q (mirrors export_webapp.py:275); keep mc etc.
        c["deck_value"] = _f(dv[c["i"]])
        c["q"] = _f(q[c["i"]])
    cards_path.write_text(json.dumps(blob))  # same formatting as export_webapp/export_doctor_assets

    n_old, n_new = int(np.isfinite(old).sum()), int(np.isfinite(dv).sum())
    rho = _rank_corr(old, dv.astype(np.float64))
    print(f"  {s}: {len(blob['cards'])} cards · deck_value non-null {n_old} -> {n_new} · "
          f"spearman(old,new) = {rho:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None, help="comma list; default = all webapp sets")
    ap.add_argument("--hf-dir", default="data/hf")
    ap.add_argument("--web-dir", default="webapp")
    ap.add_argument("--merged-dir", default="data/ratings_merged")
    a = ap.parse_args(argv)

    web = pathlib.Path(a.web_dir)
    sets = ([s.strip() for s in a.sets.split(",")] if a.sets
            else sorted(p.name.split(".")[0] for p in (web / "data").glob("*.meta.json")))
    pathlib.Path(a.merged_dir).mkdir(parents=True, exist_ok=True)
    for s in sets:
        refresh_set(s, a.hf_dir, a.web_dir, a.merged_dir)


if __name__ == "__main__":
    main()
