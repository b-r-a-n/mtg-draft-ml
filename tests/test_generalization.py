"""Tests for the new-set generalization benchmark (logic only; hashing embedder, no downloads)."""
import json

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.eval.generalization import novel_card_mask, run_experiment  # noqa: E402

# Set A: cards A,B,C,D. Set B: shares A,B but adds new cards E,F.
CSV_A = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_C,pack_card_D,pool_A,pool_B,pool_C,pool_D
d1,0,0,A,3,0,1,1,1,1,0,0,0,0
d1,0,1,B,3,0,0,1,1,1,1,0,0,0
d2,0,0,C,1,2,1,1,1,0,0,0,0,0
d2,0,1,A,1,2,1,1,0,1,0,0,1,0
d3,0,0,D,2,1,1,1,0,1,0,0,0,0
d3,0,1,B,2,1,0,1,1,1,0,0,0,1
"""
CSV_B = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_E,pack_card_F,pool_A,pool_B,pool_E,pool_F
e1,0,0,E,3,0,1,1,1,1,0,0,0,0
e1,0,1,A,3,0,1,1,0,1,0,0,1,0
e2,0,0,F,1,2,0,1,1,1,0,0,0,0
e2,0,1,B,1,2,1,1,0,1,0,0,1,0
"""
SCRY = [
    {"name": n, "oracle_id": n.lower(), "cmc": float(i), "type_line": "Creature",
     "colors": ["R"], "color_identity": ["R"], "rarity": "common", "oracle_text": f"text {n}"}
    for i, n in enumerate(["A", "B", "C", "D", "E", "F"])
]


def _prep(tmp_path, name, csv):
    p = tmp_path / f"{name}.csv"; p.write_text(csv)
    pq = tmp_path / f"{name}.parquet"; man = tmp_path / f"{name}.json"
    scry = tmp_path / "scry.json"; scry.write_text(json.dumps(SCRY))
    preprocess_set(p, pq, man, scryfall_path=scry, set_code=name)
    return str(pq), str(man), str(scry)


def test_novel_card_mask(tmp_path):
    _, man_a, _ = _prep(tmp_path, "A", CSV_A)
    _, man_b, _ = _prep(tmp_path, "B", CSV_B)
    mask = novel_card_mask(man_a, man_b)
    # B vocab is sorted: A,B,E,F -> A,B seen (False), E,F novel (True)
    names = [c["name"] for c in json.load(open(man_b))["cards"]]
    novel = {n: bool(mask[i]) for i, n in enumerate(names)}
    assert novel == {"A": False, "B": False, "E": True, "F": True}


def test_run_experiment_cross_set(tmp_path):
    pq_a, man_a, scry = _prep(tmp_path, "A", CSV_A)
    pq_b, man_b, _ = _prep(tmp_path, "B", CSV_B)
    res = run_experiment(
        pq_a, man_a, scry, pq_b, man_b, scry,
        embedder="hash", epochs=2, batch_size=4, val_frac=0.34,
        device="cpu", checkpoint_dir=str(tmp_path / "ck"), seed=0,
    )
    te = res["test"]
    assert 0.0 <= te["top1"] <= 1.0
    assert 0.0 < te["random_floor"] <= 1.0
    assert te["frac_cards_novel"] == 0.5      # E,F of {A,B,E,F}
    assert te["n_novel"] >= 1                  # at least one pick of a novel card (E, F)
    assert res["train"]["n_cards"] == 4 and te["n_cards"] == 4
