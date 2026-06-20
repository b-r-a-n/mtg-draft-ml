"""Correctness test for the Phase-0 preprocessor on a synthetic 17lands-shaped CSV."""
import json

import pyarrow.parquet as pq

from mtg_draft_ml.data.preprocess import build_vocab, preprocess_set

# Cards A,B,C,D. Columns mimic the real 17lands schema (pack_card_<name>, pool_<name>).
CSV = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,draft_time,rank,user_game_win_rate_bucket,user_n_games_bucket,pack_card_A,pack_card_B,pack_card_C,pack_card_D,pool_A,pool_B,pool_C,pool_D
d1,0,0,A,2,1,2024-01-01,gold,0.55,100,1,1,1,1,0,0,0,0
d1,0,1,C,2,1,2024-01-01,gold,0.55,100,0,1,1,1,1,0,0,0
d2,0,0,B,0,0,2024-01-02,silver,0.50,30,1,1,0,0,0,0,0,0
"""

SCRYFALL = [
    {"name": "A", "oracle_id": "oid-a"},
    {"name": "B", "oracle_id": "oid-b"},
    {"name": "Z", "oracle_id": "oid-z"},  # extra, unrelated
]


def _read(parquet):
    t = pq.read_table(parquet)
    return t.to_pylist()


def test_build_vocab():
    cols = ["draft_id", "pick", "pack_card_A", "pack_card_B", "pool_A", "pool_B", "pack_number"]
    names, idx, pack_cols, pool_cols = build_vocab(cols)
    assert names == ["A", "B"]
    assert idx == {"A": 0, "B": 1}
    assert pack_cols == ["pack_card_A", "pack_card_B"]
    assert pool_cols == ["pool_A", "pool_B"]


def test_preprocess_set(tmp_path):
    csv = tmp_path / "synthetic.csv"
    csv.write_text(CSV)
    scry = tmp_path / "scryfall.json"
    scry.write_text(json.dumps(SCRYFALL))
    out = tmp_path / "out.parquet"
    man = tmp_path / "manifest.json"

    manifest = preprocess_set(csv, out, man, scryfall_path=scry, set_code="TST")

    # vocab A=0 B=1 C=2 D=3
    assert manifest["n_cards"] == 4
    assert manifest["n_rows"] == 3
    assert manifest["n_skipped_rows"] == 0
    assert manifest["oracle_id_matched"] == 2  # only A,B in scryfall fixture
    assert [c["name"] for c in manifest["cards"]] == ["A", "B", "C", "D"]
    assert manifest["cards"][0]["oracle_id"] == "oid-a"
    assert manifest["cards"][2]["oracle_id"] is None

    rows = _read(out)
    r0, r1, r2 = rows

    # row0: pack=[A,B,C,D]=[0,1,2,3], picked A -> pos 0, empty pool
    assert r0["pack_indices"] == [0, 1, 2, 3]
    assert r0["pick_idx"] == 0 and r0["pick_pos"] == 0
    assert r0["pool_indices"] == [] and r0["pool_counts"] == []
    assert r0["event_match_wins"] == 2

    # row1: pack=[B,C,D]=[1,2,3], picked C -> global 2, pos 1; pool has A x1
    assert r1["pack_indices"] == [1, 2, 3]
    assert r1["pick_idx"] == 2 and r1["pick_pos"] == 1
    assert r1["pool_indices"] == [0] and r1["pool_counts"] == [1]

    # row2: pack=[A,B]=[0,1], picked B -> pos 1
    assert r2["pack_indices"] == [0, 1]
    assert r2["pick_idx"] == 1 and r2["pick_pos"] == 1
    assert r2["user_game_win_rate_bucket"] is None or isinstance(r2["user_game_win_rate_bucket"], float)


def test_limit_drafts(tmp_path):
    csv = tmp_path / "synthetic.csv"
    csv.write_text(CSV)
    out = tmp_path / "out.parquet"
    man = tmp_path / "manifest.json"
    manifest = preprocess_set(csv, out, man, set_code="TST", limit_drafts=1)
    # only d1's two picks survive
    assert manifest["n_rows"] == 2
    assert manifest["n_drafts"] == 1
    assert {r["draft_id"] for r in _read(out)} == {"d1"}
