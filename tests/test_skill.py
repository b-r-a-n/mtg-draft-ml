"""Tests for the good-players (skill-filtered) experiment."""
import json

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.dataset import skill_filter_indices  # noqa: E402
from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.distill.skill import run_skill_experiment  # noqa: E402

SCRY = [{"name": n, "oracle_id": n.lower(), "cmc": float(i), "type_line": "Creature",
         "colors": ["R"], "color_identity": ["R"], "rarity": "common", "oracle_text": f"text {n}"}
        for i, n in enumerate(["A", "B", "C", "D", "E", "F"])]


def _skill_csv(drafts, cards):
    """drafts: list of (draft_id, winrate, n_games, rank, [picked card per pick])."""
    hdr = ["draft_id", "pack_number", "pick_number", "pick", "event_match_wins", "event_match_losses",
           "rank", "user_game_win_rate_bucket", "user_n_games_bucket"]
    hdr += [f"pack_card_{c}" for c in cards] + [f"pool_{c}" for c in cards]
    rows = [",".join(hdr)]
    for did, wr, ng, rk, picks in drafts:
        pool = {c: 0 for c in cards}
        for pn, pick in enumerate(picks):
            r = [did, "0", str(pn), pick, "3", "0", rk, str(wr), str(ng)]
            r += [str(1) for _ in cards] + [str(pool[c]) for c in cards]   # offer all cards
            rows.append(",".join(r))
            pool[pick] += 1
    return "\n".join(rows) + "\n"


def _prep(tmp_path, name, drafts, cards):
    p = tmp_path / f"{name}.csv"; p.write_text(_skill_csv(drafts, cards))
    pq = tmp_path / f"{name}.parquet"; man = tmp_path / f"{name}.json"
    scry = tmp_path / "scry.json"; scry.write_text(json.dumps(SCRY))
    preprocess_set(p, pq, man, scryfall_path=scry, set_code=name)
    return str(pq), str(man), str(scry)


def _rich_ratings(tmp_path, name, wr_by_card):
    p = tmp_path / f"rat_{name}.json"
    p.write_text(json.dumps([{"name": n, "ever_drawn_win_rate": v,
                              "drawn_improvement_win_rate": v - 0.5, "avg_pick": (1.0 - v) * 20,
                              "ever_drawn_game_count": 3000, "drawn_game_count": 3000, "pick_count": 3000}
                             for n, v in wr_by_card.items()]))
    return str(p)


# good = winrate>=0.55 & games>=50; rows 0-7 are good drafts, 8-11 are excluded
GOOD_DRAFTS = [
    ("g1", 0.62, 100, "mythic", ["A", "B"]),
    ("g2", 0.60, 50, "diamond", ["C", "A"]),
    ("g3", 0.58, 100, "platinum", ["D", "B"]),
    ("g4", 0.56, 50, "mythic", ["A", "C"]),
    ("b1", 0.45, 100, "bronze", ["B", "A"]),     # low winrate -> excluded
    ("b2", 0.65, 10, "gold", ["A", "D"]),        # low games -> excluded
]


def test_skill_filter_indices(tmp_path):
    pq, _, _ = _prep(tmp_path, "A", GOOD_DRAFTS, ["A", "B", "C", "D"])
    keep = skill_filter_indices(pq, min_winrate=0.55, min_games=50)
    assert keep == [0, 1, 2, 3, 4, 5, 6, 7]                       # the 4 good drafts' 8 picks
    # rank filter
    keep_rank = skill_filter_indices(pq, ranks={"mythic", "diamond"})
    assert keep_rank == [0, 1, 2, 3, 6, 7]                        # g1(mythic), g2(diamond), g4(mythic)
    # no filter -> all rows
    assert skill_filter_indices(pq) == list(range(12))
    # absent columns are skipped (synthetic data without skill cols would keep all)


def test_skill_filter_missing_column_keeps_all(tmp_path):
    # a parquet built without skill columns: filtering on a missing column is a no-op
    from mtg_draft_ml.data.dataset import DraftPickDataset
    csv = "draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pool_A,pool_B\n"
    csv += "d1,0,0,A,3,0,1,1,0,0\nd1,0,1,B,3,0,0,1,1,0\n"
    p = tmp_path / "n.csv"; p.write_text(csv)
    pq = tmp_path / "n.parquet"; man = tmp_path / "n.json"
    scry = tmp_path / "s.json"; scry.write_text(json.dumps(SCRY))
    preprocess_set(p, pq, man, scryfall_path=scry, set_code="N")
    assert skill_filter_indices(str(pq), min_winrate=0.9) == list(range(len(DraftPickDataset(str(pq)))))


def test_run_skill_experiment_end_to_end(tmp_path):
    # train set A (skill cols), holdout B (skill cols); lenient filter so good>=enough data to train
    pq_a, man_a, scry = _prep(tmp_path, "A", GOOD_DRAFTS * 2, ["A", "B", "C", "D"])
    bdrafts = [("e1", 0.62, 100, "mythic", ["E", "A"]), ("e2", 0.60, 80, "diamond", ["F", "B"]),
               ("e3", 0.58, 100, "platinum", ["A", "E"]), ("e4", 0.57, 60, "mythic", ["B", "F"])]
    pq_b, man_b, _ = _prep(tmp_path, "B", bdrafts * 2, ["A", "B", "E", "F"])
    rat_a = _rich_ratings(tmp_path, "A", {"A": .60, "B": .55, "C": .50, "D": .47})
    rat_b = _rich_ratings(tmp_path, "B", {"A": .60, "B": .55, "E": .52, "F": .48})
    res = run_skill_experiment(
        [{"parquet": pq_a, "manifest": man_a, "scryfall": scry, "ratings": rat_a}],
        {"parquet": pq_b, "manifest": man_b, "scryfall": scry}, holdout_ratings=rat_b,
        min_winrate=0.5, min_games=10,                       # lenient: keeps enough to train
        embedder="hash", emb_dim=16, enc_hidden=32, enc_layers=2, pool="set_transformer",
        epochs=2, batch_size=4, val_frac=0.5, warmup_frac=0.0, grad_clip=0.0, device="cpu",
        checkpoint_dir=str(tmp_path / "ck"), seed=0,
    )
    assert res["mode"] == "skill" and 0.0 < res["kept_frac"] <= 1.0
    for k in ("all_base", "all_comp", "good_base", "good_comp"):       # the 2x2
        for view in ("full", "good_holdout"):
            m = res[k][view]
            assert 0.0 <= m["top1"] <= 1.0
            assert m["wr_agreement_model"] is not None
