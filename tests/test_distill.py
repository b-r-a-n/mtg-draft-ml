"""Tests for cold-start LLM-teacher distillation (DD-004 #4).

Offline only: the HeuristicTeacher and a stub Anthropic client stand in for the real LLM, mirroring
the hash-embedder pattern used elsewhere. No network, no API key.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.distill.coldstart import _rank_corr, run_coldstart  # noqa: E402
from mtg_draft_ml.distill.teacher import (  # noqa: E402
    RATING_FIELD,
    AnthropicTeacher,
    HeuristicTeacher,
    build_teacher_ratings,
    card_brief,
    get_teacher,
)

# reuse the generalization fixtures: set A (A-D) + set B (A,B,E,F), shared vocab
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
     "colors": ["R"], "color_identity": ["R"], "rarity": ["common", "uncommon", "rare", "mythic"][i % 4],
     "power": str(i), "toughness": str(i), "mana_cost": f"{{{i}}}{{R}}",
     "oracle_text": f"text {n}"}
    for i, n in enumerate(["A", "B", "C", "D", "E", "F"])
]


def _prep(tmp_path, name, csv):
    p = tmp_path / f"{name}.csv"; p.write_text(csv)
    pq = tmp_path / f"{name}.parquet"; man = tmp_path / f"{name}.json"
    scry = tmp_path / "scry.json"; scry.write_text(json.dumps(SCRY))
    preprocess_set(p, pq, man, scryfall_path=scry, set_code=name)
    return str(pq), str(man), str(scry)


def _ratings_file(tmp_path, names_to_score):
    """Write a 17lands-shaped ratings file with a chosen win-rate per card."""
    p = tmp_path / "real_ratings.json"
    p.write_text(json.dumps([{"name": n, "ever_drawn_win_rate": v}
                             for n, v in names_to_score.items()]))
    return str(p)


def test_card_brief_release_day_only():
    brief = card_brief(SCRY[2])
    # the teacher sees text + stats, and crucially NOT win rate / pick rate / meta
    assert brief["name"] == "C" and brief["oracle_text"] == "text C"
    assert "win_rate" not in brief and "pick_rate" not in brief
    assert set(brief) == {"name", "mana_cost", "cmc", "type_line", "power", "toughness",
                          "rarity", "oracle_text"}


def test_heuristic_teacher_monotone_in_rarity():
    t = HeuristicTeacher()
    common = card_brief({"name": "x", "rarity": "common", "cmc": 2, "power": "2", "toughness": "2"})
    mythic = card_brief({"name": "y", "rarity": "mythic", "cmc": 2, "power": "2", "toughness": "2"})
    lo, hi = t.rate([common, mythic])
    assert 0.0 <= lo <= hi <= 10.0


def test_build_teacher_ratings_shape_and_cache(tmp_path):
    _, man, scry = _prep(tmp_path, "A", CSV_A)
    out = tmp_path / "teacher.json"
    ratings = build_teacher_ratings(man, scry, HeuristicTeacher(), out_path=str(out),
                                    cache_dir=str(tmp_path / "cache"), set_code="A")
    # 17lands-shaped: list of {name, RATING_FIELD}, one per manifest card
    names = {c["name"] for c in json.load(open(man))["cards"]}
    assert {r["name"] for r in ratings} == names
    assert all(RATING_FIELD in r for r in ratings)
    assert json.load(open(out)) == ratings
    # second call is served entirely from cache (no teacher invocation)
    cache = tmp_path / "cache" / "heuristic" / "A.json"
    assert cache.exists()
    boom = _BoomTeacher()
    again = build_teacher_ratings(man, scry, boom, cache_dir=str(tmp_path / "cache"), set_code="A")
    assert {r["name"] for r in again} == names and not boom.called


class _BoomTeacher:
    id = "heuristic"  # same id -> hits the HeuristicTeacher's cache

    def __init__(self):
        self.called = False

    def rate(self, briefs):
        self.called = True
        raise AssertionError("should not be called when fully cached")


def test_align_with_winrate_helper(tmp_path):
    """Teacher ratings flow through the existing align_winrates unchanged (the whole design point)."""
    from mtg_draft_ml.eval.winrate import align_winrates

    _, man, scry = _prep(tmp_path, "A", CSV_A)
    out = tmp_path / "teacher.json"
    build_teacher_ratings(man, scry, HeuristicTeacher(), out_path=str(out),
                          cache_dir=str(tmp_path / "cache"), set_code="A")
    arr = align_winrates(man, str(out), field=RATING_FIELD)  # manifest-index order
    assert arr.shape[0] == len(json.load(open(man))["cards"])
    assert not (arr != arr).all()  # at least some non-NaN


def test_anthropic_teacher_with_stub_client():
    """AnthropicTeacher parses a stubbed structured response and aligns scores by name."""
    pytest.importorskip("pydantic")  # schema construction needs the [distill] extra

    class _Parsed:
        def __init__(self, ratings):
            self.parsed_output = type("B", (), {"ratings": ratings})()

    class _Rating:
        def __init__(self, name, power):
            self.name, self.power = name, power

    class _Messages:
        def parse(self, **kw):
            # echo a rating for each card mentioned in the prompt, scaled by name
            return _Parsed([_Rating("A", 9.0), _Rating("B", 3.0)])

    class _Client:
        messages = _Messages()

    t = AnthropicTeacher(client=_Client(), chunk_size=10)
    scores = t.rate([card_brief(SCRY[0]), card_brief(SCRY[1])])
    assert scores == [9.0, 3.0]


def test_get_teacher_resolves():
    assert isinstance(get_teacher("heuristic"), HeuristicTeacher)
    assert isinstance(get_teacher("anthropic"), AnthropicTeacher)
    assert get_teacher("anthropic:claude-opus-4-8").model == "claude-opus-4-8"
    with pytest.raises(ValueError):
        get_teacher("gpt")


def test_rank_corr():
    import numpy as np

    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _rank_corr(a, a)["spearman"] == pytest.approx(1.0)
    assert _rank_corr(a, a[::-1].copy())["spearman"] == pytest.approx(-1.0)
    # NaNs are dropped pairwise
    c = _rank_corr(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert c["n"] == 2


def test_run_coldstart_end_to_end(tmp_path):
    """Full experiment on tiny data: train on B, hold out A, heuristic teacher, real ratings."""
    pq_a, man_a, scry = _prep(tmp_path, "A", CSV_A)
    pq_b, man_b, _ = _prep(tmp_path, "B", CSV_B)
    teacher_out = tmp_path / "teacher_A.json"
    build_teacher_ratings(man_a, scry, HeuristicTeacher(), out_path=str(teacher_out),
                          cache_dir=str(tmp_path / "cache"), set_code="A")
    real = _ratings_file(tmp_path, {"A": 0.58, "B": 0.55, "C": 0.50, "D": 0.47})

    res = run_coldstart(
        [{"parquet": pq_b, "manifest": man_b, "scryfall": scry}],
        {"parquet": pq_a, "manifest": man_a, "scryfall": scry},
        coldstart_ratings=str(teacher_out), holdout_ratings=real,
        embedder="hash", emb_dim=16, enc_hidden=32, enc_layers=2, pool="set_transformer",
        blend_alphas=[0.5, 2.0], epochs=2, batch_size=4, val_frac=0.5,
        warmup_frac=0.0, grad_clip=0.0, device="cpu",
        checkpoint_dir=str(tmp_path / "ck"), seed=0,
    )
    assert res["mode"] == "coldstart"
    assert res["baseline"]["alpha"] == 0
    assert len(res["coldstart_sweep"]) == 2 and len(res["oracle_sweep"]) == 2
    for s in res["coldstart_sweep"] + res["oracle_sweep"]:
        assert 0.0 <= s["top1"] <= 1.0
    assert "spearman" in res["teacher_vs_real"]
    assert res["holdout"]["human_wr_agreement"] is not None
