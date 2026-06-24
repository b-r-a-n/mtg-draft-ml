"""Tests for the game_data Step-0 value model: preprocess, fit, and ratings comparison."""
import json

import numpy as np

from mtg_draft_ml.data.download import _17L_GAME_TMPL
from mtg_draft_ml.data.game_preprocess import preprocess_game_set
from mtg_draft_ml.eval.game_value import (
    _rank_corr,
    _rankdata,
    compare_to_ratings,
    fit_card_values,
)

# Manifest cards A,B,C,D (D absent from the game CSV -> stays all-zero). The game CSV mimics the
# real schema: per-card deck_/drawn_ columns (only deck_ is used) + game controls + `won`. One
# `user_game_win_rate_bucket` is blank to exercise NaN imputation. "B, the Big" has a comma in its
# name (quoted) to mirror real 17lands card names.
MANIFEST = {
    "set_code": "TST",
    "n_cards": 4,
    "cards": [
        {"index": 0, "name": "A"},
        {"index": 1, "name": "B, the Big"},
        {"index": 2, "name": "C"},
        {"index": 3, "name": "D"},
    ],
}

CSV = (
    'won,on_play,num_mulligans,user_game_win_rate_bucket,'
    'deck_A,"deck_B, the Big",deck_C,"drawn_B, the Big"\n'
    '1,1,0,0.55,2,1,3,1\n'
    '0,0,1,0.50,1,2,1,0\n'
    '1,1,0,,3,1,2,1\n'
)


def test_download_game_url_template():
    url = _17L_GAME_TMPL.format(set_code="DSK", event_type="PremierDraft")
    assert url.endswith("game_data/game_data_public.DSK.PremierDraft.csv.gz")


def test_preprocess_game_set(tmp_path):
    csv = tmp_path / "game.csv"
    csv.write_text(CSV)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(MANIFEST))

    data = preprocess_game_set(csv, man)
    X, y, C = data["X"], data["y"], data["C"]

    assert X.shape == (3, 4)
    assert list(data["card_names"]) == ["A", "B, the Big", "C", "D"]
    # deck counts align to manifest index; comma-named card parsed via quoting; D column all zero
    np.testing.assert_array_equal(X[0], [2.0, 1.0, 3.0, 0.0])
    np.testing.assert_array_equal(X[1], [1.0, 2.0, 1.0, 0.0])
    np.testing.assert_array_equal(X[:, 3], [0.0, 0.0, 0.0])  # D absent from game CSV
    np.testing.assert_array_equal(y, [1.0, 0.0, 1.0])
    assert int(data["n_cards_present"]) == 3  # A, B, C present in the CSV; D not

    # controls: on_play, num_mulligans, user_game_win_rate_bucket; blank bucket -> column-mean impute
    assert list(data["control_names"]) == ["on_play", "num_mulligans", "user_game_win_rate_bucket"]
    assert not np.isnan(C).any()
    assert C[2, 2] == np.float32((0.55 + 0.50) / 2)  # imputed with the mean of the present rows


def test_preprocess_game_set_npz_cache(tmp_path):
    csv = tmp_path / "game.csv"; csv.write_text(CSV)
    man = tmp_path / "manifest.json"; man.write_text(json.dumps(MANIFEST))
    npz = tmp_path / "cache.npz"
    first = preprocess_game_set(csv, man, out_npz=npz)
    assert npz.exists()
    # second call loads the cache (even if the CSV is gone) and matches
    csv.unlink()
    second = preprocess_game_set(csv, man, out_npz=npz)
    np.testing.assert_array_equal(first["X"], second["X"])
    np.testing.assert_array_equal(first["y"], second["y"])


def test_rankdata_handles_ties():
    np.testing.assert_array_equal(_rankdata(np.array([10.0, 20.0, 30.0])), [0.0, 1.0, 2.0])
    # ties get the average rank
    np.testing.assert_array_equal(_rankdata(np.array([5.0, 5.0, 9.0])), [0.5, 0.5, 2.0])
    assert _rank_corr(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) == 1.0


def test_fit_recovers_card_signal():
    # Synthetic: card 0 strongly helps winning, card 2 strongly hurts; cards 1,3 neutral. Every
    # "deck" has the same total count (collinear with intercept, like real 40-card decks).
    rng = np.random.default_rng(0)
    n = 4000
    base = rng.integers(0, 4, size=(n, 4)).astype(np.float32)
    base[:, 3] = 10 - base[:, :3].sum(axis=1)  # force a constant row-sum of 10
    true = np.array([0.6, 0.0, -0.6, 0.0])
    logit = base @ true - (base @ true).mean()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    C = rng.random((n, 1)).astype(np.float32)

    fit = fit_card_values(base, y, C, l2=1.0)
    beta = fit["beta"]
    assert beta.shape == (4,)
    assert not np.isnan(beta).any()
    # the helpful card outranks the harmful one (relative ordering is what beta encodes)
    assert beta[0] > beta[2]
    assert _rank_corr(beta, true) > 0.5
    assert 0.0 <= fit["train_acc"] <= 1.0


def test_compare_to_ratings(tmp_path):
    man = tmp_path / "manifest.json"; man.write_text(json.dumps(MANIFEST))
    ratings = [
        {"name": "A", "ever_drawn_win_rate": 0.60, "drawn_improvement_win_rate": 0.05},
        {"name": "B, the Big", "ever_drawn_win_rate": 0.55, "drawn_improvement_win_rate": 0.02},
        {"name": "C", "ever_drawn_win_rate": 0.48, "drawn_improvement_win_rate": -0.03},
    ]
    rat = tmp_path / "ratings.json"; rat.write_text(json.dumps(ratings))
    beta = np.array([0.3, 0.1, -0.2, 0.0], dtype=np.float32)
    support = np.array([5000, 5000, 5000, 5000], dtype=np.float32)

    out = compare_to_ratings(beta, man, rat, support=support, min_support=1000, n_movers=2)
    # beta ordering matches GIH ordering here -> strong positive Spearman
    assert out["spearman_gih"] > 0.9
    assert out["n_compared_iwd"] == 3  # D has no rating
    assert out["n_well_sampled"] == 3
    assert {m["name"] for m in out["promoted_vs_iwd"]} | {m["name"] for m in out["demoted_vs_iwd"]}
    assert all(m["support"] == 5000 for m in out["promoted_vs_iwd"])
