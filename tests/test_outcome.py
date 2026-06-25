"""Tests for the outcome eval (estimated deck-WR scoring + replay)."""
import numpy as np

from mtg_draft_ml.eval.outcome import (
    estimated_deck_wr,
    greedy_pick,
    human_pick,
    replay,
    run_eval,
)


def test_estimated_deck_wr_takes_top_spells():
    beta = np.array([1.0, 0.5, 0.2, -0.3, np.nan])
    # top-2 of {1.0,0.5,0.2,-0.3} = 1.0+0.5=1.5 -> sigma(1.5); NaN card skipped
    got = estimated_deck_wr([0, 1, 2, 3, 4], beta, intercept=0.0, n_spells=2)
    assert abs(got - 1 / (1 + np.exp(-1.5))) < 1e-9
    # a stronger pool scores higher
    assert estimated_deck_wr([0, 1], beta, n_spells=2) > estimated_deck_wr([2, 3], beta, n_spells=2)


def test_replay_accumulates_pool():
    draft = [{"pack": [0, 1, 2], "human": 2}, {"pack": [3, 4], "human": 3}]
    assert replay(draft, human_pick) == [2, 3]
    beta = np.array([0.0, 0.0, 0.0, 9.0, 1.0])
    # greedy on beta takes the highest-beta card in each pack
    assert replay(draft, greedy_pick(beta)) == [0, 3]  # pack1 all-0 -> argmax idx0; pack2 -> 3


def test_run_eval_ranks_policies_and_beats_human():
    beta = np.array([2.0, 1.0, 0.0, -1.0])
    drafts = [[{"pack": [0, 1, 2, 3], "human": 2}, {"pack": [0, 1], "human": 1}]]
    policies = {"human": human_pick, "oracle": greedy_pick(beta)}
    res = run_eval(drafts, policies, beta, n_spells=2)
    assert res["mean"]["oracle"] > res["mean"]["human"]          # oracle drafts a stronger pool
    assert res["beats_human_frac"]["oracle"] == 1.0
    assert res["delta_vs_human"]["oracle"] > 0
