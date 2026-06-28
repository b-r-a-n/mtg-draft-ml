"""Unit tests for the embedding-space tools (eval/card_tools.py) — pure-numpy, no checkpoint/data."""
import numpy as np
import pytest

from mtg_draft_ml.eval import card_tools as ct


def test_neighbors_ranking_and_self_excluded():
    # A and B point the same way (cos≈1); C orthogonal. Neighbors of A: B first, then C, A excluded.
    E = np.array([[1.0, 0.0], [0.9, 0.05], [0.0, 1.0]])
    names = ["A", "B", "C"]
    q, nbrs = ct.neighbors(E, names, "A", k=5)
    assert q == "A"
    assert [n for n, _ in nbrs] == ["B", "C"]            # self excluded, B before C
    assert nbrs[0][1] > nbrs[1][1]


def test_resolve_fuzzy():
    names = ["Overlord of the Balemurk", "Valgavoth, Terror Eater"]
    assert ct._resolve(names, "overlord of the balemurk") == "Overlord of the Balemurk"   # case-insensitive
    assert ct._resolve(names, "valgavoth") == "Valgavoth, Terror Eater"                   # prefix
    with pytest.raises(KeyError):
        ct._resolve(names, "Llanowar Elves")


def test_neighbors_skips_invalid():
    E = np.array([[1.0, 0.0], [0.99, 0.0], [0.98, 0.0]])
    names = ["A", "B", "C"]
    valid = np.array([True, False, True])               # B is a degenerate missing-Scryfall card
    _, nbrs = ct.neighbors(E, names, "A", k=5, valid=valid)
    assert [n for n, _ in nbrs] == ["C"]                # B filtered out despite being nearer


def test_cross_set_analogue():
    Ea = np.array([[1.0, 0.0]]); na = ["DSK-bomb"]
    Eb = np.array([[0.0, 1.0], [0.95, 0.1]]); nb = ["BLB-offcolor", "BLB-bomb"]
    q, nbrs = ct.cross_set_analogues(Ea, na, "DSK-bomb", Eb, nb, k=2)
    assert q == "DSK-bomb"
    assert nbrs[0][0] == "BLB-bomb"                      # the aligned card is the top analogue
