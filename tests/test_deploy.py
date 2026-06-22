"""Tests for the deployable Drafter: pick ranking, aggressiveness dial, save/load."""
import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.deploy import Drafter  # noqa: E402
from mtg_draft_ml.models.draft_model import ContentDraftModel  # noqa: E402

NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
CFG = {"emb_dim": 8, "enc_hidden": 16, "enc_layers": 2, "pool": "set_transformer",
       "n_heads": 2, "n_sab": 1, "aux_wr": False}


def _drafter(quality):
    torch.manual_seed(0)
    model = ContentDraftModel(torch.randn(5, 9), emb_dim=8, enc_hidden=16, enc_layers=2,
                              pool="set_transformer", n_heads=2)
    return Drafter(model, NAMES, quality, CFG)


def test_pick_returns_ranked_known_cards():
    d = _drafter([0, 0, 0, 0, 0])
    out = d.pick(pool=["Alpha"], pack=["Bravo", "Charlie", "Delta"])
    assert [r["card"] for r in out] == sorted([r["card"] for r in out], key=lambda c: -dict((x["card"], x["prob"]) for x in out)[c])
    assert {r["card"] for r in out} == {"Bravo", "Charlie", "Delta"}
    assert abs(sum(r["prob"] for r in out) - 1.0) < 1e-5


def test_aggressiveness_shifts_toward_high_quality():
    # Charlie has very high quality; cranking aggressiveness should rank it up.
    d = _drafter([0, 0, 9.0, 0, 0])  # Charlie (idx 2) high quality
    base = d.pick(["Alpha"], ["Bravo", "Charlie", "Delta"], aggressiveness=0.0)
    hot = d.pick(["Alpha"], ["Bravo", "Charlie", "Delta"], aggressiveness=5.0)
    p_base = next(r["prob"] for r in base if r["card"] == "Charlie")
    p_hot = next(r["prob"] for r in hot if r["card"] == "Charlie")
    assert p_hot > p_base                          # dial pushes toward the high-quality card
    assert hot[0]["card"] == "Charlie"             # at high aggressiveness it wins


def test_unknown_cards_skipped_and_topk(tmp_path):
    d = _drafter([0] * 5)
    out = d.pick(["Alpha"], ["Bravo", "Nonexistent", "Delta"], top_k=1)
    assert len(out) == 1 and out[0]["card"] in {"Bravo", "Delta"}


def test_save_load_roundtrip(tmp_path):
    d = _drafter([0, 0, 9.0, 0, 0])
    before = d.pick(["Alpha"], ["Bravo", "Charlie", "Delta"], aggressiveness=2.0)
    d.save(tmp_path / "bundle")
    d2 = Drafter.load(tmp_path / "bundle")
    after = d2.pick(["Alpha"], ["Bravo", "Charlie", "Delta"], aggressiveness=2.0)
    assert [r["card"] for r in before] == [r["card"] for r in after]
    for a, b in zip(before, after):
        assert abs(a["prob"] - b["prob"]) < 1e-5
