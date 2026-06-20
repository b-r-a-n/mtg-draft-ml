"""Tests for the Phase-0 metrics and training loop."""
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.eval.metrics import PickEvaluator  # noqa: E402
from mtg_draft_ml.models.baseline import OneHotMLP  # noqa: E402
from mtg_draft_ml.training.losses import pick_cross_entropy  # noqa: E402
from mtg_draft_ml.training.train import train  # noqa: E402

# A draft with >=2 drafts so the draft-level split yields non-empty train+val.
CSV = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_C,pack_card_D,pool_A,pool_B,pool_C,pool_D
d1,0,0,A,3,0,1,1,1,1,0,0,0,0
d1,0,1,B,3,0,0,1,1,1,1,0,0,0
d2,0,0,A,1,2,1,1,1,0,0,0,0,0
d2,0,1,C,1,2,0,0,1,1,1,0,0,0
d3,0,0,A,2,1,1,1,0,1,0,0,0,0
d3,0,1,D,2,1,0,1,0,1,1,0,0,0
"""


def test_pick_evaluator():
    # 3 cards in vocab; pack = {0,1,2}; logits favor card 1.
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.0, 0.1]])
    pack = torch.tensor([[True, True, True], [True, True, False]])
    logits = logits.masked_fill(~pack, float("-inf"))
    target = torch.tensor([1, 0])  # both correct
    pick_no = torch.tensor([0, 1])
    ev = PickEvaluator()
    ev.update(logits, target, pack, pick_no)
    m = ev.compute()
    assert m["top1"] == 1.0
    assert m["mtpd"] == 0.0  # no card ranked above the chosen one
    assert m["acc_by_pick"] == {0: 1.0, 1: 1.0}

    # now a wrong pick: target=card2 (score 0.2); only card1 (5.0) ranks above it
    ev2 = PickEvaluator()
    ev2.update(logits[:1], torch.tensor([2]), pack[:1], torch.tensor([0]))
    m2 = ev2.compute()
    assert m2["top1"] == 0.0
    assert m2["mtpd"] == 1.0


def test_loss_runs_and_masks():
    logits = torch.tensor([[1.0, 2.0, float("-inf")]])
    loss = pick_cross_entropy(logits, torch.tensor([1]))
    assert torch.isfinite(loss)
    w = pick_cross_entropy(logits, torch.tensor([1]), weights=torch.tensor([2.0]))
    assert torch.isfinite(w)


def test_baseline_forward_masks_pack():
    m = OneHotMLP(n_cards=4, hidden=8, layers=2)
    pool = torch.zeros(2, 4)
    pack = torch.tensor([[True, True, False, False], [False, True, True, True]])
    logits = m(pool, pack)
    assert torch.isinf(logits[0, 2]) and logits[0, 2] < 0  # masked
    assert torch.isfinite(logits[0, 0])


def test_train_smoke(tmp_path: pathlib.Path):
    csv = tmp_path / "syn.csv"
    csv.write_text(CSV)
    pq = tmp_path / "out.parquet"
    man = tmp_path / "man.json"
    preprocess_set(csv, pq, man, set_code="TST")

    best = train(
        str(pq), str(man), epochs=2, batch_size=4, hidden=8, layers=2,
        val_frac=0.34, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=0,
    )
    assert 0.0 <= best["top1"] <= 1.0
    assert (tmp_path / "ckpt" / "last.pt").exists()
    # checkpoint carries the vocab size for reload
    ck = torch.load(tmp_path / "ckpt" / "last.pt", weights_only=False)
    assert ck["n_cards"] == json.load(open(man))["n_cards"] == 4
