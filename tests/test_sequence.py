"""Phase-5 sequence model tests: dataset grouping, collate, forward shapes, and CAUSALITY."""
import json

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.data.sequence import (  # noqa: E402
    SequenceDraftDataset,
    collate_sequences,
)
from mtg_draft_ml.models.sequence_model import SequenceDraftModel  # noqa: E402

# Two drafts, 2 picks each (cards A,B,C,D).
CSV = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_C,pack_card_D,pool_A,pool_B,pool_C,pool_D
d1,0,1,B,3,0,0,1,1,1,1,0,0,0
d1,0,0,A,3,0,1,1,1,1,0,0,0,0
d2,0,0,C,1,2,1,1,1,0,0,0,0,0
d2,0,1,D,1,2,0,0,1,1,1,0,0,0
"""


def _prep(tmp_path):
    csv = tmp_path / "s.csv"; csv.write_text(CSV)
    pq = tmp_path / "s.parquet"; man = tmp_path / "s.json"
    preprocess_set(csv, pq, man, set_code="TST")
    return str(pq)


def test_sequence_dataset_orders_by_pick(tmp_path):
    ds = SequenceDraftDataset(_prep(tmp_path))
    assert len(ds) == 2
    d1 = ds[0] if len(ds[0]["packs"]) == 2 else ds[1]
    # d1 rows were given out of order (pick 1 before pick 0); dataset must sort by pick_number
    # pick 0 of d1 = A (global idx 0) from full pack [A,B,C,D]; pick 1 = B
    assert d1["pick_idx"][0] == 0 and d1["pick_idx"][1] == 1


def test_collate_shapes(tmp_path):
    ds = SequenceDraftDataset(_prep(tmp_path))
    b = collate_sequences([ds[0], ds[1]])
    B, T, P = b["pack"].shape
    assert B == 2 and T == 2
    assert b["pack_mask"].shape == (B, T, P)
    assert b["step_mask"].shape == (B, T)
    assert b["label"].shape == (B, T)


def test_forward_shapes_and_masking(tmp_path):
    ds = SequenceDraftDataset(_prep(tmp_path))
    b = collate_sequences([ds[0], ds[1]])
    m = SequenceDraftModel(torch.randn(4, 9), emb_dim=8, enc_hidden=16, enc_layers=2,
                           n_layers=1, n_heads=2).eval()
    with torch.no_grad():
        logits = m(b["pack"], b["pack_mask"], b["step_mask"], b["pick_idx"])
    assert logits.shape == b["pack"].shape  # [B,T,P]
    # padded pack slots are -inf
    assert (logits[~b["pack_mask"]] == float("-inf")).all()


def test_causality_pick_does_not_leak(tmp_path):
    """The model must not see step t's own pick when predicting step t.

    Changing the picked card AT a step must not change that step's OWN prediction logits — only
    later steps' contexts may change. We verify step 0's logits are invariant to step 0's pick_idx.
    """
    ds = SequenceDraftDataset(_prep(tmp_path))
    b = collate_sequences([ds[0], ds[1]])
    m = SequenceDraftModel(torch.randn(4, 9), emb_dim=8, enc_hidden=16, enc_layers=2,
                           n_layers=1, n_heads=2).eval()
    with torch.no_grad():
        base = m(b["pack"], b["pack_mask"], b["step_mask"], b["pick_idx"])
        tampered = b["pick_idx"].clone()
        tampered[:, 0] = (tampered[:, 0] + 1) % 4   # change EARLIEST pick
        alt = m(b["pack"], b["pack_mask"], b["step_mask"], tampered)
    # step 0 prediction must be unchanged (it can't see its own pick); BOS-only context
    assert torch.allclose(base[:, 0, :], alt[:, 0, :], atol=1e-5), "step 0 leaked its own pick!"
    # step 1 prediction SHOULD change (its context includes step 0's pick)
    assert not torch.allclose(base[:, 1, :], alt[:, 1, :], atol=1e-5), "history not used at step 1"


def test_train_smoke(tmp_path):
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    ds = SequenceDraftDataset(_prep(tmp_path))
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_sequences)
    m = SequenceDraftModel(torch.randn(4, 9), emb_dim=8, enc_hidden=16, enc_layers=2,
                           n_layers=1, n_heads=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(3):
        for b in dl:
            logits = m(b["pack"], b["pack_mask"], b["step_mask"], b["pick_idx"])
            loss = F.cross_entropy(logits[b["step_mask"]], b["label"][b["step_mask"]])
            opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)
