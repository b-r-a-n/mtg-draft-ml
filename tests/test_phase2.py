"""Phase-2 tests: Set Transformer pool encoder + contextual InfoNCE forward path."""
import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.models.draft_model import ContentDraftModel  # noqa: E402
from mtg_draft_ml.models.pool_encoder import (  # noqa: E402
    MeanPoolEncoder,
    SetTransformerEncoder,
)


def test_set_transformer_shapes_and_empty_pool():
    enc = SetTransformerEncoder(d=16, heads=2, n_sab=2)
    embs = torch.randn(3, 5, 16)
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, False, False, False, False],
        [False, False, False, False, False],  # empty pool (pick 0)
    ])
    out = enc(embs, mask)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()           # empty-pool row must not be NaN


def test_set_transformer_permutation_invariant():
    enc = SetTransformerEncoder(d=16, heads=2, n_sab=1).eval()
    embs = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    perm = torch.randperm(4)
    with torch.no_grad():
        a = enc(embs, mask)
        b = enc(embs[:, perm, :], mask)
    assert torch.allclose(a, b, atol=1e-5)


def _model(pool):
    matrix = torch.randn(10, 12)               # 10 cards, 12-d content
    return ContentDraftModel(matrix, emb_dim=8, enc_hidden=16, enc_layers=2, pool=pool, n_heads=2)


@pytest.mark.parametrize("pool", ["mean", "set_transformer"])
def test_forward_both_pools(pool):
    m = _model(pool).eval()
    pool_idx = torch.tensor([[1, 2, 0]])
    pool_mask = torch.tensor([[True, True, False]])
    pack = torch.tensor([[3, 4, 5]])
    pack_mask = torch.tensor([[True, True, False]])
    logits = m(pool_idx, pool_mask, pack, pack_mask)
    assert logits.shape == (1, 3)
    assert torch.isinf(logits[0, 2]) and logits[0, 2] < 0


def test_infonce_forward_appends_negatives():
    m = _model("set_transformer").eval()
    pool_idx = torch.tensor([[1, 2]]); pool_mask = torch.tensor([[True, True]])
    pack = torch.tensor([[3, 4, 5]]); pack_mask = torch.tensor([[True, True, True]])
    neg = torch.tensor([6, 7, 8, 9])           # K=4 global negatives
    logits = m(pool_idx, pool_mask, pack, pack_mask, neg_idx=neg)
    assert logits.shape == (1, 3 + 4)          # pack positions + negatives
    assert torch.isfinite(logits).all()        # negatives are finite (not masked)
    # the positive label (a pack position) is unchanged by appending negatives
    base = m(pool_idx, pool_mask, pack, pack_mask)
    assert torch.allclose(base, logits[:, :3], atol=1e-5)


def test_mean_pool_still_available():
    assert isinstance(_model("mean").pool_encoder, MeanPoolEncoder)
    assert isinstance(_model("set_transformer").pool_encoder, SetTransformerEncoder)
