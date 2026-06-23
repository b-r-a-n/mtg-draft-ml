"""Tests for ensemble-of-seeds soft-label distillation (DD-004 #1)."""
import json

import pytest

torch = pytest.importorskip("torch")

from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.distill.ensemble import (  # noqa: E402
    EnsembleModel,
    EnsembleTeacher,
    run_ensemble_distill,
)
from mtg_draft_ml.training.losses import pack_distillation_kl, topk_renormalize  # noqa: E402

# reuse the generalization fixtures (set A: A-D, set B: A,B,E,F, shared vocab)
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
SCRY = [{"name": n, "oracle_id": n.lower(), "cmc": float(i), "type_line": "Creature",
         "colors": ["R"], "color_identity": ["R"], "rarity": "common", "oracle_text": f"text {n}"}
        for i, n in enumerate(["A", "B", "C", "D", "E", "F"])]


def _prep(tmp_path, name, csv):
    p = tmp_path / f"{name}.csv"; p.write_text(csv)
    pq = tmp_path / f"{name}.parquet"; man = tmp_path / f"{name}.json"
    scry = tmp_path / "scry.json"; scry.write_text(json.dumps(SCRY))
    preprocess_set(p, pq, man, scryfall_path=scry, set_code=name)
    return str(pq), str(man), str(scry)


# ---- KD loss primitives ----

def test_kl_zero_when_student_matches_teacher():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    teacher = torch.softmax(logits / 2.0, dim=-1)         # teacher == student at temp=2
    kl = pack_distillation_kl(logits, teacher, mask, temp=2.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_lower_when_teacher_agrees():
    logits = torch.tensor([[3.0, 0.0, 0.0]])              # student favors position 0
    mask = torch.ones(1, 3, dtype=torch.bool)
    agree = torch.tensor([[0.9, 0.05, 0.05]])             # teacher also favors 0
    disagree = torch.tensor([[0.05, 0.05, 0.9]])          # teacher favors 2
    assert pack_distillation_kl(logits, agree, mask, 2.0) < pack_distillation_kl(logits, disagree, mask, 2.0)


def test_kl_masks_pads_no_nan():
    # last position is a pad: -inf logit, 0 teacher prob -> must not contribute / NaN
    logits = torch.tensor([[2.0, 1.0, float("-inf")]])
    mask = torch.tensor([[True, True, False]])
    teacher = torch.tensor([[0.7, 0.3, 0.0]])
    kl = pack_distillation_kl(logits, teacher, mask, temp=2.0)
    assert torch.isfinite(kl).all()


def test_topk_renormalize():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = topk_renormalize(probs, mask, k=2)
    assert out[0, 2].item() == 0.0 and out[0, 3].item() == 0.0       # tail dropped
    assert out.sum(-1).item() == pytest.approx(1.0)                  # renormalized
    assert out[0, 0].item() == pytest.approx(0.5 / 0.8)
    # k >= P is a no-op
    assert torch.allclose(topk_renormalize(probs, mask, k=4), probs)


# ---- ensemble teacher ----

def test_ensemble_mean_probs_valid_distribution(tmp_path):
    from mtg_draft_ml.cards.content_table import build_content_matrix
    from mtg_draft_ml.models.draft_model import ContentDraftModel

    _, man, scry = _prep(tmp_path, "A", CSV_A)
    mat, _ = build_content_matrix(man, scry, embedder=None)
    models = [ContentDraftModel(torch.from_numpy(mat), emb_dim=8, enc_hidden=16, enc_layers=2)
              for _ in range(3)]
    ens = EnsembleTeacher(models)
    pool = torch.tensor([[0]]); pool_mask = torch.tensor([[True]])
    pack = torch.tensor([[0, 1, 2]]); pack_mask = torch.tensor([[True, True, False]])  # last is pad
    p = ens.mean_probs(pool, pool_mask, pack, pack_mask, temp=2.0)
    assert p.shape == (1, 3)
    assert p[0, 2].item() == pytest.approx(0.0)               # pad gets 0 mass
    assert p.sum(-1).item() == pytest.approx(1.0)
    # the eval wrapper returns finite log-probs for real cards, -inf-ish for pads
    logits = EnsembleModel(ens)(pool, pool_mask, pack, pack_mask)
    assert torch.isfinite(logits[0, :2]).all() and logits[0, 2].item() < -20


# ---- end-to-end experiment ----

def test_run_ensemble_distill_end_to_end(tmp_path):
    pq_a, man_a, scry = _prep(tmp_path, "A", CSV_A)
    pq_b, man_b, _ = _prep(tmp_path, "B", CSV_B)
    real = tmp_path / "ratings.json"
    real.write_text(json.dumps([{"name": n, "ever_drawn_win_rate": v}
                                for n, v in {"A": .58, "B": .55, "C": .50, "D": .47}.items()]))
    res = run_ensemble_distill(
        [{"parquet": pq_b, "manifest": man_b, "scryfall": scry}],
        {"parquet": pq_a, "manifest": man_a, "scryfall": scry},
        n_teachers=2, distill_lambda=0.5, distill_temp=2.0, distill_topk=2,
        holdout_ratings=str(real), embedder="hash", emb_dim=16, enc_hidden=32, enc_layers=2,
        pool="set_transformer", epochs=2, batch_size=4, val_frac=0.5,
        warmup_frac=0.0, grad_clip=0.0, device="cpu", checkpoint_dir=str(tmp_path / "ck"), seed=0,
    )
    assert res["mode"] == "ensemble_distill" and res["n_teachers"] == 2
    for key in ("baseline", "distilled", "ensemble"):
        m = res[key]
        assert 0.0 <= m["top1"] <= 1.0
        assert m["wr_agreement_model"] is not None       # ratings were provided
    assert res["frac_cards_novel"] == 0.5                 # C,D novel vs train set B


def test_distill_rejects_infonce(tmp_path):
    """Distillation needs the in-pack CE logits; the InfoNCE path must be refused."""
    from mtg_draft_ml.training.train_content import train_loop

    with pytest.raises(ValueError, match="CE path"):
        train_loop(None, [], [], "cpu", epochs=1, lr=1e-3, checkpoint_dir=str(tmp_path),
                   checkpoint_every=0, n_cards=4, tag="x", loss="infonce", n_negatives=8,
                   teacher=object(), distill_lambda=0.5)
