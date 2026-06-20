"""Tests for the Phase-1 content encoder: features, embeddings, content matrix, model, training."""
import json
import pathlib

import numpy as np
import pytest

from mtg_draft_ml.cards.features import FEATURE_DIM, FEATURE_NAMES, card_features
from mtg_draft_ml.cards.text_embed import HashingTextEmbedder, normalize_oracle_text

torch = pytest.importorskip("torch")

from mtg_draft_ml.cards.content_table import build_content_matrix  # noqa: E402
from mtg_draft_ml.data.preprocess import preprocess_set  # noqa: E402
from mtg_draft_ml.models.draft_model import ContentDraftModel  # noqa: E402
from mtg_draft_ml.training.train_content import train  # noqa: E402

CSV = """\
draft_id,pack_number,pick_number,pick,event_match_wins,event_match_losses,pack_card_A,pack_card_B,pack_card_C,pack_card_D,pool_A,pool_B,pool_C,pool_D
d1,0,0,A,3,0,1,1,1,1,0,0,0,0
d1,0,1,B,3,0,0,1,1,1,1,0,0,0
d2,0,0,A,1,2,1,1,1,0,0,0,0,0
d2,0,1,C,1,2,0,0,1,1,1,0,0,0
d3,0,0,D,2,1,1,1,0,1,0,0,0,0
d3,0,1,B,2,1,0,1,0,1,0,0,0,1
"""

SCRYFALL = [
    {"name": "A", "oracle_id": "a", "cmc": 1.0, "type_line": "Creature — Goblin",
     "colors": ["R"], "color_identity": ["R"], "power": "2", "toughness": "1",
     "rarity": "common", "mana_cost": "{R}", "oracle_text": "Haste", "keywords": ["Haste"]},
    {"name": "B", "oracle_id": "b", "cmc": 3.0, "type_line": "Instant",
     "colors": ["U"], "color_identity": ["U"], "rarity": "uncommon", "mana_cost": "{2}{U}",
     "oracle_text": "Counter target spell."},
    {"name": "C", "oracle_id": "c", "cmc": 5.0, "type_line": "Creature — Dragon",
     "colors": ["R"], "color_identity": ["R"], "power": "*", "toughness": "4",
     "rarity": "rare", "mana_cost": "{3}{R}{R}", "oracle_text": "Flying", "keywords": ["Flying"]},
    {"name": "D", "oracle_id": "d", "cmc": 0.0, "type_line": "Land",
     "color_identity": [], "rarity": "common", "oracle_text": "{T}: Add {G}.",
     "produced_mana": ["G"]},
]


def test_feature_vector_shape_and_values():
    v = card_features(SCRYFALL[0])
    assert v.shape == (FEATURE_DIM,)
    assert v[FEATURE_NAMES.index("cmc")] == 1.0
    assert v[FEATURE_NAMES.index("color_R")] == 1.0
    assert v[FEATURE_NAMES.index("type_Creature")] == 1.0
    assert v[FEATURE_NAMES.index("kw_Flying")] == 0.0  # A has Haste, not Flying
    assert card_features(SCRYFALL[2])[FEATURE_NAMES.index("kw_Flying")] == 1.0
    # '*' power -> variable flag, numeric 0
    assert card_features(SCRYFALL[2])[FEATURE_NAMES.index("power_var")] == 1.0


def test_feature_handles_missing_fields():
    v = card_features({"name": "Mystery"})  # almost-empty record must not crash
    assert v.shape == (FEATURE_DIM,)
    assert np.isfinite(v).all()


def test_hashing_embedder_deterministic():
    e = HashingTextEmbedder(dim=64)
    a = e.embed(["deal 3 damage", "deal 3 damage", ""])
    assert a.shape == (3, 64)
    assert np.allclose(a[0], a[1])         # deterministic
    assert np.allclose(a[2], 0.0)          # empty -> zero
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5  # L2-normalized


def test_normalize_strips_reminder_text():
    assert "reminder" not in normalize_oracle_text("Flying (reminder here)")


def _build(tmp_path):
    csv = tmp_path / "syn.csv"; csv.write_text(CSV)
    scry = tmp_path / "scry.json"; scry.write_text(json.dumps(SCRYFALL))
    pq = tmp_path / "out.parquet"; man = tmp_path / "man.json"
    preprocess_set(csv, pq, man, scryfall_path=scry, set_code="TST")
    matrix, info = build_content_matrix(man, scry, embedder=HashingTextEmbedder(dim=32))
    return pq, man, scry, matrix, info


def test_content_matrix_aligns_to_vocab(tmp_path):
    _, man, _, matrix, info = _build(tmp_path)
    manifest = json.load(open(man))
    assert matrix.shape[0] == manifest["n_cards"] == 4
    assert matrix.shape[1] == FEATURE_DIM + 32
    assert info["n_missing_scryfall"] == 0


def test_content_model_forward_and_mask(tmp_path):
    _, _, _, matrix, _ = _build(tmp_path)
    model = ContentDraftModel(torch.from_numpy(matrix), emb_dim=16, enc_hidden=32, enc_layers=2)
    pool = torch.tensor([[0, 0]]); pool_mask = torch.tensor([[False, False]])  # empty pool
    pack = torch.tensor([[1, 2, 3]]); pack_mask = torch.tensor([[True, True, False]])
    logits = model(pool, pool_mask, pack, pack_mask)
    assert logits.shape == (1, 3)
    assert torch.isinf(logits[0, 2]) and logits[0, 2] < 0       # padded position masked
    assert torch.isfinite(logits[0, :2]).all()


def test_standardize_matrix():
    from mtg_draft_ml.cards.content_table import standardize_matrix
    m = np.array([[1.0, 100.0, 5.0], [3.0, 300.0, 5.0]], dtype=np.float32)  # col2 zero-variance
    z, stats = standardize_matrix(m)
    assert abs(z[:, 0].mean()) < 1e-5 and abs(z[:, 0].std() - 1.0) < 1e-4
    assert np.allclose(z[:, 2], 0.0)               # zero-variance column -> 0, no div-by-zero
    # applying training stats to another matrix uses the SAME scaling (no refit)
    other = np.array([[5.0, 500.0, 5.0]], dtype=np.float32)
    z2, _ = standardize_matrix(other, stats)
    mean, std = stats
    assert np.allclose(z2[0], (other[0] - mean) / std)


def test_content_train_smoke(tmp_path: pathlib.Path):
    pq, man, scry, _, _ = _build(tmp_path)
    best = train(str(pq), str(man), scryfall=str(scry), embedder="hash",
                 emb_dim=16, enc_hidden=32, enc_layers=2, epochs=2, batch_size=4,
                 val_frac=0.34, device="cpu", checkpoint_dir=str(tmp_path / "ck"), seed=0)
    assert 0.0 <= best["top1"] <= 1.0
    assert (tmp_path / "ck" / "content_last.pt").exists()
