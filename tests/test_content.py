"""Tests for the Phase-1 content encoder: features, embeddings, content matrix, model, training."""
import json
import pathlib

import numpy as np
import pytest

from mtg_draft_ml.cards.features import (
    FEATURE_DIM, FEATURE_NAMES, TAG_DIM, TAG_ROLES, TAG_SPEEDS,
    card_features, feature_names, tag_feature_names, tag_features,
)
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


# ---------------------------------------------------------------------------
# WS2.2 — tag-feature unit tests
# ---------------------------------------------------------------------------

def test_tag_dim_constant():
    assert TAG_DIM == 14
    assert len(tag_feature_names()) == TAG_DIM


def test_tag_features_untagged_all_zeros():
    v = tag_features(None)
    assert v.shape == (TAG_DIM,)
    assert np.all(v == 0.0), "untagged card must produce all-zero vector incl. is_tagged=0"


def test_tag_features_unknown_name_all_zeros():
    """Calling tag_features(None) for a name not in the file -> all zeros."""
    v = tag_features(None)
    assert float(v[-1]) == 0.0  # is_tagged must be 0


def test_tag_features_tagged_card():
    rec = {
        "name": "Test Card",
        "removal": 1, "sweeper": 0, "card_advantage": 1, "ramp_or_fixing": 0,
        "evasive": 1, "combat_trick": 0, "bomb": 0.8,
        "role": "payoff", "speed": "aggro",
    }
    v = tag_features(rec)
    assert v.shape == (TAG_DIM,)
    assert v[0] == 1.0  # removal
    assert v[1] == 0.0  # sweeper
    assert v[2] == 1.0  # card_advantage
    assert v[3] == 0.0  # ramp_or_fixing
    assert v[4] == 1.0  # evasive
    assert v[5] == 0.0  # combat_trick
    assert abs(float(v[6]) - 0.8) < 1e-6   # bomb
    # role one-hot: payoff=index 0 in TAG_ROLES
    assert v[7] == 1.0 and v[8] == 0.0 and v[9] == 0.0
    # speed one-hot: aggro=index 0 in TAG_SPEEDS
    assert v[10] == 1.0 and v[11] == 0.0 and v[12] == 0.0
    assert v[13] == 1.0  # is_tagged


def test_tag_features_enum_one_hots():
    """Each valid role/speed value produces exactly one 1 in its one-hot block."""
    for i, role in enumerate(TAG_ROLES):
        rec = {"role": role, "speed": "neutral", "bomb": 0.0}
        v = tag_features(rec)
        role_bits = list(v[7:10])
        assert role_bits[i] == 1.0
        assert sum(role_bits) == 1.0

    for i, speed in enumerate(TAG_SPEEDS):
        rec = {"role": "filler", "speed": speed, "bomb": 0.0}
        v = tag_features(rec)
        speed_bits = list(v[10:13])
        assert speed_bits[i] == 1.0
        assert sum(speed_bits) == 1.0


def test_tag_features_invalid_enum_all_zeros_in_block():
    """Unknown role/speed -> all zeros in that one-hot block (no crash)."""
    rec = {"role": "unknown_role", "speed": "unknown_speed", "bomb": 0.5}
    v = tag_features(rec)
    assert sum(v[7:10]) == 0.0   # role block
    assert sum(v[10:13]) == 0.0  # speed block
    assert v[13] == 1.0          # still tagged


def test_feature_names_with_tags():
    names_no_tags = feature_names(with_tags=False)
    names_with_tags = feature_names(with_tags=True)
    assert len(names_no_tags) == FEATURE_DIM
    assert len(names_with_tags) == FEATURE_DIM + TAG_DIM
    # tag names are appended at the end
    assert names_with_tags[-TAG_DIM:] == tag_feature_names()


def test_content_matrix_default_path_invariance(tmp_path):
    """build_content_matrix with no tags arg must be byte-identical to explicit tags=None."""
    pq, man, scry, matrix_no_tags, info_no_tags = _build(tmp_path)
    from mtg_draft_ml.cards.content_table import build_content_matrix as bcm
    scry_path = tmp_path / "scry.json"
    matrix_explicit_none, info_explicit_none = bcm(man, scry_path,
                                                    embedder=HashingTextEmbedder(dim=32))
    assert np.array_equal(matrix_no_tags, matrix_explicit_none), (
        "Default path (no tags) must be byte-identical to tags=None")
    assert info_no_tags["total_dim"] == info_explicit_none["total_dim"]
    assert "tags" not in info_no_tags
    assert "tags" not in info_explicit_none


def test_content_matrix_with_tags_wider(tmp_path):
    """Content matrix with tags must be exactly TAG_DIM wider than without."""
    pq, man, scry_path, matrix_no_tags, _ = _build(tmp_path)
    # Build a minimal tags dict: tag card "A" only
    tag_dict = {
        "A": {"name": "A", "removal": 1, "sweeper": 0, "card_advantage": 0,
              "ramp_or_fixing": 0, "evasive": 0, "combat_trick": 0,
              "bomb": 0.5, "role": "payoff", "speed": "aggro"},
    }
    from mtg_draft_ml.cards.content_table import build_content_matrix as bcm
    scry = tmp_path / "scry.json"
    matrix_with_tags, info_with = bcm(man, scry, embedder=HashingTextEmbedder(dim=32),
                                       tags=tag_dict)
    assert matrix_with_tags.shape[1] == matrix_no_tags.shape[1] + TAG_DIM
    assert info_with["feature_dim"] == FEATURE_DIM + TAG_DIM
    assert "tags" in info_with
    assert abs(info_with["tag_coverage"] - 0.25) < 0.01  # 1 of 4 cards tagged
