"""Tests for the HF integration helpers (pure logic only — no network)."""
from mtg_draft_ml.data import hf


def test_shard_patterns():
    assert hf.shard_patterns("FDN.PremierDraft") == [
        "draft/FDN.PremierDraft.parquet",
        "manifests/FDN.PremierDraft.json",
    ]
    assert hf.shard_patterns("FDN.PremierDraft.sample50000") == [
        "draft/FDN.PremierDraft.sample50000.parquet",
        "manifests/FDN.PremierDraft.sample50000.json",
    ]


def test_all_patterns_covers_layout():
    pats = hf.all_patterns()
    assert "draft/*.parquet" in pats
    assert "manifests/*.json" in pats
    assert "cards/*.parquet" in pats


def test_module_imports_without_hub_installed():
    # functions exist and are callable; the hub dep is only required at call time
    assert callable(hf.push_dataset)
    assert callable(hf.pull_dataset)
    assert callable(hf.push_file)
