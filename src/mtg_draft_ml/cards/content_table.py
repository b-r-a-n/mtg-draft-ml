"""Build the per-set content feature matrix aligned to the manifest's card vocabulary.

See docs/architecture.md and docs/data-infra.md.

For a set's manifest (index -> name/oracle_id), join each card to its Scryfall record, compute
[structured_features ; text_embedding], and stack into a [n_cards, D] matrix in vocab-index order.
The Phase-1 content model loads this matrix as a frozen lookup (the heavy text encoder runs once,
here — not during training).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .features import FEATURE_DIM, card_features
from .text_embed import get_embedder


def _name_index(cards: list[dict]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for c in cards:
        if c.get("name"):
            idx.setdefault(c["name"], c)
            if " // " in c["name"]:
                idx.setdefault(c["name"].split(" // ", 1)[0], c)
    return idx


def build_content_matrix(manifest_path, scryfall_path, embedder=None, text: bool = True):
    """Return (matrix [n_cards, D] float32, info dict). embedder defaults to HashingTextEmbedder."""
    from .scryfall import load_scryfall

    manifest = json.load(open(manifest_path))
    cards = manifest["cards"]  # [{index, name, oracle_id}], index-ordered
    by_name = _name_index(load_scryfall(scryfall_path))

    embedder = embedder or get_embedder("hash")
    n = len(cards)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    texts: list[str] = []
    n_missing = 0
    for row in cards:
        rec = by_name.get(row["name"])
        if rec is None:
            n_missing += 1
            texts.append("")
            continue
        feats[row["index"]] = card_features(rec)
        from .features import _all_text
        texts.append(_all_text(rec))

    if text:
        text_embs = embedder.embed(texts)
        matrix = np.concatenate([feats, text_embs], axis=1).astype(np.float32)
        text_dim = int(text_embs.shape[1])
    else:
        matrix = feats
        text_dim = 0

    info = {
        "n_cards": n,
        "feature_dim": FEATURE_DIM,
        "text_dim": text_dim,
        "total_dim": int(matrix.shape[1]),
        "n_missing_scryfall": n_missing,
        "embedder": type(embedder).__name__,
        "embedder_model": getattr(embedder, "model_name", None),
    }
    return matrix, info


def build_multiset_content(set_specs, embedder=None, text: bool = True):
    """Build a shared content matrix over the UNION of several sets' cards.

    set_specs: list of {"manifest": ..., "scryfall": ...}. Cards are deduped by oracle_id (or
    lowercased name). Returns (matrix [N_global, D], info, key_to_idx, per_set_local_to_global),
    where per_set_local_to_global[k] is an int array mapping set k's local indices to global rows.
    """
    from .features import FEATURE_DIM, _all_text
    from .scryfall import load_scryfall

    embedder = embedder or get_embedder("hash")
    key_to_idx: dict[str, int] = {}
    feats: list[np.ndarray] = []
    texts: list[str] = []
    per_set: list[np.ndarray] = []

    for spec in set_specs:
        manifest = json.load(open(spec["manifest"]))
        by_name = _name_index(load_scryfall(spec["scryfall"]))
        l2g = np.empty(len(manifest["cards"]), dtype=np.int64)
        for row in manifest["cards"]:  # index-ordered
            key = row.get("oracle_id") or (row.get("name") or "").lower()
            if key not in key_to_idx:
                key_to_idx[key] = len(key_to_idx)
                rec = by_name.get(row["name"])
                feats.append(card_features(rec) if rec else np.zeros(FEATURE_DIM, np.float32))
                texts.append(_all_text(rec) if rec else "")
            l2g[row["index"]] = key_to_idx[key]
        per_set.append(l2g)

    feat_mat = np.vstack(feats).astype(np.float32)
    if text:
        tembs = embedder.embed(texts)
        matrix = np.concatenate([feat_mat, tembs], axis=1).astype(np.float32)
        text_dim = int(tembs.shape[1])
    else:
        matrix = feat_mat
        text_dim = 0
    info = {
        "n_cards": int(matrix.shape[0]),
        "feature_dim": FEATURE_DIM,
        "text_dim": text_dim,
        "total_dim": int(matrix.shape[1]),
        "n_sets": len(set_specs),
        "embedder": type(embedder).__name__,
        "embedder_model": getattr(embedder, "model_name", None),
    }
    return matrix, info, key_to_idx, per_set


def standardize_matrix(matrix: np.ndarray, stats=None, eps: float = 1e-6):
    """Per-column z-score the content matrix. Returns (standardized, (mean, std)).

    Pass `stats` (from the training matrix) to apply the SAME scaling to a holdout matrix — no
    leakage, consistent scale. Fixes the raw-feature / unit-norm-text scale mismatch (Phase 1
    caveat) so the encoder sees all dims on a comparable scale. Zero-variance columns are left as-is.
    """
    if stats is None:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        stats = (mean.astype(np.float32), std.astype(np.float32))
    mean, std = stats
    return ((matrix - mean) / std).astype(np.float32), stats


def save_content_matrix(path, matrix: np.ndarray, info: dict) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    with open(path.with_suffix(".info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return path


def load_content_matrix(path):
    path = pathlib.Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    matrix = np.load(path)
    info = json.load(open(path.with_suffix(".info.json")))
    return matrix, info


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Build content feature matrix for a set")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scryfall", required=True)
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--embedder", default="hash", help="'hash' or a sentence-transformers model")
    a = ap.parse_args(argv)
    matrix, info = build_content_matrix(a.manifest, a.scryfall, embedder=get_embedder(a.embedder))
    save_content_matrix(a.out, matrix, info)
    print(f"content matrix {matrix.shape} -> {a.out}  ({info})")


if __name__ == "__main__":  # pragma: no cover
    main()
