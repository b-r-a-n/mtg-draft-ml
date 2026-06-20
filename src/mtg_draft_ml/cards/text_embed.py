"""Frozen oracle-text embeddings (card representation backbone — DD-001).

See docs/architecture.md and design-decisions.md DD-005.

Two embedders behind a common interface (`.dim`, `.embed(list[str]) -> np.ndarray`):

- `SentenceTransformerEmbedder`: the real frozen sentence-transformer (optional [embeddings]
  extra). Computed once per unique card and cached; never runs during training/inference.
- `HashingTextEmbedder`: a dependency-free deterministic fallback (hashed bag-of-tokens) for
  fast local iteration and tests. NOT semantically meaningful — do not report generalization
  numbers with it.
"""
from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")


def normalize_oracle_text(text: str) -> str:
    """Light normalization: lowercase, strip reminder text in parentheses, collapse spaces."""
    text = (text or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)        # reminder text
    text = re.sub(r"\s+", " ", text).strip()
    return text


class HashingTextEmbedder:
    """Deterministic hashed bag-of-tokens embedding (no ML deps). For dev/tests only."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(normalize_oracle_text(text)):
            h = hash((self.seed, tok))
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0  # signed hashing
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    """Frozen sentence-transformer embedder (lazy import of the optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError('install the extra: uv pip install -e ".[embeddings]"') from e
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        norm = [normalize_oracle_text(t) for t in texts]
        return self.model.encode(norm, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)


def get_embedder(name: str = "hash", **kw):
    """'hash' -> HashingTextEmbedder; anything else -> SentenceTransformerEmbedder(model=name)."""
    if name == "hash":
        return HashingTextEmbedder(**kw)
    return SentenceTransformerEmbedder(model_name=name, **kw)
