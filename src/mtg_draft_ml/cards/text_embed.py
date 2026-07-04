"""Frozen oracle-text embeddings (card representation backbone — DD-001).

See docs/architecture.md and design-decisions.md DD-005.

Two embedders behind a common interface (`.dim`, `.embed(list[str]) -> np.ndarray`):

- `SentenceTransformerEmbedder`: the real frozen sentence-transformer (optional [embeddings]
  extra). Computed once per unique card and cached; never runs during training/inference.
- `HashingTextEmbedder`: a dependency-free deterministic fallback (hashed bag-of-tokens) for
  fast local iteration and tests. NOT semantically meaningful — do not report generalization
  numbers with it.

Environment variables
---------------------
MTG_EMBED_MODEL : str, optional
    Sentence-transformer model to use when the caller does not specify one explicitly.
    Defaults to ``all-MiniLM-L6-v2`` (the frozen MiniLM used since DD-001).
MTG_EMBED_CACHE_DIR : str, optional
    Directory for the per-model text embedding cache. Defaults to ``data/text_embed_cache``.
    Set to an empty string to disable caching entirely.
"""
from __future__ import annotations

import os
import pathlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_CACHE_DIR = "data/text_embed_cache"


def _default_model() -> str:
    """Return the default sentence-transformer model (env override or all-MiniLM-L6-v2)."""
    return os.environ.get("MTG_EMBED_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _model_slug(model_name: str) -> str | None:
    """Return a filesystem-safe slug for *model_name*, or None for the default model.

    The default model gets no suffix so existing cache files remain valid without a rebuild.
    All other models get a slug derived from the model name (slashes -> underscores, etc.).
    """
    if model_name == _DEFAULT_MODEL:
        return None
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    return slug.strip("_")


def _cache_path(model_name: str, cache_dir: str) -> pathlib.Path | None:
    """Return the cache file path for *model_name*, or None if caching is disabled."""
    if not cache_dir:
        return None
    slug = _model_slug(model_name)
    fname = "text_embeddings.npz" if slug is None else f"text_embeddings_{slug}.npz"
    return pathlib.Path(cache_dir) / fname


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
    """Frozen sentence-transformer embedder with optional per-model disk cache.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. ``all-MiniLM-L6-v2``, ``BAAI/bge-large-en-v1.5``).
        Defaults to ``_default_model()`` (env ``MTG_EMBED_MODEL`` or ``all-MiniLM-L6-v2``).
    device : str or None
        Device string passed to SentenceTransformer (``cpu``, ``cuda``, …).
    cache_dir : str or None
        Directory for the text-embedding cache. ``None`` uses ``MTG_EMBED_CACHE_DIR`` env var
        or ``data/text_embed_cache``. Pass an empty string ``""`` to disable caching.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 cache_dir: str | None = None):
        if model_name is None:
            model_name = _default_model()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError('install the extra: uv pip install -e ".[embeddings]"') from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # get_sentence_embedding_dimension() is deprecated in newer sentence-transformers;
        # fall back to the new name if available.
        _dim_fn = getattr(self.model, "get_embedding_dimension",
                          getattr(self.model, "get_sentence_embedding_dimension", None))
        self.dim = _dim_fn()

        if cache_dir is None:
            cache_dir = os.environ.get("MTG_EMBED_CACHE_DIR", _DEFAULT_CACHE_DIR)
        self._cache_path = _cache_path(model_name, cache_dir)
        # {normalized_text -> float32 vector}, loaded lazily
        self._cache: dict[str, np.ndarray] | None = None
        self._cache_dirty = False

    def _load_cache(self) -> dict[str, np.ndarray]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        p = self._cache_path
        if p is not None and p.exists():
            # allow_pickle=True needed for object (string) key arrays
            data = np.load(p, allow_pickle=True)
            keys_arr = data["keys"]
            vecs_arr = data["vecs"]
            for k, v in zip(keys_arr, vecs_arr):
                self._cache[str(k)] = v
        return self._cache

    def _save_cache(self) -> None:
        p = self._cache_path
        if p is None or not self._cache_dirty or not self._cache:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._cache.keys()), dtype=object)
        vecs = np.vstack(list(self._cache.values())).astype(np.float32)
        np.savez(p, keys=keys, vecs=vecs)
        self._cache_dirty = False

    def embed(self, texts: list[str]) -> np.ndarray:
        norm = [normalize_oracle_text(t) for t in texts]
        cache = self._load_cache()

        missing_idx = [i for i, t in enumerate(norm) if t not in cache]
        if missing_idx:
            missing_texts = [norm[i] for i in missing_idx]
            vecs = self.model.encode(missing_texts, normalize_embeddings=True,
                                     show_progress_bar=False).astype(np.float32)
            for i, v in zip(missing_idx, vecs):
                cache[norm[i]] = v
            self._cache_dirty = True
            self._save_cache()

        out = np.vstack([cache[t] for t in norm]).astype(np.float32)
        return out


def get_embedder(name: str | None = None, **kw):
    """Return an embedder.

    ``name`` values:
    - ``None``       -> SentenceTransformerEmbedder with the default model
                        (``MTG_EMBED_MODEL`` env var, or ``all-MiniLM-L6-v2``)
    - ``"hash"``     -> HashingTextEmbedder (dev/test only)
    - anything else  -> SentenceTransformerEmbedder(model_name=name)

    Keyword args are forwarded to the embedder constructor (e.g. ``device``, ``cache_dir``).
    """
    if name == "hash":
        return HashingTextEmbedder(**kw)
    return SentenceTransformerEmbedder(model_name=name, **kw)
