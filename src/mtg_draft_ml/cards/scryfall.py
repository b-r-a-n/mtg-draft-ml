"""Scryfall card-attribute access: map 17lands card names -> oracle_id (+ raw card records).

See docs/architecture.md (card encoder) and docs/data-infra.md.

17lands column suffixes are card *names*. We map them to Scryfall `oracle_id` so the card
encoder / embedding table key on a stable identity. Matching is best-effort:
exact name, then case-insensitive, then the front face of split / double-faced names.
"""
from __future__ import annotations

import json
import pathlib


def load_scryfall(path: str | pathlib.Path) -> list[dict]:
    """Load the Scryfall oracle-cards bulk JSON (a list of card objects)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class OracleResolver:
    """Resolve card names to oracle_ids with case-insensitive / front-face fallbacks."""

    def __init__(self, cards: list[dict]):
        self._exact: dict[str, str] = {}
        self._lower: dict[str, str] = {}
        for c in cards:
            oid, name = c.get("oracle_id"), c.get("name")
            if not oid or not name:
                continue
            self._exact.setdefault(name, oid)
            self._lower.setdefault(name.lower(), oid)
            if " // " in name:  # split / DFC / adventure: also index the front face
                front = name.split(" // ", 1)[0]
                self._exact.setdefault(front, oid)
                self._lower.setdefault(front.lower(), oid)

    def resolve(self, name: str) -> str | None:
        if name in self._exact:
            return self._exact[name]
        if name.lower() in self._lower:
            return self._lower[name.lower()]
        if " // " in name:  # 17lands sometimes carries only the front face
            front = name.split(" // ", 1)[0]
            return self._exact.get(front) or self._lower.get(front.lower())
        return None


def resolve_oracle_ids(names: list[str], scryfall_path: str | pathlib.Path | None):
    """Map a list of card names to oracle_ids. Returns (oracle_ids, n_matched).

    Unresolved names map to None. If scryfall_path is None, returns all None.
    """
    if scryfall_path is None:
        return [None] * len(names), 0
    r = OracleResolver(load_scryfall(scryfall_path))
    out = [r.resolve(n) for n in names]
    return out, sum(o is not None for o in out)
