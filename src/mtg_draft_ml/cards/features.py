"""Structured card-feature extraction from Scryfall attributes.

See docs/architecture.md (card encoder) and research/findings/card-representation.md.

`card_features(record)` -> fixed-length float32 vector, deterministic and ID-free so it works
for cards never seen at training time. Layout is stable (see FEATURE_NAMES) so the same dims mean
the same thing across sets. Multi-faced cards fall back to / aggregate the front face.
"""
from __future__ import annotations

import re

import numpy as np

COLORS = ["W", "U", "B", "R", "G"]
TYPES = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment",
         "Planeswalker", "Land", "Battle"]
RARITY = {"common": 0.0, "uncommon": 1 / 3, "rare": 2 / 3, "mythic": 1.0}
# curated, stable keyword vocabulary (evergreen + common deciduous)
KEYWORDS = [
    "Flying", "First strike", "Double strike", "Deathtouch", "Lifelink", "Trample",
    "Vigilance", "Haste", "Menace", "Reach", "Hexproof", "Ward", "Flash", "Defender",
    "Indestructible", "Protection", "Prowess", "Scry", "Cycling", "Kicker", "Flashback",
    "Convoke", "Equip", "Enchant", "Adapt", "Explore", "Surveil", "Embalm", "Crew", "Mill",
]
_PIP = re.compile(r"\{([^}]+)\}")


def _front(card: dict, key: str, default=None):
    """Top-level value, else front face's value (for split / MDFC / adventure cards)."""
    if card.get(key) not in (None, ""):
        return card[key]
    faces = card.get("card_faces")
    if faces:
        return faces[0].get(key, default)
    return default


def _all_text(card: dict) -> str:
    if card.get("oracle_text"):
        return card["oracle_text"]
    faces = card.get("card_faces")
    if faces:
        return "\n".join(f.get("oracle_text", "") for f in faces)
    return ""


def _num(v) -> tuple[float, float]:
    """Parse a power/toughness/loyalty value -> (numeric, is_variable_flag)."""
    if v is None:
        return 0.0, 0.0
    try:
        return float(str(v)), 0.0
    except ValueError:
        return 0.0, 1.0  # '*', 'X', '1+*', etc.


def _mana_pips(cost: str):
    """Return (w,u,b,r,g, generic, colorless, has_X) pip counts from a mana_cost string."""
    acc = {"W": 0.0, "U": 0.0, "B": 0.0, "R": 0.0, "G": 0.0, "gen": 0.0, "C": 0.0}
    has_x = 0.0
    for tok in _PIP.findall(cost or ""):
        parts = tok.split("/")
        frac = 1.0 / len(parts)
        for p in parts:
            if p.isdigit():
                acc["gen"] += float(p) * frac
            elif p == "X":
                has_x = 1.0
            elif p in acc:
                acc[p] += frac
    return acc["W"], acc["U"], acc["B"], acc["R"], acc["G"], acc["gen"], acc["C"], has_x


def feature_names() -> list[str]:
    names = ["cmc", "cmc_log"]
    names += [f"color_{c}" for c in COLORS]
    names += [f"identity_{c}" for c in COLORS]
    names += [f"type_{t}" for t in TYPES]
    names += ["legendary"]
    names += ["power", "power_var", "toughness", "toughness_var", "loyalty", "loyalty_flag"]
    names += ["rarity"]
    names += [f"pip_{c}" for c in COLORS] + ["pip_generic", "pip_colorless", "pip_X"]
    names += [f"produces_{c}" for c in COLORS] + ["produces_C"]
    names += [f"kw_{k}" for k in KEYWORDS]
    names += ["text_len"]
    return names


FEATURE_NAMES = feature_names()
FEATURE_DIM = len(FEATURE_NAMES)


def card_features(card: dict) -> np.ndarray:
    """Build the structured feature vector for one Scryfall card record."""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    i = 0

    cmc = float(card.get("cmc", 0.0) or 0.0)
    v[i] = cmc; i += 1
    v[i] = np.log1p(cmc); i += 1

    colors = set(_front(card, "colors", []) or [])
    for c in COLORS:
        v[i] = 1.0 if c in colors else 0.0; i += 1
    identity = set(card.get("color_identity", []) or [])
    for c in COLORS:
        v[i] = 1.0 if c in identity else 0.0; i += 1

    type_line = _front(card, "type_line", "") or ""
    for t in TYPES:
        v[i] = 1.0 if t in type_line else 0.0; i += 1
    v[i] = 1.0 if "Legendary" in type_line else 0.0; i += 1

    p, pv = _num(_front(card, "power"))
    t, tv = _num(_front(card, "toughness"))
    loy, _ = _num(_front(card, "loyalty"))
    has_loy = 1.0 if _front(card, "loyalty") is not None else 0.0
    v[i] = p; i += 1
    v[i] = pv; i += 1
    v[i] = t; i += 1
    v[i] = tv; i += 1
    v[i] = loy; i += 1
    v[i] = has_loy; i += 1

    v[i] = RARITY.get(card.get("rarity", ""), 0.0); i += 1

    for x in _mana_pips(_front(card, "mana_cost", "") or ""):  # 8 dims
        v[i] = x; i += 1

    produced = set(card.get("produced_mana", []) or [])
    for c in COLORS + ["C"]:
        v[i] = 1.0 if c in produced else 0.0; i += 1

    text = _all_text(card)
    kw_set = {k.lower() for k in (card.get("keywords") or [])}
    text_lower = text.lower()
    for k in KEYWORDS:
        v[i] = 1.0 if (k.lower() in kw_set or k.lower() in text_lower) else 0.0; i += 1

    v[i] = np.log1p(len(text)); i += 1
    assert i == FEATURE_DIM, (i, FEATURE_DIM)
    return v
