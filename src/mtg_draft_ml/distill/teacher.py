"""LLM card-quality teacher for cold-start distillation (DD-004 pattern #4).

A `Teacher` reads the **release-day-available** view of each card (oracle text + stats — no meta,
no win rate) and emits a per-card quality score. `build_teacher_ratings` aligns those scores to a
set's manifest and writes a ratings JSON in the same shape as a 17lands ratings file, so the result
is a drop-in stand-in for the data that doesn't exist yet on a new set's release day.

Two teachers:
  - `HeuristicTeacher` — deterministic, no API key. A crude stats-only power proxy. It exists so the
    whole pipeline (and the tests) run offline, mirroring the `hash` text-embedder stand-in pattern.
  - `AnthropicTeacher` — the real teacher: Claude reads each card's text+stats and rates its Limited
    power. This is the only component with a genuine LLM advantage on a set with no human data.

Scores are cached per (teacher, set, card) to disk so re-runs are free and partial progress
survives an interruption. The teacher NEVER sees win rate / pick rate / meta — that's the whole
point: it has to judge from the card alone, exactly as a human would on day one.
"""
from __future__ import annotations

import json
import pathlib
from typing import Protocol

import numpy as np

from ..cards.features import _all_text, _front
from ..cards.scryfall import load_scryfall

# Field name under which teacher scores are written, mirroring 17lands' "ever_drawn_win_rate" etc.
RATING_FIELD = "llm_quality"
MODEL = "claude-opus-4-8"


def card_brief(record: dict) -> dict:
    """Compact, release-day-only view of a Scryfall card for the teacher (no meta / win rate)."""
    return {
        "name": record.get("name", ""),
        "mana_cost": _front(record, "mana_cost", "") or "",
        "cmc": float(record.get("cmc", 0.0) or 0.0),
        "type_line": _front(record, "type_line", "") or "",
        "power": _front(record, "power"),
        "toughness": _front(record, "toughness"),
        "rarity": record.get("rarity", ""),
        "oracle_text": _all_text(record),
    }


class Teacher(Protocol):
    """Rate cards 0-10 on Limited power, from text+stats alone. Higher = stronger pick."""

    id: str

    def rate(self, briefs: list[dict]) -> list[float]:
        ...


class HeuristicTeacher:
    """Deterministic stats-only proxy — no API. A floor/baseline teacher and the offline test path.

    Not meant to be *good*: it just turns CMC/rarity/body into a monotone-ish score so the pipeline
    has a quality signal to blend. The real signal comes from `AnthropicTeacher`.
    """

    id = "heuristic"

    def rate(self, briefs: list[dict]) -> list[float]:
        out = []
        for b in briefs:
            rarity = {"common": 0.0, "uncommon": 1.0, "rare": 2.0, "mythic": 3.0}.get(b["rarity"], 0.0)
            body = _safe_num(b.get("power")) + _safe_num(b.get("toughness"))
            cmc = float(b.get("cmc") or 0.0)
            # cheap, statty creatures and higher rarities score higher; pure curve heuristic
            score = 4.0 + 0.8 * rarity + 0.35 * body - 0.25 * max(cmc - 3.0, 0.0)
            out.append(float(np.clip(score, 0.0, 10.0)))
        return out


def _safe_num(v) -> float:
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return 0.0


_TEACHER_SYSTEM = (
    "You are an expert Magic: The Gathering Limited (booster draft) evaluator. You rate how strong "
    "each card is as a draft pick in its set, judging ONLY from its printed text and stats — you "
    "have no win-rate or pick-rate data, because this is a brand-new set on release day. Rate each "
    "card from 0 to 10: 0-2 unplayable/sideboard, 3-4 filler, 5-6 solid playable, 7-8 strong, "
    "9-10 premium bomb/removal. Judge raw power and how reliably the card improves a deck, not how "
    "niche or build-around it is. Return a rating for every card you are given."
)


class AnthropicTeacher:
    """Claude reads oracle text + stats and rates each card's Limited power (the real teacher).

    Calls are batched (`chunk_size` cards per request) and use structured outputs so the result is a
    validated list of {name, power}. Requires `anthropic` + `pydantic` (the project's [embeddings]
    extra installs neither by default — install `anthropic` to use this path) and ANTHROPIC_API_KEY.
    """

    def __init__(self, model: str = MODEL, chunk_size: int = 25, client=None):
        self.model = model
        self.chunk_size = chunk_size
        self.id = f"anthropic-{model}"
        self._client = client  # injectable for tests

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: keep the module importable without the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def rate(self, briefs: list[dict]) -> list[float]:
        scores: dict[str, float] = {}
        for start in range(0, len(briefs), self.chunk_size):
            chunk = briefs[start: start + self.chunk_size]
            for name, power in self._rate_chunk(chunk).items():
                scores[name.lower()] = power
        # return in input order; missing (model dropped a card) -> neutral 5.0, logged by caller
        return [scores.get(b["name"].lower(), 5.0) for b in briefs]

    def _rate_chunk(self, chunk: list[dict]) -> dict[str, float]:
        from pydantic import BaseModel  # lazy

        class CardRating(BaseModel):
            name: str
            power: float

        class RatingBatch(BaseModel):
            ratings: list[CardRating]

        cards_json = json.dumps(
            [{k: b[k] for k in ("name", "mana_cost", "type_line", "power", "toughness",
                                "rarity", "oracle_text")} for b in chunk],
            ensure_ascii=False, indent=2,
        )
        prompt = (
            "Rate each of these cards 0-10 on Limited power (text+stats only). Return one entry per "
            f"card, echoing its exact name.\n\n{cards_json}"
        )
        resp = self._get_client().messages.parse(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},  # judging power level is a reasoning task
            system=_TEACHER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=RatingBatch,
        )
        return {r.name: float(np.clip(r.power, 0.0, 10.0)) for r in resp.parsed_output.ratings}


# ---- ratings-file builder (the public entry point) ----

def _load_cache(path: pathlib.Path) -> dict[str, float]:
    if path.exists():
        return {k: float(v) for k, v in json.load(open(path)).items()}
    return {}


def build_teacher_ratings(
    manifest_path,
    scryfall_path,
    teacher: Teacher,
    out_path: str | None = None,
    cache_dir: str = "data/teacher_cache",
    set_code: str = "set",
) -> list[dict]:
    """Rate every card in a manifest and write a 17lands-shaped ratings JSON.

    Returns (and optionally writes to `out_path`) a list of {"name", RATING_FIELD} — the same shape
    `eval.winrate.align_winrates` consumes, so the teacher's output is a drop-in for a missing
    ratings file. Scores are memoized to `cache_dir/<teacher.id>/<set_code>.json` by card name, so
    only un-rated cards hit the (possibly paid) teacher on a re-run.
    """
    manifest = json.load(open(manifest_path))
    cards = manifest["cards"]                       # [{index, name, oracle_id}]
    by_name = {c.get("name"): c for c in load_scryfall(scryfall_path) if c.get("name")}

    cache_path = pathlib.Path(cache_dir) / teacher.id / f"{set_code}.json"
    cache = _load_cache(cache_path)

    todo = [row for row in cards if row["name"] not in cache and by_name.get(row["name"])]
    if todo:
        briefs = [card_brief(by_name[row["name"]]) for row in todo]
        scores = teacher.rate(briefs)
        for row, s in zip(todo, scores):
            cache[row["name"]] = float(s)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_path, "w"), indent=2)

    ratings = [{"name": row["name"], RATING_FIELD: cache[row["name"]]}
               for row in cards if row["name"] in cache]
    n_missing = len(cards) - len(ratings)
    print(f"teacher={teacher.id}  rated {len(ratings)}/{len(cards)} cards"
          f"{f' ({n_missing} missing Scryfall)' if n_missing else ''}  cache={cache_path}")
    if out_path:
        json.dump(ratings, open(out_path, "w"), indent=2)
        print(f"wrote {out_path}")
    return ratings


class CachedTeacher:
    """Cache-only teacher: never calls an LLM; raises if any card is missing from the disk cache.

    Used when a ratings cache (``data/teacher_cache/<id>/<SET>.json``) is already fully populated by
    an external process and we want to consume it without holding an API key. Because
    ``build_teacher_ratings`` only calls ``teacher.rate`` for cards MISSING from the cache, a fully
    pre-populated cache means ``rate`` is never invoked.
    """

    def __init__(self, id_: str):
        self.id = id_

    def rate(self, briefs: list[dict]) -> list[float]:
        missing = [b.get("name", "?") for b in briefs]
        raise RuntimeError(
            f"cache-only teacher: card missing from cache — {missing}"
        )


def get_teacher(name: str) -> Teacher:
    """Resolve a teacher by name: 'heuristic', 'anthropic[:model]', or 'cached:<id>'."""
    if name == "heuristic":
        return HeuristicTeacher()
    if name == "anthropic" or name.startswith("anthropic:"):
        model = name.split(":", 1)[1] if ":" in name else MODEL
        return AnthropicTeacher(model=model)
    if name.startswith("cached:"):
        id_ = name.split(":", 1)[1]
        return CachedTeacher(id_)
    raise ValueError(f"unknown teacher: {name!r} (use 'heuristic', 'anthropic[:model]', or 'cached:<id>')")


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Build an LLM-teacher card-quality ratings file")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scryfall", required=True)
    ap.add_argument("--out", required=True, help="output ratings .json (17lands-shaped)")
    ap.add_argument("--teacher", default="anthropic", help="'heuristic' or 'anthropic[:model]'")
    ap.add_argument("--cache-dir", default="data/teacher_cache")
    ap.add_argument("--set-code", default="set")
    a = ap.parse_args(argv)
    build_teacher_ratings(a.manifest, a.scryfall, get_teacher(a.teacher),
                          out_path=a.out, cache_dir=a.cache_dir, set_code=a.set_code)


if __name__ == "__main__":  # pragma: no cover
    main()
