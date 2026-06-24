"""Download 17lands public draft data and Scryfall bulk card data.

See docs/data-infra.md and docs/roadmap.md Phase 0.

- 17lands draft data: a gzipped CSV per (set, event) on a public S3 bucket.
- Scryfall: the `oracle_cards` bulk JSON (one entry per unique oracle card).

`download_17lands_draft(..., sample_rows=N)` streams + gunzips on the fly and writes only
the first N rows to a plain .csv — a few MB instead of GBs, for fast local iteration.
"""
from __future__ import annotations

import gzip
import pathlib

# 17lands public bucket. e.g. .../draft_data_public.FDN.PremierDraft.csv.gz
_17L_TMPL = (
    "https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/"
    "draft_data_public.{set_code}.{event_type}.csv.gz"
)
# 17lands per-game data — one row per GAME (not pick), with `won` + per-card deck/draw columns.
# Same bucket, different path/prefix. e.g. .../game_data_public.DSK.PremierDraft.csv.gz
_17L_GAME_TMPL = (
    "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/"
    "game_data_public.{set_code}.{event_type}.csv.gz"
)
# 17lands aggregate card ratings (GIH WR, ALSA, IWD, ...) per set/format.
# A date range is REQUIRED — without it the API returns all-zero counts / null win rates.
_17L_RATINGS = ("https://www.17lands.com/card_ratings/data?expansion={set_code}"
                "&format={event_type}&start_date={start}&end_date={end}")
_SCRYFALL_BULK_INDEX = "https://api.scryfall.com/bulk-data"
# Scryfall asks API clients to send a descriptive User-Agent + Accept.
_SCRYFALL_HEADERS = {"User-Agent": "mtg-draft-ml/0.0 (research)", "Accept": "*/*"}


def _require_requests():
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise ImportError("`requests` is required for downloads: pip install requests") from e
    return requests


def download_17lands_draft(
    set_code: str,
    event_type: str = "PremierDraft",
    dest_dir: str | pathlib.Path = "data/raw",
    sample_rows: int | None = None,
    force: bool = False,
    timeout: int = 120,
) -> pathlib.Path:
    """Download a 17lands draft CSV. Returns the local path.

    sample_rows=N -> stream-decompress and keep only header + N rows as a plain .csv.
    sample_rows=None -> download the full .csv.gz.
    """
    requests = _require_requests()
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = _17L_TMPL.format(set_code=set_code, event_type=event_type)

    if sample_rows:
        out = dest_dir / f"{set_code}.{event_type}.sample{sample_rows}.csv"
        return _stream_sample(requests, url, out, sample_rows, force, timeout)

    out = dest_dir / f"{set_code}.{event_type}.csv.gz"
    if out.exists() and not force:
        return out
    _stream_to_file(requests, url, out, timeout)
    return out


def download_17lands_game(
    set_code: str,
    event_type: str = "PremierDraft",
    dest_dir: str | pathlib.Path = "data/raw",
    sample_rows: int | None = None,
    force: bool = False,
    timeout: int = 120,
) -> pathlib.Path:
    """Download a 17lands per-GAME data CSV (`game_data_public.*`). Returns the local path.

    Same shape/semantics as `download_17lands_draft` (a different S3 prefix): one row per game with
    a binary `won` label, per-card `deck_/drawn_/opening_hand_/...` columns, and game controls
    (`on_play`, `num_mulligans`, `user_game_win_rate_bucket`, ...). The full file is GBs/set, so
    `sample_rows=N` (stream-decompress + keep header + first N rows) is the default for local work.
    """
    requests = _require_requests()
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = _17L_GAME_TMPL.format(set_code=set_code, event_type=event_type)

    if sample_rows:
        out = dest_dir / f"game.{set_code}.{event_type}.sample{sample_rows}.csv"
        return _stream_sample(requests, url, out, sample_rows, force, timeout)

    out = dest_dir / f"game.{set_code}.{event_type}.csv.gz"
    if out.exists() and not force:
        return out
    _stream_to_file(requests, url, out, timeout)
    return out


def _stream_sample(requests, url, out, sample_rows, force, timeout) -> pathlib.Path:
    """Stream-decompress a gzipped 17lands CSV and keep only header + first `sample_rows` rows."""
    if out.exists() and not force:
        return out
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        r.raw.decode_content = False  # the body *is* a gzip file; decode it ourselves
        with gzip.GzipFile(fileobj=r.raw) as gz, open(out, "wb") as f:
            for i, line in enumerate(gz):
                f.write(line)
                if i >= sample_rows:  # header is line 0, then sample_rows rows
                    break
    return out


def download_scryfall_oracle(
    dest_path: str | pathlib.Path = "data/scryfall/oracle-cards.json",
    force: bool = False,
    timeout: int = 120,
) -> pathlib.Path:
    """Resolve the current `oracle_cards` bulk download URI and fetch it."""
    requests = _require_requests()
    dest_path = pathlib.Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force:
        return dest_path

    idx = requests.get(_SCRYFALL_BULK_INDEX, headers=_SCRYFALL_HEADERS, timeout=timeout)
    idx.raise_for_status()
    entries = idx.json()["data"]
    uri = next(e["download_uri"] for e in entries if e["type"] == "oracle_cards")
    _stream_to_file(requests, uri, dest_path, timeout, headers=_SCRYFALL_HEADERS)
    with open(dest_path) as f:
        if f.read(1) != "[":  # sanity: it's a JSON array
            raise ValueError(f"unexpected Scryfall payload at {dest_path}")
    return dest_path


def download_card_ratings(
    set_code: str,
    event_type: str = "PremierDraft",
    dest_dir: str | pathlib.Path = "data/ratings",
    start_date: str = "2019-01-01",
    end_date: str = "2030-01-01",
    force: bool = False,
    timeout: int = 120,
) -> pathlib.Path:
    """Download 17lands aggregate card ratings (GIH WR / ALSA / IWD ...) for a set. Returns path.

    The default wide date range aggregates the set's whole life. A range is required — the API
    returns all-zero counts without one.
    """
    requests = _require_requests()
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{set_code}.{event_type}.ratings.json"
    if out.exists() and not force:
        return out
    url = _17L_RATINGS.format(set_code=set_code, event_type=event_type,
                              start=start_date, end=end_date)
    r = requests.get(url, headers={"User-Agent": "mtg-draft-ml/0.0 (research)"}, timeout=timeout)
    r.raise_for_status()
    data = r.json()  # validate it parses as JSON
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected 17lands ratings payload for {set_code}")
    out.write_text(r.text)
    return out


def _stream_to_file(requests, url, out, timeout, headers=None):
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=out.name) if tqdm else None
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                if bar:
                    bar.update(len(chunk))
        if bar:
            bar.close()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Download 17lands draft/game + Scryfall data")
    ap.add_argument("--set", dest="set_code", required=True)
    ap.add_argument("--event", dest="event_type", default="PremierDraft")
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--game", action="store_true", help="fetch per-game data instead of draft picks")
    ap.add_argument("--scryfall", action="store_true", help="also fetch Scryfall oracle bulk")
    a = ap.parse_args()
    if a.game:
        print("game:", download_17lands_game(a.set_code, a.event_type, a.dest, sample_rows=a.sample_rows))
    else:
        print("draft:", download_17lands_draft(a.set_code, a.event_type, a.dest, sample_rows=a.sample_rows))
    if a.scryfall:
        print("scryfall:", download_scryfall_oracle())
