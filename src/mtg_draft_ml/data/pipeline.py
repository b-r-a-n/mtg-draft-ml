"""Phase-0 data pipeline CLI: download -> preprocess -> manifest.

See docs/roadmap.md Phase 0 and docs/data-infra.md.

Examples
--------
Fast local iteration on a real set (small download via sample mode):
    python -m mtg_draft_ml.data.pipeline --set FDN --sample-rows 50000 --scryfall

Full set (downloads the whole .csv.gz):
    python -m mtg_draft_ml.data.pipeline --set FDN --scryfall

Reprocess an already-downloaded / local CSV without downloading:
    python -m mtg_draft_ml.data.pipeline --set FDN --csv data/raw/FDN.PremierDraft.csv.gz
"""
from __future__ import annotations

import argparse
import pathlib

from .preprocess import preprocess_set


def run(
    set_code: str,
    event_type: str = "PremierDraft",
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    scryfall_path: str | None = None,
    fetch_scryfall: bool = False,
    csv_path: str | None = None,
    sample_rows: int | None = None,
    limit_drafts: int | None = None,
    force: bool = False,
) -> dict:
    csv: str | pathlib.Path
    if csv_path is None:
        from .download import download_17lands_draft  # optional dep (requests)
        csv = download_17lands_draft(set_code, event_type, raw_dir,
                                     sample_rows=sample_rows, force=force)
    else:
        csv = csv_path
    if fetch_scryfall and scryfall_path is None:
        from .download import download_scryfall_oracle
        scryfall_path = str(download_scryfall_oracle(force=force))

    processed = pathlib.Path(processed_dir)
    tag = f"{set_code}.{event_type}" + (f".sample{sample_rows}" if sample_rows else "")
    out_parquet = processed / "draft" / f"{tag}.parquet"
    manifest_path = processed / "manifests" / f"{tag}.json"

    manifest = preprocess_set(
        csv, out_parquet, manifest_path,
        scryfall_path=scryfall_path, set_code=set_code, event_type=event_type,
        limit_drafts=limit_drafts,
    )
    print(f"[{tag}] {manifest['n_rows']} picks, {manifest['n_cards']} cards, "
          f"{manifest['oracle_id_matched']} oracle_ids matched, "
          f"{manifest['n_skipped_rows']} skipped -> {out_parquet}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-0 17lands data pipeline")
    ap.add_argument("--set", dest="set_code", required=True)
    ap.add_argument("--event", dest="event_type", default="PremierDraft")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--csv", dest="csv_path", default=None, help="use a local CSV, skip download")
    ap.add_argument("--scryfall", action="store_true", help="download Scryfall oracle bulk")
    ap.add_argument("--scryfall-path", default=None, help="use a local Scryfall JSON")
    ap.add_argument("--sample-rows", type=int, default=None, help="small download for dev")
    ap.add_argument("--limit-drafts", type=int, default=None, help="cap distinct drafts")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    run(
        a.set_code, a.event_type, a.raw_dir, a.processed_dir,
        scryfall_path=a.scryfall_path, fetch_scryfall=a.scryfall,
        csv_path=a.csv_path, sample_rows=a.sample_rows,
        limit_drafts=a.limit_drafts, force=a.force,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
