"""Convert raw 17lands CSVs into a compact integer-index Parquet format.

See docs/roadmap.md Phase 0; docs/data-infra.md; design-decisions.md DD-006.

The raw 17lands draft CSV is very wide: metadata columns plus one `pack_card_<name>` and one
`pool_<name>` column per card in the set (~500+ columns). This module streams it in row-chunks
(never loading the whole file) and emits one compact row per pick:

    draft_id, pack_number, pick_number,
    pick_idx        (global card index chosen)
    pick_pos        (index of the pick within pack_indices -> label for masked softmax)
    pack_indices    list[int32]  (cards available this pick)
    pool_indices    list[int32]  (distinct cards already picked)
    pool_counts     list[int16]  (count of each pooled card)
    event_match_wins, event_match_losses, draft_time,
    user_game_win_rate_bucket, user_n_games_bucket

It also writes a manifest.json with the card vocabulary (index <-> name <-> oracle_id).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"
# metadata columns carried through when present
_META_PASSTHROUGH = [
    "draft_time",
    "rank",
    "user_game_win_rate_bucket",
    "user_n_games_bucket",
]

_ARROW_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("pack_number", pa.int8()),
        ("pick_number", pa.int8()),
        ("pick_idx", pa.int32()),
        ("pick_pos", pa.int16()),
        ("pack_indices", pa.list_(pa.int32())),
        ("pool_indices", pa.list_(pa.int32())),
        ("pool_counts", pa.list_(pa.int16())),
        ("event_match_wins", pa.int8()),
        ("event_match_losses", pa.int8()),
        ("draft_time", pa.string()),
        ("rank", pa.string()),
        ("user_game_win_rate_bucket", pa.float32()),
        ("user_n_games_bucket", pa.float32()),
    ]
)


def build_vocab(columns: list[str]):
    """From CSV header, derive (names, name_to_idx, pack_cols, pool_cols).

    `names` is the sorted union of card names seen in pack_card_* / pool_* columns.
    `pack_cols`/`pool_cols` preserve original column order for fast matrix slicing.
    """
    pack_cols = [c for c in columns if c.startswith(PACK_PREFIX)]
    pool_cols = [c for c in columns if c.startswith(POOL_PREFIX)]
    pack_names = {c[len(PACK_PREFIX):] for c in pack_cols}
    pool_names = {c[len(POOL_PREFIX):] for c in pool_cols}
    names = sorted(pack_names | pool_names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    return names, name_to_idx, pack_cols, pool_cols


def _nonzero_by_row(mat: np.ndarray):
    """Return (list_of_col_index_arrays_per_row, list_of_value_arrays_per_row) for mat > 0."""
    n = mat.shape[0]
    present = mat > 0
    r, c = np.nonzero(present)
    counts = np.bincount(r, minlength=n)
    bounds = np.cumsum(counts)[:-1]
    col_splits = np.split(c, bounds)
    val_splits = np.split(mat[r, c], bounds)
    return col_splits, val_splits


def preprocess_set(
    csv_path: str | pathlib.Path,
    out_parquet: str | pathlib.Path,
    manifest_path: str | pathlib.Path,
    scryfall_path: str | pathlib.Path | None = None,
    set_code: str | None = None,
    event_type: str | None = None,
    limit_drafts: int | None = None,
    batch_size: int = 20_000,
) -> dict:
    """Stream a 17lands CSV into compact Parquet + a manifest. Returns manifest dict."""
    from ..cards.scryfall import resolve_oracle_ids  # local import: optional dep path

    csv_path = pathlib.Path(csv_path)
    out_parquet = pathlib.Path(out_parquet)
    manifest_path = pathlib.Path(manifest_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    names, name_to_idx, pack_cols, pool_cols = build_vocab(header)
    pack_to_global = np.array([name_to_idx[c[len(PACK_PREFIX):]] for c in pack_cols], dtype=np.int32)
    pool_to_global = np.array([name_to_idx[c[len(POOL_PREFIX):]] for c in pool_cols], dtype=np.int32)
    meta_cols = [c for c in _META_PASSTHROUGH if c in header]

    n_rows = n_skipped = n_drafts = 0
    seen_drafts: set[str] = set()
    writer = pq.ParquetWriter(out_parquet, _ARROW_SCHEMA)
    stop = False

    try:
        reader = pd.read_csv(csv_path, chunksize=batch_size)
        for chunk in reader:
            if limit_drafts is not None:
                chunk, stop = _apply_draft_limit(chunk, seen_drafts, limit_drafts)
                if len(chunk) == 0:
                    break
            n_drafts = len(seen_drafts) if limit_drafts is not None else n_drafts

            kept, skipped = _process_chunk(
                chunk, pack_cols, pool_cols, pack_to_global, pool_to_global,
                name_to_idx, meta_cols,
            )
            n_skipped += skipped
            if kept is not None and kept.num_rows:
                writer.write_table(kept)
                n_rows += kept.num_rows
            if stop:
                break
    finally:
        writer.close()

    oracle_ids, n_matched = resolve_oracle_ids(names, scryfall_path)
    manifest = {
        "set_code": set_code,
        "event_type": event_type,
        "source_csv": str(csv_path),
        "parquet": str(out_parquet),
        "n_cards": len(names),
        "n_rows": n_rows,
        "n_skipped_rows": n_skipped,
        "n_drafts": len(seen_drafts) if limit_drafts is not None else None,
        "oracle_id_matched": n_matched,
        "meta_columns": meta_cols,
        "cards": [
            {"index": i, "name": names[i], "oracle_id": oracle_ids[i]}
            for i in range(len(names))
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _apply_draft_limit(chunk: pd.DataFrame, seen: set, limit: int):
    """Keep only rows belonging to the first `limit` distinct draft_ids. Returns (chunk, stop)."""
    ids = chunk["draft_id"].to_numpy()
    keep = np.ones(len(ids), dtype=bool)
    stop = False
    for i, did in enumerate(ids):
        if did not in seen:
            if len(seen) >= limit:
                keep[i:] = False
                stop = True
                break
            seen.add(did)
    return chunk.loc[keep], stop


def _process_chunk(chunk, pack_cols, pool_cols, pack_to_global, pool_to_global,
                   name_to_idx, meta_cols):
    n = len(chunk)
    if n == 0:
        return None, 0
    pack_mat = chunk[pack_cols].fillna(0).to_numpy()
    pool_mat = chunk[pool_cols].fillna(0).to_numpy().astype(np.int32)

    pack_cols_split, _ = _nonzero_by_row(pack_mat)
    pool_cols_split, pool_vals_split = _nonzero_by_row(pool_mat)

    pick_global = chunk["pick"].map(name_to_idx).to_numpy()  # NaN if unknown card
    pnum = chunk["pack_number"].to_numpy()
    pcknum = chunk["pick_number"].to_numpy()
    did = chunk["draft_id"].astype(str).to_numpy()
    emw = chunk.get("event_match_wins", pd.Series([0] * n)).fillna(0).to_numpy()
    eml = chunk.get("event_match_losses", pd.Series([0] * n)).fillna(0).to_numpy()
    meta = {m: chunk[m].to_numpy() for m in meta_cols}

    rec = {k: [] for k in (
        "draft_id", "pack_number", "pick_number", "pick_idx", "pick_pos",
        "pack_indices", "pool_indices", "pool_counts",
        "event_match_wins", "event_match_losses",
        "draft_time", "rank", "user_game_win_rate_bucket", "user_n_games_bucket",
    )}
    skipped = 0
    for i in range(n):
        if np.isnan(pick_global[i]):
            skipped += 1
            continue
        pidx = int(pick_global[i])
        pack = pack_to_global[pack_cols_split[i]]
        pos = np.flatnonzero(pack == pidx)
        if pos.size == 0:  # picked card not in pack -> data anomaly
            skipped += 1
            continue
        rec["draft_id"].append(did[i])
        rec["pack_number"].append(int(pnum[i]))
        rec["pick_number"].append(int(pcknum[i]))
        rec["pick_idx"].append(pidx)
        rec["pick_pos"].append(int(pos[0]))
        rec["pack_indices"].append(pack.astype(np.int32).tolist())
        rec["pool_indices"].append(pool_to_global[pool_cols_split[i]].astype(np.int32).tolist())
        rec["pool_counts"].append(pool_vals_split[i].astype(np.int16).tolist())
        rec["event_match_wins"].append(int(emw[i]))
        rec["event_match_losses"].append(int(eml[i]))
        rec["draft_time"].append(_s(meta.get("draft_time"), i))
        rec["rank"].append(_s(meta.get("rank"), i))
        rec["user_game_win_rate_bucket"].append(_f(meta.get("user_game_win_rate_bucket"), i))
        rec["user_n_games_bucket"].append(_f(meta.get("user_n_games_bucket"), i))

    if not rec["draft_id"]:
        return None, skipped
    table = pa.table(
        {
            "draft_id": pa.array(rec["draft_id"], pa.string()),
            "pack_number": pa.array(rec["pack_number"], pa.int8()),
            "pick_number": pa.array(rec["pick_number"], pa.int8()),
            "pick_idx": pa.array(rec["pick_idx"], pa.int32()),
            "pick_pos": pa.array(rec["pick_pos"], pa.int16()),
            "pack_indices": pa.array(rec["pack_indices"], pa.list_(pa.int32())),
            "pool_indices": pa.array(rec["pool_indices"], pa.list_(pa.int32())),
            "pool_counts": pa.array(rec["pool_counts"], pa.list_(pa.int16())),
            "event_match_wins": pa.array(rec["event_match_wins"], pa.int8()),
            "event_match_losses": pa.array(rec["event_match_losses"], pa.int8()),
            "draft_time": pa.array(rec["draft_time"], pa.string()),
            "rank": pa.array(rec["rank"], pa.string()),
            "user_game_win_rate_bucket": pa.array(rec["user_game_win_rate_bucket"], pa.float32()),
            "user_n_games_bucket": pa.array(rec["user_n_games_bucket"], pa.float32()),
        },
        schema=_ARROW_SCHEMA,
    )
    return table, skipped


def _s(arr, i):
    if arr is None:
        return None
    v = arr[i]
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)


def _f(arr, i):
    if arr is None:
        return None
    v = arr[i]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v
