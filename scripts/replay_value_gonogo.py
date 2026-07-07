"""WS3.0 go/no-go: does replay_data give a cast-conditioned card-value signal?

See docs/breakthrough-plan.md WS3.0. Mirrors scripts/game_value_gonogo.py (Step 0) but sources
from the 17lands *replay* CSV instead of the game_data CSV.

Estimand (WS3.0): per-card CAST-CONDITIONED value.
  logistic won ~ Σ cast_count_c + controls
  where cast_count_c is the number of times card c was cast during the game.

  This is a DIFFERENT estimand from game_value / deck_value (WS1.1), which uses deck composition
  (how many copies of each card were in the deck). Cast-conditioning conditions on the game actually
  developing the card — it measures value conditional on being able to cast, not just on being in
  the deck. Cards that are hard to cast or stuck in hand are down-weighted relative to deck_value.

Schema (confirmed 2026-07-07):
  Per-turn columns user_turn_{N}_creatures_cast and user_turn_{N}_non_creatures_cast contain
  PIPE-DELIMITED ARENA CARD ID LISTS (e.g., '92212', '92212|92234'). Empty turns are NaN.
  user_turn_{N}_user_instants_sorceries_cast is a SEPARATE column (not a subset of
  non_creatures_cast — they partition non-creature spells: non_creatures covers artifacts and
  enchantments; instants/sorceries is the spell subset). Both are included; lands are excluded
  (not casts, mirrors the spell-quality framing of game_value).

Arena ID mapping:
  MDFC room cards have a primary arena_id (the card), a secondary face at arena_id+1, and an
  unlocked/variant at arena_id+2. All three map to the same manifest card. ~0.07% of cast tokens
  are unmapped after this (tokens, expected).

Cast-count sanity expectations:
  mean DISTINCT cards cast per game: ~8-20 (vs ~23 in deck_value)
  mean TOTAL casts per game: ~10-25 (each copy of a card cast counts separately)

Gates (mirrors Step 0 / WS1.1):
  (a) split-half rho >= full-data deck_value WS1.1 DSK reference (0.909), with a matched-n
      fairness comparison (subsample game_data to replay's n_games and compute split-half there).
  (b) Spearman(beta_cast, GIH) < 0.95 (it's not a re-derivation of GIH).
  (c) face-plausibility (top/bottom 15 by beta — left for human review).

    PYTHONPATH=src .venv/bin/python scripts/replay_value_gonogo.py --set DSK --sample-rows 20000
    PYTHONPATH=src .venv/bin/python scripts/replay_value_gonogo.py --set DSK  # full run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

import numpy as np
import pandas as pd

from mtg_draft_ml.data.download import download_17lands_replay
from mtg_draft_ml.eval.game_value import (
    _rank_corr,
    build_value_ratings,
    compare_to_ratings,
    fit_card_values,
)

# WS1.1 full-data split-half rho for DSK PremierDraft (the reference gate).
WS11_DSK_SPLIT_HALF_RHO = 0.909
# Gate (b): replay-beta must differ from GIH at least this much.
GIH_STOP_THRESHOLD = 0.95
# Label column
LABEL_COL = "won"
# Controls present in replay CSV (confirmed by schema scout).
REPLAY_CONTROLS = ("on_play", "num_mulligans", "user_game_win_rate_bucket")
# Turn range for cast columns
TURN_RANGE = range(1, 31)


# ---------------------------------------------------------------------------
# Arena ID -> manifest index mapping
# ---------------------------------------------------------------------------

def build_arena_to_manifest(scryfall_path: pathlib.Path,
                             manifest_path: pathlib.Path) -> tuple[dict[int, int], dict]:
    """Build arena_id -> manifest_index mapping.

    MDFC room cards: primary arena_id maps to the card; secondary face (arena_id+1) and
    unlocked/variant (arena_id+2) also map to the same manifest card.

    Returns (arena_to_mfidx, diagnostics_dict).
    """
    scryfall_cards = json.load(open(scryfall_path))
    manifest = json.load(open(manifest_path))
    cards = manifest["cards"]
    name_to_idx = {c["name"]: c["index"] for c in cards}
    n_manifest = len(cards)

    # Primary arena_id -> name
    arena_to_name: dict[int, str] = {}
    mdfc_primaries: list[int] = []
    for c in scryfall_cards:
        aid = c.get("arena_id")
        if aid is None:
            continue
        arena_to_name[aid] = c["name"]
        if "card_faces" in c:
            # MDFC: secondary face (+1) and variant (+2) share the same manifest card
            mdfc_primaries.append(aid)

    for aid in mdfc_primaries:
        name = arena_to_name[aid]
        arena_to_name[aid + 1] = name  # secondary face (e.g., back room)
        arena_to_name[aid + 2] = name  # unlocked/foil variant

    # arena_id -> manifest index (only for cards in the manifest)
    arena_to_mfidx: dict[int, int] = {}
    for aid, name in arena_to_name.items():
        if name in name_to_idx:
            arena_to_mfidx[aid] = name_to_idx[name]

    # Diagnostics: how many manifest cards got an arena_id?
    manifest_names_with_arena = set(arena_to_name.values())
    n_manifest_covered = sum(1 for c in cards if c["name"] in manifest_names_with_arena)

    diag = {
        "n_manifest": n_manifest,
        "n_manifest_covered": n_manifest_covered,
        "n_scryfall_primary": len([c for c in scryfall_cards if c.get("arena_id")]),
        "n_mdfc_cards": len(mdfc_primaries),
        "n_arena_ids_total": len(arena_to_mfidx),
        "manifest_no_arena": [c["name"] for c in cards if c["name"] not in manifest_names_with_arena],
    }
    return arena_to_mfidx, diag


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _resolve_manifest(hf_dir: pathlib.Path, tag: str) -> pathlib.Path:
    """Find <tag>.json or <tag>.sampleN.json under hf_dir/manifests/."""
    exact = hf_dir / "manifests" / f"{tag}.json"
    if exact.exists():
        return exact
    cands = sorted((hf_dir / "manifests").glob(f"{tag}.sample*.json"))
    if cands:
        return cands[0]
    raise SystemExit(f"no manifest for {tag} under {hf_dir / 'manifests'}")


def _cast_usecols(all_cols: list[str]) -> list[str]:
    """Return the set of cast columns to read: creatures_cast + non_creatures_cast +
    user_instants_sorceries_cast for turns 1-30 (user only, no oppo_)."""
    wanted: set[str] = set()
    for t in TURN_RANGE:
        for suffix in ("creatures_cast", "non_creatures_cast",
                       "user_instants_sorceries_cast"):
            col = f"user_turn_{t}_{suffix}"
            if col in set(all_cols):
                wanted.add(col)
    return sorted(wanted)


def _parse_cast_cell(cell_val, arena_to_mfidx: dict[int, int],
                     unmapped_counter: Counter) -> list[int]:
    """Parse a pipe-delimited arena-id cell into manifest indices.

    Skips NaN/empty; counts unmapped IDs in unmapped_counter.
    Returns list of manifest indices (may have duplicates if same card cast twice in a turn).
    """
    if pd.isna(cell_val):
        return []
    s = str(cell_val).strip()
    if not s or s == "nan":
        return []
    result = []
    for part in s.split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            aid = int(float(part))
        except ValueError:
            continue
        idx = arena_to_mfidx.get(aid)
        if idx is None:
            unmapped_counter[aid] += 1
        else:
            result.append(idx)
    return result


def preprocess_replay(
    csv_path: pathlib.Path,
    manifest_path: pathlib.Path,
    scryfall_path: pathlib.Path,
    out_npz: pathlib.Path | None = None,
    controls: tuple[str, ...] = REPLAY_CONTROLS,
    limit_games: int | None = None,
    batch_size: int = 20_000,
) -> dict:
    """Stream-parse the replay CSV into (X, y, C) arrays aligned to the manifest.

    X[i, c] = number of times card c was CAST in game i (cast-conditioned estimand).
    y[i] = won (bool as float32).
    C[i, :] = control variables (on_play, num_mulligans, user_game_win_rate_bucket).

    Returns dict with keys: X, y, C, card_names, control_names, n_games, n_cast_total,
    unmapped_id_fraction, n_manifest_covered, manifest_no_arena.
    """
    if out_npz is not None and out_npz.exists():
        z = np.load(out_npz, allow_pickle=True)
        return {k: z[k] for k in z.files}

    manifest = json.load(open(manifest_path))
    cards = manifest["cards"]
    n_cards = len(cards)
    card_names = [c["name"] for c in sorted(cards, key=lambda d: d["index"])]

    # Build arena_id -> manifest index
    arena_to_mfidx, arena_diag = build_arena_to_manifest(scryfall_path, manifest_path)
    print(f"  [arena_map] manifest={arena_diag['n_manifest']}  "
          f"covered_by_arena_id={arena_diag['n_manifest_covered']}  "
          f"mdfc_room_cards={arena_diag['n_mdfc_cards']} "
          f"(each adds +1/+2 face IDs)")
    if arena_diag["manifest_no_arena"]:
        print(f"  [arena_map] {len(arena_diag['manifest_no_arena'])} manifest cards have no "
              f"arena_id (bonus-sheet cards not in scryfall dump): "
              f"{arena_diag['manifest_no_arena'][:10]}")

    # Read header once to determine usecols
    header_df = pd.read_csv(csv_path, nrows=0)
    all_cols = header_df.columns.tolist()
    cast_cols = _cast_usecols(all_cols)

    missing_controls = [c for c in controls if c not in set(all_cols)]
    if missing_controls:
        raise ValueError(f"control column(s) absent from {csv_path.name}: {missing_controls}")
    if LABEL_COL not in set(all_cols):
        raise ValueError(f"label column '{LABEL_COL}' absent from {csv_path.name}")

    print(f"  [cast_cols] selected {len(cast_cols)} cast columns "
          f"(user turns 1-30: creatures + non_creatures + user_instants_sorceries)")

    usecols = [LABEL_COL, *controls, *cast_cols]

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    n_games = 0
    n_cast_total = 0
    unmapped_counter: Counter = Counter()

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=batch_size,
                              low_memory=False):
        if limit_games is not None and n_games >= limit_games:
            break
        if limit_games is not None:
            chunk = chunk.iloc[: limit_games - n_games]

        n_chunk = len(chunk)
        x = np.zeros((n_chunk, n_cards), dtype=np.float32)

        # Accumulate cast counts per game
        for col in cast_cols:
            col_series = chunk[col]
            for row_i, cell_val in enumerate(col_series):
                for mf_idx in _parse_cast_cell(cell_val, arena_to_mfidx, unmapped_counter):
                    x[row_i, mf_idx] += 1.0
                    n_cast_total += 1

        X_parts.append(x)
        y_parts.append(chunk[LABEL_COL].to_numpy(dtype=np.float32))
        c_parts.append(chunk[list(controls)].to_numpy(dtype=np.float32, na_value=np.nan))
        n_games += n_chunk

    X = np.concatenate(X_parts) if X_parts else np.zeros((0, n_cards), np.float32)
    y = np.concatenate(y_parts) if y_parts else np.zeros((0,), np.float32)
    C = np.concatenate(c_parts) if c_parts else np.zeros((0, len(controls)), np.float32)

    # Impute rare NaN controls with column mean
    if C.size:
        col_mean = np.nanmean(C, axis=0)
        bad = np.isnan(C)
        if bad.any():
            C[bad] = np.take(col_mean, np.nonzero(bad)[1])

    # Unmapped fraction
    n_unmapped = sum(unmapped_counter.values())
    unmapped_frac = n_unmapped / max(n_cast_total + n_unmapped, 1)

    out = {
        "X": X, "y": y, "C": C,
        "card_names": np.array(card_names, dtype=object),
        "control_names": np.array(list(controls), dtype=object),
        "n_games": np.int64(n_games),
        "n_cast_total": np.int64(n_cast_total),
        "unmapped_id_fraction": np.float64(unmapped_frac),
        "n_manifest_covered": np.int64(arena_diag["n_manifest_covered"]),
        "manifest_no_arena": np.array(arena_diag["manifest_no_arena"], dtype=object),
    }

    # Report unmapped
    top5_unmapped = unmapped_counter.most_common(5)
    print(f"  [unmapped] total_cast_tokens={n_cast_total + n_unmapped}  "
          f"unmapped={n_unmapped}  fraction={unmapped_frac:.4f}")
    if top5_unmapped:
        print(f"  [unmapped] top-5 unmapped IDs: "
              f"{[(aid, cnt) for aid, cnt in top5_unmapped]}")
    if unmapped_frac > 0.01:
        print(f"  [WARN] unmapped fraction {unmapped_frac:.3f} > 1% — investigate above IDs!")
    else:
        print(f"  [unmapped] fraction < 1% — expected (tokens/alchemy variants)")

    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **out)
    return out


# ---------------------------------------------------------------------------
# Overlap check: instants_sorceries vs non_creatures_cast
# ---------------------------------------------------------------------------

def overlap_check(csv_path: pathlib.Path, n_rows: int = 20_000,
                  sample_cells: int = 200) -> dict:
    """Check whether user_turn_N_user_instants_sorceries_cast ⊆ non_creatures_cast per turn.

    Samples up to `sample_cells` nonempty turn-cells where both instants_sorceries and
    non_creatures are non-null in the same turn. Returns dict with finding.
    """
    df = pd.read_csv(csv_path, nrows=n_rows, low_memory=False)

    violations = 0
    checked = 0
    # Also count cases where instants_sorceries is nonempty but non_creatures is NaN
    is_nonempty_nc_null = 0
    total_is_nonempty = 0

    for t in TURN_RANGE:
        nc_col = f"user_turn_{t}_non_creatures_cast"
        is_col = f"user_turn_{t}_user_instants_sorceries_cast"
        if nc_col not in df.columns or is_col not in df.columns:
            continue
        for _, row in df.iterrows():
            is_val = row[is_col]
            nc_val = row[nc_col]
            if pd.isna(is_val) or str(is_val).strip() == "":
                continue
            total_is_nonempty += 1
            if pd.isna(nc_val) or str(nc_val).strip() == "":
                is_nonempty_nc_null += 1
            else:
                if checked < sample_cells:
                    nc_set = set(str(nc_val).split("|"))
                    is_set = set(str(is_val).split("|"))
                    if is_set - nc_set:
                        violations += 1
                    checked += 1

    # Interpretation: if instants_sorceries is frequently nonempty when non_creatures is NaN,
    # they are SEPARATE lists (not subset). If violations > 0, also separate.
    is_subset = (violations == 0 and is_nonempty_nc_null == 0)
    conclusion = (
        "SEPARATE_COLUMNS"
        if (is_nonempty_nc_null > 0 or violations > 0)
        else "SUBSET"
    )
    return {
        "is_subset": is_subset,
        "conclusion": conclusion,
        "violations_in_sample": violations,
        "cells_checked": checked,
        "is_nonempty_nc_null": is_nonempty_nc_null,
        "total_is_nonempty": total_is_nonempty,
    }


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------

def _split_half_rho(X: np.ndarray, y: np.ndarray, C: np.ndarray,
                    l2: float, min_support: float, label: str = "") -> tuple[float, int]:
    """Fit on first/second half of games, correlate betas over well-supported cards."""
    n = len(y)
    h = n // 2
    fits = []
    supports = []
    for idx in [slice(0, h), slice(h, n)]:
        Xi, yi, Ci = X[idx], y[idx], C[idx]
        fit = fit_card_values(Xi, yi, Ci, l2=l2)
        fits.append(fit["beta"])
        supports.append(Xi.sum(axis=0))

    well = (supports[0] >= min_support) & (supports[1] >= min_support)
    n_well = int(well.sum())
    b0 = np.where(well, fits[0], np.nan)
    b1 = np.where(well, fits[1], np.nan)
    rho = _rank_corr(b0, b1)
    if label:
        print(f"      split-half [{label}]: rho={rho:.3f}  n_well={n_well}  "
              f"(support>={min_support:.0f} in both halves)")
    return rho, n_well


# ---------------------------------------------------------------------------
# Matched-n fairness comparison
# ---------------------------------------------------------------------------

def _matched_n_split_half(game_npz: pathlib.Path, n_replay: int,
                           l2: float, min_support: float, rng_seed: int = 42) -> float:
    """Load the full game_data npz, subsample to n_replay rows, run split-half."""
    z = np.load(game_npz, allow_pickle=True)
    X_full = z["X"].astype(np.float32)
    y_full = z["y"].astype(np.float32)
    C_full = z["C"].astype(np.float32)
    n_full = len(y_full)
    if n_full <= n_replay:
        rho, _ = _split_half_rho(X_full, y_full, C_full, l2=l2, min_support=min_support,
                                  label=f"game_data (all {n_full} games, not subsampled)")
        return rho
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(n_full, n_replay, replace=False)
    X_sub, y_sub, C_sub = X_full[idx], y_full[idx], C_full[idx]
    rho, _ = _split_half_rho(X_sub, y_sub, C_sub, l2=l2, min_support=min_support,
                              label=f"game_data subsample @ n={n_replay} (matched)")
    return rho


# ---------------------------------------------------------------------------
# Top/bottom β table
# ---------------------------------------------------------------------------

def _topbottom(beta: np.ndarray, support: np.ndarray, card_names: list[str],
                min_support: float, n: int = 15) -> tuple[list[dict], list[dict]]:
    well = support >= min_support
    idx = np.flatnonzero(well & np.isfinite(beta))
    order = np.argsort(beta[idx])
    bottom = [{"name": card_names[idx[k]], "beta": float(beta[idx[k]]),
               "support": int(support[idx[k]])} for k in order[:n]]
    top = [{"name": card_names[idx[k]], "beta": float(beta[idx[k]]),
             "support": int(support[idx[k]])} for k in order[-n:][::-1]]
    return top, bottom


# ---------------------------------------------------------------------------
# Deck-value correlation
# ---------------------------------------------------------------------------

def _load_deck_value_betas(gamevalue_path: pathlib.Path,
                            manifest_path: pathlib.Path) -> np.ndarray:
    """Load deck_value field from a gamevalue JSON aligned to manifest order."""
    manifest = json.load(open(manifest_path))
    cards = sorted(manifest["cards"], key=lambda d: d["index"])
    name_to_idx = {c["name"]: c["index"] for c in cards}
    n_cards = len(cards)
    gv = {r["name"]: r.get("deck_value") for r in json.load(open(gamevalue_path))}
    beta_dv = np.full(n_cards, np.nan)
    for name, val in gv.items():
        if name in name_to_idx and val is not None:
            beta_dv[name_to_idx[name]] = val
    return beta_dv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--event", dest="event_type", default="PremierDraft")
    ap.add_argument("--sample-rows", type=int, default=None,
                    help="stream only first N rows (None = full file)")
    ap.add_argument("--l2", type=float, default=30.0)
    ap.add_argument("--hf-dir", default="data/hf")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cache-dir", default="data/game")
    ap.add_argument("--out", default=None,
                    help="JSON summary output path (default: /tmp/ws30_gonogo.json)")
    a = ap.parse_args(argv)

    tag = f"{a.set_code}.{a.event_type}"
    hf = pathlib.Path(a.hf_dir)
    cache_dir = pathlib.Path(a.cache_dir)
    out_path = pathlib.Path(a.out) if a.out else pathlib.Path("/tmp/ws30_gonogo.json")

    manifest = _resolve_manifest(hf, tag)
    ratings = hf / "ratings" / f"{tag}.ratings.json"
    gamevalue_path = hf / "gamevalue" / f"{tag}.gamevalue.json"
    game_npz_full = cache_dir / f"game.{tag}.full.npz"
    scryfall_path = hf / "scryfall" / f"{a.set_code.lower()}.json"

    if not ratings.exists():
        raise SystemExit(f"missing ratings {ratings} — run hf pull first")
    if not scryfall_path.exists():
        raise SystemExit(f"missing scryfall dump {scryfall_path} — run hf pull first")

    # -----------------------------------------------------------------------
    # Step 1: Download replay data
    # -----------------------------------------------------------------------
    sample_rows = a.sample_rows
    sample_tag = f"sample{sample_rows}" if sample_rows else "full"
    print(f"[1/6] download replay data ({sample_tag}) for {tag} ...")
    csv = download_17lands_replay(a.set_code, a.event_type, a.raw_dir, sample_rows=sample_rows)
    print(f"      -> {csv}")

    # -----------------------------------------------------------------------
    # Step 1b: Overlap check (on first 20k rows)
    # -----------------------------------------------------------------------
    print("[1b/6] overlap check: user_instants_sorceries_cast vs non_creatures_cast ...")
    ov = overlap_check(csv, n_rows=20_000, sample_cells=200)
    print(f"       conclusion={ov['conclusion']}  "
          f"violations_in_{ov['cells_checked']}_cells={ov['violations_in_sample']}  "
          f"is_nonempty_nc_null={ov['is_nonempty_nc_null']}/{ov['total_is_nonempty']}")
    if ov["conclusion"] == "SEPARATE_COLUMNS":
        print("       -> instants/sorceries and non_creatures are SEPARATE (non-overlapping) columns. "
              "BOTH included in cast accumulation (correct).")
    else:
        print("       -> WARNING: instants/sorceries appears to be a subset of non_creatures. "
              "Check cast column logic — may need deduplication.")

    # -----------------------------------------------------------------------
    # Step 2: Parse into cast-count arrays
    # -----------------------------------------------------------------------
    npz = cache_dir / f"replay_cast.{tag}.{sample_tag}.npz"
    print(f"[2/6] preprocess replay -> cast-count matrix (manifest {manifest.name}) ...")
    data = preprocess_replay(csv, manifest, scryfall_path, out_npz=npz,
                             limit_games=sample_rows)
    X, y, C = data["X"], data["y"], data["C"]
    card_names = list(data["card_names"])
    n_games = int(data["n_games"])
    n_cast_total = int(data["n_cast_total"])
    unmapped_frac = float(data["unmapped_id_fraction"])
    n_cards = X.shape[1]
    support = X.sum(axis=0)  # total cast-copies per card across all games

    # Cast-count sanity (the key diagnostic for the new estimand)
    distinct_cast_per_game = (X > 0).sum(axis=1).astype(float)
    total_cast_per_game = X.sum(axis=1)

    print(f"      games={n_games}  cards={n_cards}  win_rate={float(y.mean()):.3f}")
    print(f"      CAST-COUNT SANITY (WS3.0 estimand):")
    print(f"        mean distinct cards cast/game={distinct_cast_per_game.mean():.1f}  "
          f"median={float(np.median(distinct_cast_per_game)):.1f}  "
          f"(expect ~8-20)")
    print(f"        mean total casts/game={total_cast_per_game.mean():.1f}  "
          f"median={float(np.median(total_cast_per_game)):.1f}  "
          f"(expect ~10-25)")
    print(f"        unmapped_id_fraction={unmapped_frac:.4f}  "
          f"(< 0.01 expected; these are tokens/variants)")

    # -----------------------------------------------------------------------
    # Step 3: Fit L2 logistic regression
    # -----------------------------------------------------------------------
    print(f"[3/6] fit L2 logistic regression (l2={a.l2}) ...")
    fit = fit_card_values(X, y, C, l2=a.l2)
    ctrl = dict(zip(list(data["control_names"]), fit["control_coef"].tolist()))
    beta = fit["beta"]
    print(f"      loss={fit['loss']:.4f}  train_acc={fit['train_acc']:.3f}"
          f"  intercept={fit['intercept']:+.3f}")
    print(f"      control coefs (standardized): "
          f"{{ {', '.join(f'{k}={v:+.3f}' for k, v in ctrl.items())} }}")

    # -----------------------------------------------------------------------
    # Step 4: Diagnostics
    # -----------------------------------------------------------------------
    print("[4/6] diagnostics ...")

    # Scale min_support to dataset size.
    # Cast-based support is lower than deck-based (~15 casts/game vs ~40 deck copies/game).
    # Use 1000 at 80k games but scale; floor 50 (casts are sparser).
    min_support = max(50.0, 1000.0 * (n_games / 80_000))
    print(f"      support gate: {min_support:.0f} total casts "
          f"(scaled from 1000 @ 80k games to {n_games} games; floor=50)")

    # 4a. Compare to GIH/IWD ratings
    cmp = compare_to_ratings(beta, manifest, ratings,
                              card_names=card_names, support=support, min_support=min_support)
    print(f"      Spearman(beta_cast, GIH)={cmp['spearman_gih']:.3f}  "
          f"well-sampled={cmp['spearman_gih_well']:.3f}  (n={cmp['n_compared_gih']})")
    print(f"      Spearman(beta_cast, IWD)={cmp['spearman_iwd']:.3f}  "
          f"well-sampled={cmp['spearman_iwd_well']:.3f}  (n_well={cmp['n_well_sampled']})")

    # 4b. Spearman vs full-data deck_value
    spearman_vs_dv = float("nan")
    spearman_vs_dv_well = float("nan")
    if gamevalue_path.exists():
        beta_dv = _load_deck_value_betas(gamevalue_path, manifest)
        spearman_vs_dv = _rank_corr(beta, beta_dv)
        spearman_vs_dv_well = _rank_corr(
            np.where(support >= min_support, beta, np.nan), beta_dv)
        print(f"      Spearman(beta_cast, deck_value_full)={spearman_vs_dv:.3f}  "
              f"well-sampled={spearman_vs_dv_well:.3f}")
    else:
        print(f"      [warn] {gamevalue_path} not found — skipping vs deck_value comparison")

    # 4c. Split-half reliability
    sh_rho_replay, n_well_sh = _split_half_rho(X, y, C, l2=a.l2,
                                                min_support=min_support,
                                                label="replay_cast")

    # 4d. Matched-n split-half on game_data for fairness comparison
    sh_rho_matched = float("nan")
    if game_npz_full.exists():
        print(f"      running matched-n split-half on {game_npz_full.name} ...")
        sh_rho_matched = _matched_n_split_half(game_npz_full, n_replay=n_games,
                                               l2=a.l2, min_support=min_support)
    else:
        print(f"      [warn] {game_npz_full} not found — skipping matched-n comparison")

    # 4e. Top/bottom 15 by beta (well-sampled)
    top15, bottom15 = _topbottom(beta, support, card_names, min_support)
    print(f"\n  Top 15 by beta_cast (well-sampled, support>={min_support:.0f} total casts):")
    for r in top15:
        print(f"    {r['name'][:38]:38s}  beta={r['beta']:+.3f}  n_cast={r['support']}")
    print(f"\n  Bottom 15 by beta_cast (well-sampled):")
    for r in bottom15:
        print(f"    {r['name'][:38]:38s}  beta={r['beta']:+.3f}  n_cast={r['support']}")

    # -----------------------------------------------------------------------
    # Step 5: Gates
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GATES (WS3.0 — cast-conditioned value)")
    print("=" * 70)

    gate_a_pass = sh_rho_replay >= WS11_DSK_SPLIT_HALF_RHO
    print(f"\n  (a) split-half rho: replay_cast={sh_rho_replay:.3f}  "
          f"reference WS1.1 DSK={WS11_DSK_SPLIT_HALF_RHO}  "
          f"-> {'PASS' if gate_a_pass else 'FAIL'}")
    print(f"      FAIRNESS NOTE: replay has n={n_games} games; game_data (full) has ~1M+ games.")
    print(f"      Matched-n split-half on game_data @ n={n_games}: {sh_rho_matched:.3f}"
          if not np.isnan(sh_rho_matched) else
          f"      Matched-n split-half: N/A (game_data full npz not found)")
    print(f"      (A fair comparison is replay rho vs matched-n game_data rho, not vs full WS1.1)")

    gate_b_pass = cmp["spearman_gih"] < GIH_STOP_THRESHOLD
    print(f"\n  (b) Spearman(beta_cast, GIH)={cmp['spearman_gih']:.3f}  "
          f"< {GIH_STOP_THRESHOLD} threshold  "
          f"-> {'PASS (new signal)' if gate_b_pass else 'FAIL (re-derivation)'}")

    print(f"\n  (c) Face-plausibility: review top/bottom 15 above (human gate).")

    overall = "PROCEED" if (gate_a_pass and gate_b_pass) else "NO-GO"
    print(f"\n  ==> DECISION: {overall}  "
          f"(split-half rho {'>='+str(WS11_DSK_SPLIT_HALF_RHO) if gate_a_pass else '<'+str(WS11_DSK_SPLIT_HALF_RHO)}"
          f", GIH {'< threshold' if gate_b_pass else '>= threshold'})")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Write JSON summary
    # -----------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "set": a.set_code, "event": a.event_type, "sample_rows": a.sample_rows,
        "n_games": n_games,
        "n_cast_total": n_cast_total,
        "mean_distinct_cast_per_game": float(distinct_cast_per_game.mean()),
        "mean_total_cast_per_game": float(total_cast_per_game.mean()),
        "unmapped_id_fraction": unmapped_frac,
        "n_manifest_covered_by_arena_id": int(data["n_manifest_covered"]),
        "overlap_check": ov,
        "win_rate": float(y.mean()),
        "l2": a.l2, "min_support": float(min_support),
        "train_acc": fit["train_acc"], "intercept": fit["intercept"],
        "control_coef": ctrl,
        "spearman_gih": cmp["spearman_gih"],
        "spearman_iwd": cmp["spearman_iwd"],
        "spearman_gih_well": cmp["spearman_gih_well"],
        "spearman_iwd_well": cmp["spearman_iwd_well"],
        "n_well_sampled": cmp["n_well_sampled"],
        "spearman_vs_deck_value_full": spearman_vs_dv,
        "spearman_vs_deck_value_full_well": spearman_vs_dv_well,
        "split_half_rho_replay_cast": sh_rho_replay,
        "split_half_n_well": n_well_sh,
        "split_half_rho_matched_game_data": sh_rho_matched,
        "ws11_dsk_reference_rho": WS11_DSK_SPLIT_HALF_RHO,
        "gate_a_split_half": bool(gate_a_pass),
        "gate_b_gih_divergence": bool(gate_b_pass),
        "decision": overall,
        "top15": top15,
        "bottom15": bottom15,
        "promoted_vs_iwd": cmp["promoted_vs_iwd"],
        "demoted_vs_iwd": cmp["demoted_vs_iwd"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
