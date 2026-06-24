"""Stream a 17lands per-GAME CSV into a compact per-game deck-count matrix (Step 0 value model).

See docs/game-data-plan.md (Step 0) and docs/results/game-data-value-model.md.

The `game_data_public.<SET>.<EVENT>.csv` is one row per game and ~1500 columns wide: game controls
plus one `deck_<card>` / `drawn_<card>` / `opening_hand_<card>` / ... per card in the set. For the
go/no-go value-model fit we need only:

    X        [n_games, n_cards]  count of each card in the deck (manifest-index order)
    y        [n_games]           `won` (0/1)
    controls [n_games, n_ctrl]   on_play, num_mulligans, user_game_win_rate_bucket (+ intercept-free)

This module reads the CSV in row-chunks (never the whole file), keeps only the needed columns
(`usecols`), and aligns the wide `deck_*` block to a draft manifest's vocabulary so the resulting
`X` columns line up 1:1 with `ratings`/manifest card indices. Card names containing commas are
quoted in the CSV, so we rely on pandas' CSV parser (not a naive split) — same as `preprocess.py`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

DECK_PREFIX = "deck_"
LABEL_COL = "won"
# Pre-outcome game controls. Deliberately NOT `num_turns` — turn count is a *consequence* of the
# game (a mediator on the path to the result), so conditioning on it would bias the card
# coefficients. on_play / mulligans / player-skill are determined before/independent of the result.
DEFAULT_CONTROLS = ("on_play", "num_mulligans", "user_game_win_rate_bucket")


def preprocess_game_set(
    csv_path: str | pathlib.Path,
    manifest_path: str | pathlib.Path,
    out_npz: str | pathlib.Path | None = None,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    limit_games: int | None = None,
    batch_size: int = 20_000,
) -> dict:
    """Build the per-game (X, y, controls) arrays aligned to a manifest. Returns a dict of arrays.

    `X` columns are in manifest-card-index order (length = manifest `n_cards`); cards present in the
    manifest but absent from the game CSV stay all-zero. If `out_npz` is given the arrays are cached
    there (and reloaded on a later call, skipping the CSV stream).
    """
    csv_path = pathlib.Path(csv_path)
    manifest_path = pathlib.Path(manifest_path)
    if out_npz is not None:
        out_npz = pathlib.Path(out_npz)
        if out_npz.exists():
            return _load_npz(out_npz)

    manifest = json.load(open(manifest_path))
    cards = manifest["cards"]
    name_to_idx = {c["name"]: c["index"] for c in cards}
    n_cards = len(cards)

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    deck_cols = [c for c in header if c.startswith(DECK_PREFIX)]
    # deck column -> global manifest index (drop cards not in the manifest vocab, e.g. basic lands)
    col_to_global = {c: name_to_idx[c[len(DECK_PREFIX):]] for c in deck_cols
                     if c[len(DECK_PREFIX):] in name_to_idx}
    kept_deck_cols = list(col_to_global)
    global_idx = np.array([col_to_global[c] for c in kept_deck_cols], dtype=np.int64)
    missing_controls = [c for c in controls if c not in header]
    if missing_controls:
        raise ValueError(f"control column(s) absent from {csv_path.name}: {missing_controls}")

    usecols = [LABEL_COL, *controls, *kept_deck_cols]
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    n_games = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=batch_size):
        if limit_games is not None and n_games >= limit_games:
            break
        if limit_games is not None:
            chunk = chunk.iloc[: limit_games - n_games]
        deck = chunk[kept_deck_cols].to_numpy(dtype=np.float32, na_value=0.0)
        x = np.zeros((len(chunk), n_cards), dtype=np.float32)
        x[:, global_idx] = deck
        X_parts.append(x)
        y_parts.append(chunk[LABEL_COL].to_numpy(dtype=np.float32))
        c_parts.append(chunk[list(controls)].to_numpy(dtype=np.float32))
        n_games += len(chunk)

    X = np.concatenate(X_parts) if X_parts else np.zeros((0, n_cards), np.float32)
    y = np.concatenate(y_parts) if y_parts else np.zeros((0,), np.float32)
    C = np.concatenate(c_parts) if c_parts else np.zeros((0, len(controls)), np.float32)
    # impute the rare missing control (e.g. a null user_game_win_rate_bucket) with its column mean
    # so a couple of NaNs don't poison the standardized fit.
    if C.size:
        col_mean = np.nanmean(C, axis=0)
        bad = np.isnan(C)
        if bad.any():
            C[bad] = np.take(col_mean, np.nonzero(bad)[1])
    card_names = [c["name"] for c in sorted(cards, key=lambda d: d["index"])]
    out = {
        "X": X, "y": y, "C": C,
        "card_names": np.array(card_names, dtype=object),
        "control_names": np.array(list(controls), dtype=object),
        "n_games": np.int64(n_games),
        "n_cards_present": np.int64(len(kept_deck_cols)),
    }
    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **out)
    return out


def _load_npz(path: pathlib.Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
