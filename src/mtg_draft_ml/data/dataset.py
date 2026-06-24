"""Streaming Dataset / collate over compact Parquet (flat RAM, any dataset size).

See docs/architecture.md; design-decisions.md DD-006.

Yields per-pick examples; `collate_picks` pads the variable-size pack/pool to the batch max
and builds masks for the masked-softmax / pointer head (DD-002). `pool_multihot` builds the
fixed-width pool vector used by the Phase-0 one-hot baseline.

torch is imported lazily so this module (and preprocessing) can be used without torch installed.
"""
from __future__ import annotations

import pathlib

import pyarrow.parquet as pq

_PICK_COLUMNS = ["pack_indices", "pool_indices", "pool_counts", "pick_pos", "pick_idx",
                 "pick_number", "pack_number", "event_match_wins"]


def _require_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("`torch` is required for the dataset: pip install torch") from e
    return torch


class DraftPickDataset:
    """Map-style Dataset over a compact-Parquet shard.

    For Phase-0 / single-set work the shard is read into Arrow memory once (a few hundred MB
    at most). For the full corpus, instantiate one Dataset per shard and chain with a sampler,
    or switch to a pyarrow.dataset streaming scan (see docs/data-infra.md).
    """

    def __init__(self, parquet_path: str | pathlib.Path, columns=_PICK_COLUMNS):
        self.path = pathlib.Path(parquet_path)
        self._table = pq.read_table(self.path, columns=columns)
        self._cols = {c: self._table.column(c) for c in columns}
        self.n = self._table.num_rows

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {
            "pack_indices": self._cols["pack_indices"][i].as_py(),
            "pool_indices": self._cols["pool_indices"][i].as_py(),
            "pool_counts": self._cols["pool_counts"][i].as_py(),
            "pick_pos": self._cols["pick_pos"][i].as_py(),
            "pick_idx": self._cols["pick_idx"][i].as_py(),
            "pick_number": self._cols["pick_number"][i].as_py(),
            "pack_number": self._cols["pack_number"][i].as_py(),
            "event_match_wins": self._cols["event_match_wins"][i].as_py(),
        }


class RemappedDataset:
    """Wrap a DraftPickDataset, remapping set-local card indices to a shared/global vocab.

    Used for multi-set training (leave-one-set-out): each set's compact Parquet stores its own
    local indices; `local_to_global[i]` maps set-local index i to the shared content-matrix row.
    pick_pos (position within the pack) is unaffected by remapping.
    """

    def __init__(self, base: "DraftPickDataset", local_to_global):
        self.base = base
        self.l2g = [int(x) for x in local_to_global]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        b = dict(self.base[i])
        g = self.l2g
        b["pack_indices"] = [g[x] for x in b["pack_indices"]]
        b["pool_indices"] = [g[x] for x in b["pool_indices"]]
        b["pick_idx"] = g[b["pick_idx"]]
        return b


def collate_picks(batch: list[dict], pad_value: int = 0):
    """Pad packs/pools to the batch max and build masks.

    Returns a dict of tensors:
      pack[B, P] (int64), pack_mask[B, P] (bool, True = real card),
      pool[B, L] (int64), pool_mask[B, L] (bool),
      label[B] (int64, position of the pick within pack),
      pick_idx[B] (int64, global card index), wins[B] (int64).
    Padded pack logits should be set to -inf before softmax using ~pack_mask.
    """
    torch = _require_torch()
    B = len(batch)
    P = max(len(b["pack_indices"]) for b in batch)
    L = max((len(b["pool_indices"]) for b in batch), default=0)
    L = max(L, 1)

    pack = torch.full((B, P), pad_value, dtype=torch.long)
    pack_mask = torch.zeros((B, P), dtype=torch.bool)
    pool = torch.full((B, L), pad_value, dtype=torch.long)
    pool_mask = torch.zeros((B, L), dtype=torch.bool)
    label = torch.empty(B, dtype=torch.long)
    pick_idx = torch.empty(B, dtype=torch.long)
    pick_number = torch.empty(B, dtype=torch.long)
    pack_number = torch.empty(B, dtype=torch.long)
    wins = torch.empty(B, dtype=torch.long)

    for i, b in enumerate(batch):
        pk = b["pack_indices"]
        pack[i, : len(pk)] = torch.tensor(pk, dtype=torch.long)
        pack_mask[i, : len(pk)] = True
        pl = b["pool_indices"]
        if pl:
            pool[i, : len(pl)] = torch.tensor(pl, dtype=torch.long)
            pool_mask[i, : len(pl)] = True
        label[i] = b["pick_pos"]
        pick_idx[i] = b["pick_idx"]
        pick_number[i] = b["pick_number"]
        pack_number[i] = b["pack_number"]
        wins[i] = b["event_match_wins"]

    return {
        "pack": pack, "pack_mask": pack_mask,
        "pool": pool, "pool_mask": pool_mask,
        "label": label, "pick_idx": pick_idx,
        "pick_number": pick_number, "pack_number": pack_number, "wins": wins,
    }


def skill_filter_indices(parquet_path, min_winrate=None, min_games=None, ranks=None) -> list[int]:
    """Row indices whose drafter passes a skill bar (the "good players" filter).

    Reads the per-pick drafter-skill columns 17lands ships and `preprocess` carries through:
      - user_game_win_rate_bucket >= `min_winrate`  (the player's lifetime game win rate)
      - user_n_games_bucket        >= `min_games`    (confidence — enough games to trust the estimate)
      - rank in `ranks`                              (e.g. {"mythic","diamond","platinum"})
    A criterion is skipped if its column is absent OR entirely null in the shard (so a set lacking
    skill data isn't silently starved); where the column *has* data, an individual null row fails the
    bar (unknown skill can't be confirmed good). Callers should check the kept fraction to confirm
    the filter fired. None for every criterion = keep all.
    """
    avail = set(pq.read_schema(parquet_path).names)
    want = [c for c in ("user_game_win_rate_bucket", "user_n_games_bucket", "rank") if c in avail]
    t = pq.read_table(parquet_path, columns=want) if want else None
    n = t.num_rows if t is not None else pq.read_metadata(parquet_path).num_rows

    def col(name):
        return t.column(name).to_pylist() if (t is not None and name in want) else [None] * n

    wr, ng, rk = col("user_game_win_rate_bucket"), col("user_n_games_bucket"), col("rank")
    wr_on = min_winrate is not None and any(x is not None for x in wr)
    ng_on = min_games is not None and any(x is not None for x in ng)
    rk_on = ranks is not None and any(x is not None for x in rk)
    keep = []
    for i in range(n):
        if wr_on and (wr[i] is None or wr[i] < min_winrate):
            continue
        if ng_on and (ng[i] is None or ng[i] < min_games):
            continue
        if rk_on and (rk[i] is None or rk[i] not in ranks):
            continue
        keep.append(i)
    return keep


def pool_multihot(pool_indices, pool_counts, n_cards: int):
    """Fixed-width multi-hot (counts) pool vector for the Phase-0 one-hot baseline."""
    torch = _require_torch()
    v = torch.zeros(n_cards, dtype=torch.float32)
    for idx, cnt in zip(pool_indices, pool_counts):
        v[idx] = float(cnt)
    return v


def collate_onehot(batch: list[dict], n_cards: int):
    """Collate for the Phase-0 one-hot MLP baseline (full-vocab tensors).

    Returns:
      pool[B, n_cards] float (collection counts),
      pack[B, n_cards] bool (cards available this pick),
      label[B] long (global index of the picked card),
      pick_number[B], pack_number[B] long, wins[B] long.
    Use with functools.partial(collate_onehot, n_cards=N) as a DataLoader collate_fn.
    """
    torch = _require_torch()
    B = len(batch)
    pool = torch.zeros((B, n_cards), dtype=torch.float32)
    pack = torch.zeros((B, n_cards), dtype=torch.bool)
    label = torch.empty(B, dtype=torch.long)
    pick_number = torch.empty(B, dtype=torch.long)
    pack_number = torch.empty(B, dtype=torch.long)
    wins = torch.empty(B, dtype=torch.long)
    for i, b in enumerate(batch):
        for idx, cnt in zip(b["pool_indices"], b["pool_counts"]):
            pool[i, idx] = float(cnt)
        if b["pack_indices"]:
            pack[i, b["pack_indices"]] = True
        label[i] = b["pick_idx"]
        pick_number[i] = b["pick_number"]
        pack_number[i] = b["pack_number"]
        wins[i] = b["event_match_wins"]
    return {"pool": pool, "pack": pack, "label": label,
            "pick_number": pick_number, "pack_number": pack_number, "wins": wins}
