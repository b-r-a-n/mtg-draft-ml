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
                 "event_match_wins"]


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
            "event_match_wins": self._cols["event_match_wins"][i].as_py(),
        }


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
        wins[i] = b["event_match_wins"]

    return {
        "pack": pack, "pack_mask": pack_mask,
        "pool": pool, "pool_mask": pool_mask,
        "label": label, "pick_idx": pick_idx, "wins": wins,
    }


def pool_multihot(pool_indices, pool_counts, n_cards: int):
    """Fixed-width multi-hot (counts) pool vector for the Phase-0 one-hot baseline."""
    torch = _require_torch()
    v = torch.zeros(n_cards, dtype=torch.float32)
    for idx, cnt in zip(pool_indices, pool_counts):
        v[idx] = float(cnt)
    return v
