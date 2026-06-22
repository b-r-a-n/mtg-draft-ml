"""Sequence dataset: group compact-Parquet pick rows into per-draft ordered sequences.

See docs/phase5-sequence-modeling.md.

The per-pick format already stores, for each pick, the pack seen (`pack_indices`) and the card taken
(`pick_idx`/`pick_pos`), grouped by `draft_id`. Ordering a draft's rows by (pack_number, pick_number)
reconstructs the full draft sequence — including the packs you saw and *passed*, which is the
signal-reading information the memoryless model can't see.

A draft -> dict of per-step lists. `collate_sequences` pads to [B, T, P] with step + pack masks.
torch is imported lazily (preprocessing/this grouping needs no torch).
"""
from __future__ import annotations

import pathlib

import pyarrow.parquet as pq

_COLS = ["draft_id", "pack_number", "pick_number", "pack_indices", "pick_pos", "pick_idx",
         "event_match_wins"]


def _require_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("`torch` is required: pip install torch") from e
    return torch


class SequenceDraftDataset:
    """Map-style dataset of whole drafts (each item = one draft's ordered sequence of picks)."""

    def __init__(self, parquet_path: str | pathlib.Path, columns=_COLS):
        self.path = pathlib.Path(parquet_path)
        t = pq.read_table(self.path, columns=columns)
        # group row indices by draft_id, preserving first-seen order
        draft_ids = t.column("draft_id").to_pylist()
        pack_no = t.column("pack_number").to_pylist()
        pick_no = t.column("pick_number").to_pylist()
        groups: dict[str, list[int]] = {}
        for i, d in enumerate(draft_ids):
            groups.setdefault(d, []).append(i)
        # sort each draft's rows by (pack_number, pick_number)
        order = []
        for d, idxs in groups.items():
            idxs.sort(key=lambda i: (pack_no[i], pick_no[i]))
            order.append((d, idxs))
        self._drafts = order
        self._pack = t.column("pack_indices")
        self._pick_pos = t.column("pick_pos")
        self._pick_idx = t.column("pick_idx")
        self._wins = t.column("event_match_wins")

    def __len__(self):
        return len(self._drafts)

    def __getitem__(self, i: int) -> dict:
        _, rows = self._drafts[i]
        return {
            "packs": [self._pack[r].as_py() for r in rows],          # list[T] of list[pack cards]
            "pick_pos": [self._pick_pos[r].as_py() for r in rows],    # list[T]
            "pick_idx": [self._pick_idx[r].as_py() for r in rows],    # list[T]
            "wins": self._wins[rows[0]].as_py(),                      # draft-level
        }


class RemappedSequenceDataset:
    """Wrap a SequenceDraftDataset, mapping set-local card indices to a shared/global vocab.

    Multi-set sequence training analog of data.dataset.RemappedDataset. pick_pos (position within a
    pack) is unaffected by remapping.
    """

    def __init__(self, base: "SequenceDraftDataset", local_to_global):
        self.base = base
        self.l2g = [int(x) for x in local_to_global]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        b = dict(self.base[i])
        g = self.l2g
        b["packs"] = [[g[c] for c in pk] for pk in b["packs"]]
        b["pick_idx"] = [g[x] for x in b["pick_idx"]]
        return b


def collate_sequences(batch: list[dict], pad_value: int = 0):
    """Pad a batch of drafts to [B, T, P]. Returns tensors + masks.

    pack[B,T,P] long (global card indices), pack_mask[B,T,P] bool (real card),
    step_mask[B,T] bool (real pick step), label[B,T] long (pick position within the step's pack),
    pick_idx[B,T] long (global index taken at each step), wins[B] long.
    """
    torch = _require_torch()
    B = len(batch)
    T = max(len(b["packs"]) for b in batch)
    P = max((len(p) for b in batch for p in b["packs"]), default=1)
    P = max(P, 1)

    pack = torch.full((B, T, P), pad_value, dtype=torch.long)
    pack_mask = torch.zeros((B, T, P), dtype=torch.bool)
    step_mask = torch.zeros((B, T), dtype=torch.bool)
    label = torch.zeros((B, T), dtype=torch.long)
    pick_idx = torch.zeros((B, T), dtype=torch.long)
    wins = torch.empty(B, dtype=torch.long)

    for i, b in enumerate(batch):
        wins[i] = b["wins"]
        for t, pk in enumerate(b["packs"]):
            step_mask[i, t] = True
            label[i, t] = b["pick_pos"][t]
            pick_idx[i, t] = b["pick_idx"][t]
            if pk:
                pack[i, t, : len(pk)] = torch.tensor(pk, dtype=torch.long)
                pack_mask[i, t, : len(pk)] = True
    return {"pack": pack, "pack_mask": pack_mask, "step_mask": step_mask,
            "label": label, "pick_idx": pick_idx, "wins": wins}
