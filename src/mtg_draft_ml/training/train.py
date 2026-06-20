"""Phase-0 training loop for the one-hot MLP baseline (MPS-friendly, with checkpointing).

See docs/roadmap.md Phase 0.

Example:
    uv run python -m mtg_draft_ml.training.train \\
        --parquet data/processed/draft/FDN.PremierDraft.parquet \\
        --manifest data/processed/manifests/FDN.PremierDraft.json \\
        --epochs 8 --batch-size 512

Train/val split is by draft_id (no same-draft leakage across the split).
"""
from __future__ import annotations

import argparse
import json
import pathlib
from functools import partial

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from ..data.dataset import DraftPickDataset, collate_onehot
from ..eval.metrics import PickEvaluator
from ..models.baseline import OneHotMLP
from .losses import pick_cross_entropy


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def draft_level_split(parquet: str | pathlib.Path, val_frac: float, seed: int):
    """Return (train_idx, val_idx) row indices, splitting whole drafts into val."""
    draft_ids = pq.read_table(parquet, columns=["draft_id"]).column("draft_id").to_numpy(
        zero_copy_only=False
    )
    uniq = np.unique(draft_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_drafts = set(uniq[:n_val].tolist())
    is_val = np.array([d in val_drafts for d in draft_ids])
    all_idx = np.arange(len(draft_ids))
    return all_idx[~is_val].tolist(), all_idx[is_val].tolist()


def evaluate(model, loader, device) -> dict:
    model.eval()
    ev = PickEvaluator()
    with torch.no_grad():
        for batch in loader:
            pool = batch["pool"].to(device)
            pack = batch["pack"].to(device)
            label = batch["label"].to(device)
            logits = model(pool, pack)
            ev.update(logits, label, pack, batch["pick_number"])
    return ev.compute()


def train(
    parquet: str,
    manifest: str,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden: int = 512,
    layers: int = 3,
    dropout: float = 0.1,
    val_frac: float = 0.1,
    device: str = "auto",
    checkpoint_dir: str = "data/checkpoints",
    checkpoint_every: int = 2000,
    num_workers: int = 0,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    dev = pick_device(device)
    n_cards = json.load(open(manifest))["n_cards"]
    print(f"device={dev}  n_cards={n_cards}")

    ds = DraftPickDataset(parquet)
    train_idx, val_idx = draft_level_split(parquet, val_frac, seed)
    collate = partial(collate_onehot, n_cards=n_cards)
    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
                          collate_fn=collate, num_workers=num_workers, drop_last=False)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers)
    print(f"picks: {len(train_idx)} train / {len(val_idx)} val")

    model = OneHotMLP(n_cards, hidden=hidden, layers=layers, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt_dir = pathlib.Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best = {"top1": -1.0}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_dl:
            pool = batch["pool"].to(dev)
            pack = batch["pack"].to(dev)
            label = batch["label"].to(dev)
            logits = model(pool, pack)
            loss = pick_cross_entropy(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            running += loss.item() * len(label)
            seen += len(label)
            if checkpoint_every and step % checkpoint_every == 0:
                _save(ckpt_dir / "last.pt", model, opt, step, epoch, n_cards, manifest)

        metrics = evaluate(model, val_dl, dev)
        train_loss = running / max(seen, 1)
        mid = _midpack_acc(metrics["acc_by_pick"])
        print(f"epoch {epoch}: train_loss={train_loss:.4f}  "
              f"val_top1={metrics['top1']:.4f}  val_mtpd={metrics['mtpd']:.3f}  "
              f"mid-pack_top1={mid:.4f}")
        _save(ckpt_dir / "last.pt", model, opt, step, epoch, n_cards, manifest)
        if metrics["top1"] > best["top1"]:
            best = metrics
            _save(ckpt_dir / "best.pt", model, opt, step, epoch, n_cards, manifest)

    print(f"best val_top1={best['top1']:.4f}")
    return best


def _midpack_acc(acc_by_pick: dict) -> float:
    """Mean top-1 over picks 3..9 (the hard, synergy-driven mid-pack region)."""
    vals = [v for p, v in acc_by_pick.items() if 3 <= p <= 9]
    return sum(vals) / len(vals) if vals else float("nan")


def _save(path, model, opt, step, epoch, n_cards, manifest):
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
         "epoch": epoch, "n_cards": n_cards, "manifest": str(manifest)},
        path,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-0 baseline training")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    train(a.parquet, a.manifest, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          hidden=a.hidden, layers=a.layers, dropout=a.dropout, val_frac=a.val_frac,
          device=a.device, checkpoint_dir=a.checkpoint_dir, seed=a.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
