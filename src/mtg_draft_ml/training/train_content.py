"""Phase-1 training for the content card-encoder model.

See docs/roadmap.md Phase 1; docs/architecture.md.

`fit()` builds the model from a content matrix and trains it, returning the trained model (used by
the generalization benchmark). `train()` is the CLI path: it resolves the content matrix (prebuilt
.npy or built on the fly from Scryfall) and calls `fit()`.

Example (hashing text embedder, no heavy deps):
    uv run python -m mtg_draft_ml.training.train_content \\
        --parquet data/processed/draft/FDN.PremierDraft.parquet \\
        --manifest data/processed/manifests/FDN.PremierDraft.json \\
        --scryfall data/scryfall/oracle-cards.json --embedder hash --epochs 8

Use --embedder all-MiniLM-L6-v2 (needs the [embeddings] extra) for the real text encoder.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..cards.content_table import build_content_matrix, load_content_matrix
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, collate_picks
from ..eval.metrics import PickEvaluator
from ..models.draft_model import ContentDraftModel
from .losses import pick_cross_entropy
from .train import _midpack_acc, _save, draft_level_split, pick_device


def evaluate(model, loader, device) -> dict:
    model.eval()
    ev = PickEvaluator()
    with torch.no_grad():
        for b in loader:
            logits = model(b["pool"].to(device), b["pool_mask"].to(device),
                           b["pack"].to(device), b["pack_mask"].to(device))
            ev.update(logits, b["label"].to(device), b["pack_mask"].to(device), b["pick_number"])
    return ev.compute()


def fit(
    parquet: str,
    manifest: str,
    matrix: np.ndarray,
    info: dict,
    *,
    emb_dim: int = 256,
    enc_hidden: int = 512,
    enc_layers: int = 3,
    dropout: float = 0.1,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    device: str = "auto",
    checkpoint_dir: str = "data/checkpoints",
    checkpoint_every: int = 2000,
    num_workers: int = 0,
    seed: int = 0,
):
    """Train a ContentDraftModel on `matrix`. Returns (model, best_val_metrics)."""
    torch.manual_seed(seed)
    dev = pick_device(device)
    print(f"device={dev}  content matrix {tuple(matrix.shape)}  {info}")

    ds = DraftPickDataset(parquet)
    train_idx, val_idx = draft_level_split(parquet, val_frac, seed)
    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
                          collate_fn=collate_picks, num_workers=num_workers)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_picks, num_workers=num_workers)
    print(f"picks: {len(train_idx)} train / {len(val_idx)} val")

    model = ContentDraftModel(torch.from_numpy(matrix), emb_dim=emb_dim, enc_hidden=enc_hidden,
                              enc_layers=enc_layers, dropout=dropout).to(dev)
    best = train_loop(model, train_dl, val_dl, dev, epochs=epochs, lr=lr,
                      checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
                      n_cards=info["n_cards"], tag=manifest)
    return model, best


def train_loop(model, train_dl, val_dl, dev, *, epochs, lr, checkpoint_dir, checkpoint_every,
               n_cards, tag, ckpt_prefix="content"):
    """Shared epoch loop used by both single-set fit() and multi-set LOSO. Returns best metrics."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt_dir = pathlib.Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best = {"top1": -1.0}
    for epoch in range(epochs):
        model.train()
        running = seen = 0.0
        for b in train_dl:
            logits = model(b["pool"].to(dev), b["pool_mask"].to(dev),
                           b["pack"].to(dev), b["pack_mask"].to(dev))
            loss = pick_cross_entropy(logits, b["label"].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            running += loss.item() * len(b["label"])
            seen += len(b["label"])
            if checkpoint_every and step % checkpoint_every == 0:
                _save(ckpt_dir / f"{ckpt_prefix}_last.pt", model, opt, step, epoch, n_cards, tag)

        m = evaluate(model, val_dl, dev)
        print(f"epoch {epoch}: train_loss={running / max(seen, 1):.4f}  "
              f"val_top1={m['top1']:.4f}  val_mtpd={m['mtpd']:.3f}  "
              f"mid-pack_top1={_midpack_acc(m['acc_by_pick']):.4f}")
        _save(ckpt_dir / f"{ckpt_prefix}_last.pt", model, opt, step, epoch, n_cards, tag)
        if m["top1"] > best["top1"]:
            best = m
            _save(ckpt_dir / f"{ckpt_prefix}_best.pt", model, opt, step, epoch, n_cards, tag)

    print(f"best val_top1={best['top1']:.4f}")
    return best


def _resolve_matrix(manifest, content, scryfall, embedder):
    if content is not None:
        return load_content_matrix(content)
    if scryfall is not None:
        return build_content_matrix(manifest, scryfall, embedder=get_embedder(embedder))
    raise ValueError("provide content (prebuilt .npy) or scryfall (to build the matrix)")


def train(parquet, manifest, content=None, scryfall=None, embedder="hash", **fit_kwargs) -> dict:
    matrix, info = _resolve_matrix(manifest, content, scryfall, embedder)
    _, best = fit(parquet, manifest, matrix, info, **fit_kwargs)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-1 content-encoder training")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--content", default=None, help="prebuilt content matrix .npy")
    ap.add_argument("--scryfall", default=None, help="Scryfall JSON to build the matrix")
    ap.add_argument("--embedder", default="hash", help="'hash' or a sentence-transformers model")
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--enc-hidden", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    train(a.parquet, a.manifest, content=a.content, scryfall=a.scryfall, embedder=a.embedder,
          emb_dim=a.emb_dim, enc_hidden=a.enc_hidden, enc_layers=a.enc_layers, dropout=a.dropout,
          epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, val_frac=a.val_frac,
          device=a.device, checkpoint_dir=a.checkpoint_dir, seed=a.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
