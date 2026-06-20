"""New-set generalization benchmark (the headline Phase-1 metric).

See docs/roadmap.md Phase 1; research/findings/data-and-eval.md; design-decisions.md DD-001.

Train the content model on set A, then swap in set B's content matrix (`model.set_content`) and
evaluate on B's picks — cards the model never saw. We report:

- overall top-1 / MTPD on B (the generalization number),
- top-1 restricted to picks whose chosen card is NOVEL (oracle_id absent from A),
- the random-pick floor (mean 1/pack_size) for context, vs the ~22% floor / ~55% published bar.

A fixed-vocabulary baseline (one-hot) cannot be evaluated on B at all — it has no parameters for
B's cards — which is the whole point of the content encoder.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, RemappedDataset, collate_picks
from ..eval.metrics import PickEvaluator
from ..models.draft_model import ContentDraftModel
from ..training.train import draft_level_split, pick_device
from ..training.train_content import fit, train_loop


def _card_keys(manifest_cards: list[dict]) -> list[str | None]:
    """Stable identity per card index: oracle_id if present, else the name (lowercased)."""
    return [c.get("oracle_id") or (c.get("name") or "").lower() or None for c in manifest_cards]


def novel_card_mask(train_manifest: str, test_manifest: str) -> torch.Tensor:
    """Bool tensor over the test vocab: True where the card is absent from the train set."""
    train = json.load(open(train_manifest))["cards"]
    test = json.load(open(test_manifest))["cards"]
    seen = {k for k in _card_keys(train) if k is not None}
    keys = _card_keys(test)
    return torch.tensor([(k is None) or (k not in seen) for k in keys], dtype=torch.bool)


@torch.no_grad()
def evaluate_on_set(model, parquet: str, device, novel_card: torch.Tensor | None = None,
                    batch_size: int = 512) -> dict:
    """Evaluate `model` (with its current content table) on a set's picks."""
    model.eval()
    ds = DraftPickDataset(parquet)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_picks)
    ev = PickEvaluator()
    ev_novel = PickEvaluator()
    floor_sum = 0.0
    n = 0
    if novel_card is not None:
        novel_card = novel_card.to(device)
    for b in dl:
        pack_mask = b["pack_mask"].to(device)
        label = b["label"].to(device)
        logits = model(b["pool"].to(device), b["pool_mask"].to(device),
                       b["pack"].to(device), pack_mask)
        pick_number = b["pick_number"]
        ev.update(logits, label, pack_mask, pick_number)
        sizes = pack_mask.sum(dim=-1).clamp_min(1)
        floor_sum += float((1.0 / sizes).sum())
        n += sizes.shape[0]
        if novel_card is not None:
            sel = novel_card[b["pick_idx"].to(device)]
            if bool(sel.any()):
                ev_novel.update(logits[sel], label[sel], pack_mask[sel], pick_number[sel.cpu()])
    out = ev.compute()
    out["random_floor"] = floor_sum / max(n, 1)
    if novel_card is not None:
        nm = ev_novel.compute()
        out["novel_top1"] = nm["top1"]
        out["novel_mtpd"] = nm["mtpd"]
        out["n_novel"] = nm["n"]
    return out


def novel_mask_for_holdout(holdout_manifest: str, train_keys: set) -> torch.Tensor:
    """Bool tensor over the holdout vocab: True where the card is absent from the training union."""
    cards = json.load(open(holdout_manifest))["cards"]
    keys = _card_keys(cards)
    return torch.tensor([(k is None) or (k not in train_keys) for k in keys], dtype=torch.bool)


def run_loso(
    train_specs: list[dict], holdout_spec: dict,
    embedder: str = "all-MiniLM-L6-v2", text: bool = True,
    emb_dim: int = 256, enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
    pool: str = "mean", n_heads: int = 4, n_sab: int = 1,
    loss: str = "ce", n_negatives: int = 512,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
    device: str = "auto", checkpoint_dir: str = "data/checkpoints", seed: int = 0,
    out_json: str | None = None,
) -> dict:
    """Leave-one-set-out: train on the union of train_specs, evaluate zero-shot on holdout_spec.

    Each spec is {"parquet", "manifest", "scryfall"}.
    """
    emb = get_embedder(embedder)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=text)
    hmat, hinfo = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                       embedder=emb, text=text)
    assert gmat.shape[1] == hmat.shape[1], (gmat.shape, hmat.shape)

    torch.manual_seed(seed)
    dev = pick_device(device)
    print(f"device={dev}  global train matrix {tuple(gmat.shape)} from {ginfo['n_sets']} sets")

    train_subsets, val_subsets = [], []
    for spec, l2g in zip(train_specs, l2gs):
        rds = RemappedDataset(DraftPickDataset(spec["parquet"]), l2g)
        tr, va = draft_level_split(spec["parquet"], val_frac, seed)
        train_subsets.append(Subset(rds, tr))
        val_subsets.append(Subset(rds, va))
    train_dl = DataLoader(ConcatDataset(train_subsets), batch_size=batch_size, shuffle=True,
                          collate_fn=collate_picks)
    val_dl = DataLoader(ConcatDataset(val_subsets), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_picks)
    n_tr = sum(len(s) for s in train_subsets)
    n_va = sum(len(s) for s in val_subsets)
    print(f"picks: {n_tr} train / {n_va} val across {len(train_specs)} sets")

    model = ContentDraftModel(torch.from_numpy(gmat), emb_dim=emb_dim, enc_hidden=enc_hidden,
                              enc_layers=enc_layers, dropout=dropout, pool=pool,
                              n_heads=n_heads, n_sab=n_sab).to(dev)
    best = train_loop(model, train_dl, val_dl, dev, epochs=epochs, lr=lr,
                      checkpoint_dir=checkpoint_dir, checkpoint_every=0,
                      n_cards=ginfo["n_cards"], tag="loso", ckpt_prefix="loso",
                      loss=loss, n_negatives=n_negatives)

    novel = novel_mask_for_holdout(holdout_spec["manifest"], set(key_to_idx))
    frac_novel = float(novel.float().mean())
    model.set_content(torch.from_numpy(hmat))
    cross = evaluate_on_set(model, holdout_spec["parquet"], dev, novel_card=novel)

    results = {
        "mode": "loso", "embedder": embedder, "text": text,
        "pool": pool, "loss": loss,
        "n_train_sets": len(train_specs), "train_cards": ginfo["n_cards"],
        "train_in_set_top1": best["top1"],
        "holdout": {"parquet": holdout_spec["parquet"], "n_cards": hinfo["n_cards"],
                    "frac_cards_novel": frac_novel, **cross},
    }
    _print_loso(results)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _print_loso(r: dict):
    h = r["holdout"]
    print("\n=== Leave-one-set-out report ===")
    print(f"embedder: {r['embedder']}  text={r['text']}  pool={r['pool']}  loss={r['loss']}  "
          f"train_sets={r['n_train_sets']}  train_cards={r['train_cards']}")
    print(f"in-set (train union) val: top1={r['train_in_set_top1']:.4f}")
    print(f"held-out (unseen set)   : top1={h['top1']:.4f}  mtpd={h['mtpd']:.3f}  n={h['n']}")
    print(f"  novel-card picks      : top1={h.get('novel_top1', float('nan')):.4f}  "
          f"n_novel={h.get('n_novel', 0)}  ({h['frac_cards_novel']*100:.0f}% of cards novel)")
    print(f"  random floor          : {h['random_floor']:.4f}   "
          f"(published: ~0.22 chance, ~0.55 pretrained — Bertram et al. 2024)")


def run_experiment(
    train_parquet: str, train_manifest: str, train_scryfall: str,
    test_parquet: str, test_manifest: str, test_scryfall: str,
    embedder: str = "all-MiniLM-L6-v2", out_json: str | None = None, **fit_kwargs,
) -> dict:
    """Train on A, evaluate zero-shot on B. Returns the results dict."""
    emb = get_embedder(embedder)
    A, infoA = build_content_matrix(train_manifest, train_scryfall, embedder=emb)
    B, infoB = build_content_matrix(test_manifest, test_scryfall, embedder=emb)
    assert A.shape[1] == B.shape[1], (A.shape, B.shape)

    model, in_set = fit(train_parquet, train_manifest, A, infoA, **fit_kwargs)
    dev = next(model.parameters()).device

    novel = novel_card_mask(train_manifest, test_manifest)
    frac_novel = float(novel.float().mean())
    model.set_content(torch.from_numpy(B))
    cross = evaluate_on_set(model, test_parquet, dev, novel_card=novel)

    results = {
        "embedder": embedder,
        "train": {"parquet": train_parquet, "n_cards": infoA["n_cards"],
                  "in_set_top1": in_set["top1"], "in_set_mtpd": in_set["mtpd"]},
        "test": {"parquet": test_parquet, "n_cards": infoB["n_cards"],
                 "frac_cards_novel": frac_novel, **cross},
    }
    _print_report(results)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _print_report(r: dict):
    t, te = r["train"], r["test"]
    print("\n=== Generalization report ===")
    print(f"embedder: {r['embedder']}")
    print(f"in-set (train) val: top1={t['in_set_top1']:.4f}  mtpd={t['in_set_mtpd']:.3f}")
    print(f"cross-set (unseen) : top1={te['top1']:.4f}  mtpd={te['mtpd']:.3f}  n={te['n']}")
    print(f"  novel-card picks : top1={te.get('novel_top1', float('nan')):.4f}  "
          f"n_novel={te.get('n_novel', 0)}  ({te['frac_cards_novel']*100:.0f}% of B's cards novel)")
    print(f"  random floor     : {te['random_floor']:.4f}   "
          f"(published bars: ~0.22 chance, ~0.55 pretrained, per Bertram et al. 2024)")


def _spec(triple: str) -> dict:
    """Parse a 'parquet,manifest,scryfall' triple into a spec dict."""
    pq, man, scry = triple.split(",")
    return {"parquet": pq.strip(), "manifest": man.strip(), "scryfall": scry.strip()}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Leave-one-set-out generalization benchmark. Repeat --train for each "
                    "training set; one --holdout. Each value is 'parquet,manifest,scryfall'.")
    ap.add_argument("--train", action="append", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--holdout", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--no-text", action="store_true", help="structured features only (ablation)")
    ap.add_argument("--pool", default="mean", choices=["mean", "set_transformer"])
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-sab", type=int, default=1)
    ap.add_argument("--loss", default="ce", choices=["ce", "infonce"])
    ap.add_argument("--n-negatives", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_loso(
        [_spec(t) for t in a.train], _spec(a.holdout),
        embedder=a.embedder, text=not a.no_text, pool=a.pool, n_heads=a.n_heads, n_sab=a.n_sab,
        loss=a.loss, n_negatives=a.n_negatives, out_json=a.out_json,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
