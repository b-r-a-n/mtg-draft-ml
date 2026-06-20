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
from torch.utils.data import DataLoader

from ..cards.content_table import build_content_matrix
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, collate_picks
from ..eval.metrics import PickEvaluator
from ..training.train_content import fit


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


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="New-set generalization benchmark (train A -> test B)")
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--train-scryfall", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-scryfall", required=True)
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_experiment(
        a.train_parquet, a.train_manifest, a.train_scryfall,
        a.test_parquet, a.test_manifest, a.test_scryfall,
        embedder=a.embedder, out_json=a.out_json,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
