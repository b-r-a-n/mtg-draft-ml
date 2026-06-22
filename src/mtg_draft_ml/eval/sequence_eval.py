"""Leave-one-set-out for the Phase-5 sequence model (does signal-reading beat the ~0.58 ceiling?).

See docs/phase5-sequence-modeling.md. Mirrors eval.generalization.run_loso but with the sequence
model + sequence datasets. Per-step masked cross-entropy; held-out evaluated by per-step top-1
(= pick accuracy) on the unseen set's draft sequences via set_content swap.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.sequence import RemappedSequenceDataset, SequenceDraftDataset, collate_sequences
from ..training.train import pick_device


@torch.no_grad()
def _eval_topk(model, parquet, dev, batch_size=16):
    model.eval()
    dl = DataLoader(SequenceDraftDataset(parquet), batch_size=batch_size, shuffle=False,
                    collate_fn=collate_sequences)
    correct = total = 0
    per_pos_c, per_pos_t = {}, {}
    for b in dl:
        logits = model(b["pack"].to(dev), b["pack_mask"].to(dev),
                       b["step_mask"].to(dev), b["pick_idx"].to(dev))
        pred = logits.argmax(-1).cpu()
        sm = b["step_mask"]
        hit = (pred == b["label"]) & sm
        correct += int(hit.sum()); total += int(sm.sum())
        # per pick-position (step index within draft) accuracy
        for t in range(sm.shape[1]):
            m = sm[:, t]
            if bool(m.any()):
                per_pos_c[t] = per_pos_c.get(t, 0) + int(((pred[:, t] == b["label"][:, t]) & m).sum())
                per_pos_t[t] = per_pos_t.get(t, 0) + int(m.sum())
    return {"top1": correct / max(total, 1), "n": total,
            "acc_by_pos": {t: per_pos_c[t] / per_pos_t[t] for t in sorted(per_pos_c)}}


def run_loso_sequence(train_specs, holdout_spec, embedder="all-MiniLM-L6-v2",
                      emb_dim=256, enc_hidden=512, enc_layers=3, n_layers=2, n_heads=4,
                      epochs=10, lr=1e-3, batch_size=16, warmup_frac=0.1, grad_clip=1.0,
                      device="auto", seed=0, out_json=None):
    from ..models.sequence_model import SequenceDraftModel

    torch.manual_seed(seed)
    dev = pick_device(device)
    emb = get_embedder(embedder)
    gmat, ginfo, k2i, l2gs = build_multiset_content(train_specs, embedder=emb, text=True)
    hmat, hinfo = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                       embedder=emb, text=True)
    assert gmat.shape[1] == hmat.shape[1]
    print(f"device={dev} global matrix {tuple(gmat.shape)} from {ginfo['n_sets']} sets")

    subsets = [RemappedSequenceDataset(SequenceDraftDataset(s["parquet"]), l2g)
               for s, l2g in zip(train_specs, l2gs)]
    dl = DataLoader(ConcatDataset(subsets), batch_size=batch_size, shuffle=True,
                    collate_fn=collate_sequences)
    n_drafts = sum(len(s) for s in subsets)
    print(f"train drafts: {n_drafts}")

    model = SequenceDraftModel(torch.from_numpy(gmat), emb_dim=emb_dim, enc_hidden=enc_hidden,
                               enc_layers=enc_layers, n_layers=n_layers, n_heads=n_heads).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = max(1, epochs * max(1, len(dl)))
    warmup_steps = int(warmup_frac * total_steps)
    step = 0
    for epoch in range(epochs):
        model.train()
        run = seen = 0.0
        for b in dl:
            if warmup_steps and step < warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = lr * (step + 1) / warmup_steps
            logits = model(b["pack"].to(dev), b["pack_mask"].to(dev),
                           b["step_mask"].to(dev), b["pick_idx"].to(dev))
            sm = b["step_mask"].to(dev)
            loss = torch.nn.functional.cross_entropy(logits[sm], b["label"].to(dev)[sm])
            opt.zero_grad(); loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); step += 1
            run += loss.item() * int(sm.sum()); seen += int(sm.sum())
        # quick in-set proxy on the last train batch loader is skipped; report train loss
        print(f"epoch {epoch}: train_loss={run / max(seen, 1):.4f}", flush=True)

    model.set_content(torch.from_numpy(hmat))
    held = _eval_topk(model, holdout_spec["parquet"], dev)
    mid = [v for p, v in held["acc_by_pos"].items() if 3 <= p <= 9]
    results = {"mode": "loso_sequence", "embedder": embedder, "n_train_sets": len(train_specs),
               "train_cards": ginfo["n_cards"], "holdout_top1": held["top1"], "n": held["n"],
               "midpack_top1": (sum(mid) / len(mid)) if mid else None}
    print(f"\n=== SEQUENCE LOSO: held-out top1={held['top1']:.4f} (n={held['n']}) "
          f"midpack={results['midpack_top1']:.4f} ===" if mid else
          f"\n=== SEQUENCE LOSO: held-out top1={held['top1']:.4f} ===")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
    return results
