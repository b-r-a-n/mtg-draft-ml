"""Top-k pick accuracy (is the human pick among the model's top-k?) overall + by region.

Top-1 is bounded by human disagreement (~0.58); top-k asks whether the model rates the human pick as
reasonable even when it ranks another reasonable card first. In-set BLB model, eval on val drafts.
"""
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from mtg_draft_ml.cards.content_table import build_content_matrix
from mtg_draft_ml.cards.text_embed import get_embedder
from mtg_draft_ml.data.dataset import DraftPickDataset, collate_picks
from mtg_draft_ml.training.train import draft_level_split
from mtg_draft_ml.training.train_content import fit

P = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1proc/hf_push"
SET, SIZE, SEED, VAL = "BLB", "60000", 0, 0.15
pq = f"{P}/draft/{SET}.PremierDraft.sample{SIZE}.parquet"
man = f"{P}/manifests/{SET}.PremierDraft.sample{SIZE}.json"
scry = f"{P}/scryfall/{SET.lower()}.json"

matrix, info = build_content_matrix(man, scry, embedder=get_embedder("all-MiniLM-L6-v2"))
model, best = fit(pq, man, matrix, info, pool="set_transformer", epochs=8,
                  warmup_frac=0.1, grad_clip=1.0, val_frac=VAL, seed=SEED, checkpoint_dir="/tmp/tk_ck")
dev = next(model.parameters()).device
model.eval()

_, val_idx = draft_level_split(pq, VAL, SEED)
dl = DataLoader(Subset(DraftPickDataset(pq), val_idx), batch_size=512, shuffle=False,
                collate_fn=collate_picks)


def region(k):
    return "early(1-3)" if k <= 2 else ("mid(4-10)" if k <= 9 else "late(11+)")


agg = defaultdict(lambda: {"n": 0, "t1": 0, "t3": 0, "t5": 0, "packsum": 0})
with torch.no_grad():
    for b in dl:
        logits = model(b["pool"].to(dev), b["pool_mask"].to(dev), b["pack"].to(dev), b["pack_mask"].to(dev))
        tgt = b["label"].to(dev)
        true_score = logits.gather(1, tgt[:, None]).squeeze(1)
        better = ((logits > true_score[:, None]) & b["pack_mask"].to(dev)).sum(-1).cpu()
        packsize = b["pack_mask"].sum(-1)
        for j, pn in enumerate(b["pick_number"].tolist()):
            for key in (region(pn), "ALL"):
                a = agg[key]; a["n"] += 1; a["packsum"] += int(packsize[j])
                a["t1"] += int(better[j] < 1); a["t3"] += int(better[j] < 3); a["t5"] += int(better[j] < 5)

print(f"\nin-set val top1={best['top1']:.4f}")
print("=== top-k: is the human pick among the model's top-k? ===")
print(f"{'region':<12} {'n':>7} {'avg_pack':>8}  {'top1':>6} {'top3':>6} {'top5':>6}")
for key in ["early(1-3)", "mid(4-10)", "late(11+)", "ALL"]:
    a = agg[key]
    if a["n"]:
        print(f"{key:<12} {a['n']:>7} {a['packsum']/a['n']:>8.1f}  "
              f"{a['t1']/a['n']:>6.3f} {a['t3']/a['n']:>6.3f} {a['t5']/a['n']:>6.3f}")
