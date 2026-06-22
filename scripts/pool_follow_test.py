"""Does the trained model exploit pool context as well as humans?

Humans take the pool-favored card 89.3% of the time on mid-pack cross-color decisions where the pool
clearly leans (docs/results/human-disagreement-probe.md). This computes the SAME metric with the
model's pick swapped in for the human's, on held-out drafts — directly comparable.

Trains an in-set BLB model (set_transformer + aux-WR), evaluates pool-follow on val drafts.
Run: python scripts/pool_follow_test.py [data_dir]
"""
import json
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

# card index -> color set
manifest = json.load(open(man))
scryd = {c["name"]: c for c in json.load(open(scry))}
idx_colors = {}
for c in manifest["cards"]:
    rec = scryd.get(c["name"]) or (scryd.get(c["name"].split(" // ")[0]) if " // " in c["name"] else None)
    idx_colors[c["index"]] = set(rec.get("colors", [])) if rec else set()

# train an in-set model
matrix, info = build_content_matrix(man, scry, embedder=get_embedder("all-MiniLM-L6-v2"))
model, best = fit(pq, man, matrix, info, pool="set_transformer",
                  epochs=8, warmup_frac=0.1, grad_clip=1.0, val_frac=VAL, seed=SEED,
                  checkpoint_dir="/tmp/pf_ck")
dev = next(model.parameters()).device
model.eval()
print(f"trained: in-set val top1={best['top1']:.4f}")

# predict model argmax (global card idx) for each VAL row, in order
_, val_idx = draft_level_split(pq, VAL, SEED)
ds = DraftPickDataset(pq)
val = Subset(ds, val_idx)
dl = DataLoader(val, batch_size=512, shuffle=False, collate_fn=collate_picks)
model_pick = []
with torch.no_grad():
    for b in dl:
        logits = model(b["pool"].to(dev), b["pool_mask"].to(dev), b["pack"].to(dev), b["pack_mask"].to(dev))
        pos = logits.argmax(-1).cpu()
        model_pick.append(b["pack"].gather(1, pos[:, None]).squeeze(1))
model_pick = torch.cat(model_pick).tolist()


def pool_follow(choice_of_row):
    """choice_of_row(j) -> the global card idx 'chosen' for val row j. Returns pool-follow rate."""
    follow = against = 0
    for j, ridx in enumerate(val_idx):
        item = ds[ridx]
        if not (4 <= item["pick_number"] <= 9):
            continue
        pool, pc = item["pool_indices"], item["pool_counts"]
        if sum(pc) < 4:
            continue
        cc = defaultdict(int)
        for ci, cnt in zip(pool, pc):
            for col in idx_colors.get(ci, ()):
                cc[col] += cnt
        if not cc:
            continue
        c = choice_of_row(j, item)
        ccol = idx_colors.get(c, set())
        if not ccol:
            continue
        c_aff = sum(cc[x] for x in ccol)
        for o in item["pack_indices"]:
            if o == c:
                continue
            oc = idx_colors.get(o, set())
            if not oc or oc == ccol:
                continue
            o_aff = sum(cc[x] for x in oc)
            if abs(c_aff - o_aff) < 3:
                continue
            if c_aff > o_aff:
                follow += 1
            else:
                against += 1
    n = follow + against
    return follow / n if n else float("nan"), n


human_rate, n_h = pool_follow(lambda j, item: item["pick_idx"])
model_rate, n_m = pool_follow(lambda j, item: model_pick[j])
print("\n=== POOL-FOLLOW TEST (val drafts, mid-pack cross-color, pool clearly leans) ===")
print(f"  human pool-follow: {human_rate:.3f}  (n={n_h})")
print(f"  model pool-follow: {model_rate:.3f}  (n={n_m})")
print(f"  gap (human - model): {human_rate - model_rate:+.3f}")
print("  -> model >= human: it exploits pool context as well as humans (0.58 ~ real ceiling)")
print("  -> model <  human: headroom in pool-conditioning (a NEW lever vs data/capacity/sequence)")
