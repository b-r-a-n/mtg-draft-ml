"""Train a deployable model, save a Drafter bundle, and demo the aggressiveness dial.

Produces data/bundles/<SET>/ (model.pt, content.npy, cards.json, quality.npy) loadable by
mtg_draft_ml.deploy.Drafter. Quality = 17lands IWD (the dial signal). Run:
    python scripts/package_model.py [data_dir] [SET]
"""
import json
import sys

from mtg_draft_ml.cards.content_table import build_content_matrix
from mtg_draft_ml.cards.text_embed import get_embedder
from mtg_draft_ml.deploy import Drafter
from mtg_draft_ml.eval.winrate import align_winrates
from mtg_draft_ml.training.train_content import fit

P = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1proc/hf_push"
SET = sys.argv[2] if len(sys.argv) > 2 else "BLB"
SIZE = "60000"
pq = f"{P}/draft/{SET}.PremierDraft.sample{SIZE}.parquet"
man = f"{P}/manifests/{SET}.PremierDraft.sample{SIZE}.json"
scry = f"{P}/scryfall/{SET.lower()}.json"
ratings = f"{P}/ratings/{SET}.PremierDraft.ratings.json"

CFG = {"emb_dim": 256, "enc_hidden": 512, "enc_layers": 3, "pool": "set_transformer",
       "n_heads": 4, "n_sab": 1, "aux_wr": False}

matrix, info = build_content_matrix(man, scry, embedder=get_embedder("all-MiniLM-L6-v2"))
model, best = fit(pq, man, matrix, info, pool="set_transformer", emb_dim=CFG["emb_dim"],
                  enc_hidden=CFG["enc_hidden"], enc_layers=CFG["enc_layers"], n_heads=CFG["n_heads"],
                  n_sab=CFG["n_sab"], epochs=8, warmup_frac=0.1, grad_clip=1.0, val_frac=0.1,
                  checkpoint_dir="/tmp/pkg_ck")
print(f"trained: in-set val top1={best['top1']:.4f}")

names = [c["name"] for c in json.load(open(man))["cards"]]
quality = align_winrates(man, ratings, field="drawn_improvement_win_rate")  # IWD = the dial signal
drafter = Drafter(model, names, quality, CFG)
bundle = drafter.save(f"data/bundles/{SET}")
print(f"saved bundle -> {bundle}")

# Demo: reload and pick, showing the aggressiveness dial shift a real mid-pack decision.
d = Drafter.load(bundle)
# pick a real draft context: a mid-pack pick from the data
import pyarrow.parquet as pq_
t = pq_.read_table(pq, columns=["pick_number", "pack_indices", "pool_indices"])
row = next(i for i in range(t.num_rows) if 4 <= t.column("pick_number")[i].as_py() <= 7
           and len(t.column("pack_indices")[i].as_py()) >= 5
           and len(t.column("pool_indices")[i].as_py()) >= 4)
pool = [names[j] for j in t.column("pool_indices")[row].as_py()]
pack = [names[j] for j in t.column("pack_indices")[row].as_py()]
print(f"\nDemo pick — pool: {pool[:6]}{'...' if len(pool) > 6 else ''}")
print(f"            pack: {pack}")
for agg in (0.0, 3.0):
    top = d.pick(pool, pack, aggressiveness=agg, top_k=3)
    print(f"  aggressiveness={agg}: " + ", ".join(f"{r['card']}({r['prob']:.2f})" for r in top))
