"""WR-agreement data-scaling curve — does MORE sets help the win-rate axis (not top-1)?

The Phase-4 scaling curve saturated at ~3-4 sets *on top-1*, and we capped the corpus at 7. But the
good-player / deck_value levers showed "grows with scale" up to 7 — never tested beyond. This re-runs
the scaling curve on **WR-agreement** (the axis with headroom), training the established best recipe
(good players + composite-WR teacher, big net) at increasing corpus sizes, holding out DSK, and
reading WR-agreement vs GIH (the metric the 0.29-0.31 ceiling is defined on) and vs deck_value.

Corpus = nested prefixes of a fixed diverse ordering (so size k is a superset of k-1). Multi-seed for
error bars. Needs a GPU pod (up to 15 train sets x big net x seeds).

    uv run python scripts/pod_wr_scaling.py --sizes 4,7,11,15 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import torch

from mtg_draft_ml.cards.content_table import build_content_matrix, build_multiset_content
from mtg_draft_ml.cards.text_embed import get_embedder
from mtg_draft_ml.data.dataset import DraftPickDataset, RemappedDataset
from mtg_draft_ml.distill.ensemble import _build_model, _loaders
from mtg_draft_ml.distill.wr import WRSoftmaxTeacher
from mtg_draft_ml.eval.game_value import merged_ratings_with_value
from mtg_draft_ml.eval.generalization import evaluate_on_set, novel_mask_for_holdout
from mtg_draft_ml.eval.winrate import align_winrates, composite_card_quality
from mtg_draft_ml.training.train import pick_device
from mtg_draft_ml.training.train_content import train_loop

D = "data/hf"
SIZE = "60000"
# fixed diverse priority over all 16 sets; for a given holdout, ORDER = this minus the holdout, and
# corpus size k = the first k. (holdout=DSK reproduces the original curve's [BLB..DMU] ordering.)
PRIORITY = ["BLB", "OTJ", "WOE", "MKM", "LCI", "MH3", "MOM",
            "FDN", "DFT", "TDM", "FIN", "EOE", "ONE", "BRO", "DMU",
            "SNC", "NEO", "MID", "LTR", "STX", "SIR", "PIO", "DSK"]
GIH = "ever_drawn_win_rate"
# the established composite-WR target (good-players.md); hold it FIXED to isolate the data-scaling effect.
TEACHER_FIELDS = ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"]


def spec(s, merged_dir):
    base = {
        "parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
        "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
        "scryfall": f"{D}/scryfall/{s.lower()}.json",
        "ratings_orig": f"{D}/ratings/{s}.PremierDraft.ratings.json",
        "gamevalue": f"{D}/gamevalue/{s}.PremierDraft.gamevalue.json",
    }
    gv = pathlib.Path(base["gamevalue"])
    base["ratings"] = (merged_ratings_with_value(base["ratings_orig"], base["gamevalue"],
                                                 f"{merged_dir}/{s}.merged.json")
                       if gv.exists() else base["ratings_orig"])
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", default="DSK")
    ap.add_argument("--sizes", default="4,7,11,15")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--min-games", type=float, default=50)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--enc-hidden", type=int, default=1024)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)

    merged_dir = "data/ratings_merged"
    pathlib.Path(merged_dir).mkdir(parents=True, exist_ok=True)
    holdout = a.holdout
    order = [s for s in PRIORITY if s != holdout]
    out_json = a.out_json or f"data/wr_scaling_{holdout}.json"
    sizes = [int(x) for x in a.sizes.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    skill = {"min_winrate": a.min_winrate, "min_games": a.min_games, "ranks": None}
    emb = get_embedder("all-MiniLM-L6-v2")
    dev = pick_device(a.device)
    hold = spec(holdout, merged_dir)
    hmat, _ = build_content_matrix(hold["manifest"], hold["scryfall"], embedder=emb, text=True)
    wr_gih = align_winrates(hold["manifest"], hold["ratings"], field=GIH)
    wr_val = align_winrates(hold["manifest"], hold["ratings"], field="deck_value")
    hp = dict(emb_dim=a.emb_dim, enc_hidden=a.enc_hidden, enc_layers=a.enc_layers, dropout=0.1,
              pool="set_transformer", n_heads=4, n_sab=1)
    common = dict(checkpoint_dir="data/checkpoints", checkpoint_every=0, epochs=a.epochs, lr=1e-3,
                  warmup_frac=0.1, grad_clip=1.0)

    print(f"### HOLDOUT {holdout} · order={order}", flush=True)
    curve = []
    for size in sizes:
        train_sets = order[:size]
        print(f"\n############ {holdout} · CORPUS SIZE {size}: {train_sets} ############", flush=True)
        train_specs = [spec(s, merged_dir) for s in train_sets]
        gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=True)
        bases = [(RemappedDataset(DraftPickDataset(s["parquet"]), l2g), s["parquet"])
                 for s, l2g in zip(train_specs, l2gs)]
        rating_specs = [{"manifest": s["manifest"], "ratings": s["ratings"]} for s in train_specs]
        cs, cm = composite_card_quality(rating_specs, l2gs, ginfo["n_cards"], fields=TEACHER_FIELDS)
        teacher = WRSoftmaxTeacher(cs, cm, tau=1.0, device=dev)
        novel = novel_mask_for_holdout(hold["manifest"], set(key_to_idx))

        seed_rows = []
        for seed in seeds:
            print(f"-- size {size} seed {seed} --", flush=True)
            torch.manual_seed(seed)
            tdl, vdl = _loaders(bases, 0.05, split_seed=seed, batch_size=512, skill=skill)
            m = _build_model(gmat, dev, **hp)
            train_loop(m, tdl, vdl, dev, tag=f"sz{size}s{seed}", ckpt_prefix=f"sz{size}s{seed}",
                       n_cards=ginfo["n_cards"], teacher=teacher, distill_lambda=0.5,
                       distill_temp=1.0, **common)
            m.set_content(torch.from_numpy(hmat))
            g = evaluate_on_set(m, hold["parquet"], dev, novel_card=novel, card_wr=wr_gih)
            v = evaluate_on_set(m, hold["parquet"], dev, novel_card=novel, card_wr=wr_val)
            seed_rows.append({"seed": seed, "top1": g["top1"], "wr_gih": g.get("wr_agreement_model"),
                              "wr_value": v.get("wr_agreement_model"),
                              "human_gih": g.get("wr_agreement_human")})
        curve.append({"size": size, "train_sets": train_sets, "n_global_cards": ginfo["n_cards"],
                      "seeds": seed_rows})

    # report
    def agg(rows, k):
        vs = [r[k] for r in rows if r[k] is not None]
        return (statistics.mean(vs), statistics.pstdev(vs) if len(vs) > 1 else 0.0)
    print(f"\n=== WR-agreement scaling curve (holdout {holdout}; good players + composite, big net) ===")
    print(f"  {'sets':>5}{'cards':>7}{'top1':>9}{'WR:GIH':>16}{'WR:deck_value':>16}")
    for c in curve:
        t1 = agg(c["seeds"], "top1"); g = agg(c["seeds"], "wr_gih"); v = agg(c["seeds"], "wr_value")
        print(f"  {c['size']:>5}{c['n_global_cards']:>7}{t1[0]:>9.4f}"
              f"{g[0]:>10.4f}±{g[1]:.4f}{v[0]:>9.4f}±{v[1]:.4f}")
    hum = agg(curve[0]["seeds"], "human_gih")
    print(f"  human WR-agree:GIH = {hum[0]:.4f}")
    print("\n  Δ WR:GIH vs the 7-set point (does scale push past the ceiling?):")
    base7 = next((agg(c["seeds"], "wr_gih")[0] for c in curve if c["size"] == 7), None)
    if base7 is not None:
        for c in curve:
            d = agg(c["seeds"], "wr_gih")[0] - base7
            print(f"    {c['size']:>2} sets: {d:+.4f}")

    out = pathlib.Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"holdout": holdout, "order": order, "teacher_fields": TEACHER_FIELDS,
                   "net": f"emb{a.emb_dim}h{a.enc_hidden}L{a.enc_layers}", "curve": curve}, indent=2,
                   default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
