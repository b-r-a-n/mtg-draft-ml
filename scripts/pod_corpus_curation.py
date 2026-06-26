"""Corpus curation — separate corpus RELEVANCE from corpus COUNT for WR-agreement.

The scaling curve ([wr-scaling.md]) peaked at ~19 sets and declined at 22, but it used a NESTED-PREFIX
corpus with STX/SIR/PIO pinned last — so "19 vs 22" conflates *which* sets with *how many*. This trains
the same recipe (good players + composite-WR teacher, big net) on EXPLICIT named corpora, holding out
DSK, multi-seed, and reads WR-agreement vs GIH (the metric the 0.29-0.31 ceiling is defined on).

Findings driving the arms (from scryfall set composition):
  - SIR = 2016 Innistrad *remaster* (cards resolve to soi/emn/inr), PIO = Pioneer *Masters* (2012-19
    reprints): genuinely off-distribution draft environments → suspected to hurt.
  - STX = a real 2021 Standard expansion, just older + caught in the same ingest batch → suspected fine.

    uv run python scripts/pod_corpus_curation.py --seeds 0,1,2          # named arms
    uv run python scripts/pod_corpus_curation.py --loo --seeds 0,1      # leave-one-out ranking
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import torch

from mtg_draft_ml.cards.content_table import build_multiset_content
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
GIH = "ever_drawn_win_rate"
TEACHER_FIELDS = ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"]

# the full 22-set train corpus (everything ingested except the DSK holdout)
ALL22 = ["BLB", "OTJ", "WOE", "MKM", "LCI", "MH3", "MOM", "FDN", "DFT", "TDM", "FIN", "EOE",
         "ONE", "BRO", "DMU", "SNC", "NEO", "MID", "LTR", "STX", "SIR", "PIO"]
REPRINT = ["SIR", "PIO"]          # off-distribution remaster/Masters (suspected to hurt)
NESTED_DROP = ["STX", "SIR", "PIO"]  # what the nested-prefix curve dropped to reach its 19-set peak
# count-vs-relevance control: a 19-set corpus that KEEPS the off-distribution sets but drops 3 strong
# recent expansions instead. If this < nested19, the lever is WHICH sets (relevance), not the count 19.
CTRL_DROP = ["FIN", "EOE", "TDM"]


def _arms():
    """name -> explicit train-set list."""
    a = {
        "all22": ALL22,
        "nested19": [s for s in ALL22 if s not in NESTED_DROP],
        "drop_reprint20": [s for s in ALL22 if s not in REPRINT],          # keep STX
        "relevance_ctrl19": [s for s in ALL22 if s not in CTRL_DROP],      # keep SIR/PIO, drop 3 recent
    }
    return a


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
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--loo", action="store_true", help="leave-one-out over ALL22 instead of named arms")
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
    seeds = [int(x) for x in a.seeds.split(",")]
    skill = {"min_winrate": a.min_winrate, "min_games": a.min_games, "ranks": None}
    emb = get_embedder("all-MiniLM-L6-v2")
    dev = pick_device(a.device)

    if a.loo:
        base = [s for s in ALL22 if s != holdout]
        arms = {"all": base, **{f"drop_{s}": [x for x in base if x != s] for s in base}}
    else:
        arms = {k: [s for s in v if s != holdout] for k, v in _arms().items()}
    out_json = a.out_json or f"data/corpus_curation_{holdout}{'_loo' if a.loo else ''}.json"

    hold = spec(holdout, merged_dir)
    from mtg_draft_ml.cards.content_table import build_content_matrix
    hmat, _ = build_content_matrix(hold["manifest"], hold["scryfall"], embedder=emb, text=True)
    wr_gih = align_winrates(hold["manifest"], hold["ratings"], field=GIH)
    hp = dict(emb_dim=a.emb_dim, enc_hidden=a.enc_hidden, enc_layers=a.enc_layers, dropout=0.1,
              pool="set_transformer", n_heads=4, n_sab=1)
    common = dict(checkpoint_dir="data/checkpoints", checkpoint_every=0, epochs=a.epochs, lr=1e-3,
                  warmup_frac=0.1, grad_clip=1.0)

    print(f"### HOLDOUT {holdout} · {len(arms)} arms · seeds={seeds}", flush=True)
    results = []
    for name, train_sets in arms.items():
        print(f"\n############ ARM {name} ({len(train_sets)} sets): {train_sets} ############", flush=True)
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
            print(f"-- {name} seed {seed} --", flush=True)
            torch.manual_seed(seed)
            tdl, vdl = _loaders(bases, 0.05, split_seed=seed, batch_size=512, skill=skill)
            m = _build_model(gmat, dev, **hp)
            train_loop(m, tdl, vdl, dev, tag=f"{name}s{seed}", ckpt_prefix=f"{name}s{seed}",
                       n_cards=ginfo["n_cards"], teacher=teacher, distill_lambda=0.5,
                       distill_temp=1.0, **common)
            m.set_content(torch.from_numpy(hmat))
            g = evaluate_on_set(m, hold["parquet"], dev, novel_card=novel, card_wr=wr_gih)
            seed_rows.append({"seed": seed, "top1": g["top1"], "wr_gih": g.get("wr_agreement_model"),
                              "human_gih": g.get("wr_agreement_human")})
        results.append({"arm": name, "n_sets": len(train_sets), "train_sets": train_sets,
                        "n_global_cards": ginfo["n_cards"], "seeds": seed_rows})

    def agg(rows, k):
        vs = [r[k] for r in rows if r[k] is not None]
        return (statistics.mean(vs), statistics.pstdev(vs) if len(vs) > 1 else 0.0)

    print(f"\n=== corpus curation (holdout {holdout}; good players + composite, big net) ===")
    print(f"  {'arm':<18}{'sets':>5}{'cards':>7}{'top1':>9}{'WR:GIH':>16}")
    for c in sorted(results, key=lambda r: -agg(r["seeds"], "wr_gih")[0]):
        t1 = agg(c["seeds"], "top1"); g = agg(c["seeds"], "wr_gih")
        print(f"  {c['arm']:<18}{c['n_sets']:>5}{c['n_global_cards']:>7}{t1[0]:>9.4f}{g[0]:>10.4f}±{g[1]:.4f}")
    hum = agg(results[0]["seeds"], "human_gih")
    print(f"  human WR-agree:GIH = {hum[0]:.4f}")
    if a.loo:
        base_g = next((agg(c["seeds"], "wr_gih")[0] for c in results if c["arm"] == "all"), None)
        if base_g is not None:
            print("\n  per-set contribution (Δ = WR:GIH[without set] − WR:GIH[all]):")
            print("    Δ>0 ⇒ removing it HELPED ⇒ the set HURTS (drop); Δ<0 ⇒ the set HELPS (keep)")
            for c in sorted(results, key=lambda r: -(agg(r["seeds"], "wr_gih")[0] - base_g)):
                if c["arm"] == "all":
                    continue
                d = agg(c["seeds"], "wr_gih")[0] - base_g
                print(f"    {c['arm'][5:]:<5} Δ={d:+.4f}  ({'HURTS→drop' if d > 0.002 else 'HELPS→keep' if d < -0.002 else 'flat'})")

    out = pathlib.Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"holdout": holdout, "teacher_fields": TEACHER_FIELDS,
                   "net": f"emb{a.emb_dim}h{a.enc_hidden}L{a.enc_layers}", "loo": a.loo,
                   "results": results}, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
