"""Step 2 — does the de-confounded game_data target beat the GIH-WR ceiling?

See docs/game-data-plan.md (Step 2). Plug the Step-1 `deck_value` field into the composite-WR teacher
target, re-train the seed-confirmed best config (good players + composite, 7-set big net, LOSO→DSK),
and evaluate the picks two ways on the holdout:
  - WR-agreement vs **GIH-WR** (the old, confounded metric — for comparability), and
  - WR-agreement vs **deck_value** (the de-confounded "estimated deck-WR" — the metric we actually
    care about: does the model take the card that genuinely wins more?).

Arms (single seed first; promote to multi-seed only if a target clears the ~0.01 prereq bar):
  - good/base         CE only, no WR target (reference)
  - good/comp:gih     composite teacher = GIH+IWD+ALSA      (the current best target)
  - good/comp:+value  composite teacher = GIH+IWD+ALSA+deck_value
  - good/comp:value   composite teacher = deck_value only

Needs a GPU pod (7-set big net). Run via the RunPod flow (docs/runpod-runbook.md):
    uv run python scripts/pod_game_value_target.py --seeds 0
"""
from __future__ import annotations

import argparse
import json
import pathlib

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
TRAIN = ["BLB", "OTJ", "WOE", "MKM", "LCI", "MH3", "MOM"]
HOLDOUT = "DSK"
GIH = "ever_drawn_win_rate"
BASE_FIELDS = ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"]
TARGETS = {                                   # teacher composite field sets
    "gih": BASE_FIELDS,
    "+value": BASE_FIELDS + ["deck_value"],
    "value": ["deck_value"],
}


def spec(s, merged_dir):
    """Set spec with a ratings file that has deck_value merged in (so composite can read it)."""
    base = {
        "parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
        "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
        "scryfall": f"{D}/scryfall/{s.lower()}.json",
        "ratings_orig": f"{D}/ratings/{s}.PremierDraft.ratings.json",
        "gamevalue": f"{D}/gamevalue/{s}.PremierDraft.gamevalue.json",
    }
    base["ratings"] = merged_ratings_with_value(
        base["ratings_orig"], base["gamevalue"], f"{merged_dir}/{s}.merged.json")
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-sets", default=",".join(TRAIN))
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--min-games", type=float, default=50)
    ap.add_argument("--targets", default="gih,+value,value")
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--enc-hidden", type=int, default=1024)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--n-sab", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--wr-tau", type=float, default=1.0)
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=1.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default="data/game_value_target_DSK.json")
    a = ap.parse_args(argv)

    merged_dir = "data/ratings_merged"
    pathlib.Path(merged_dir).mkdir(parents=True, exist_ok=True)
    train_sets = [s.strip() for s in a.train_sets.split(",")]
    targets = [t.strip() for t in a.targets.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    skill = {"min_winrate": a.min_winrate, "min_games": a.min_games, "ranks": None}

    train_specs = [spec(s, merged_dir) for s in train_sets]
    hold = spec(a.holdout, merged_dir)
    print(f"train={train_sets} holdout={a.holdout}  net=emb{a.emb_dim}/h{a.enc_hidden}/L{a.enc_layers} "
          f"targets={targets} seeds={seeds}")

    emb = get_embedder("all-MiniLM-L6-v2")
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=True)
    hmat, _ = build_content_matrix(hold["manifest"], hold["scryfall"], embedder=emb, text=True)
    dev = pick_device(a.device)
    bases = [(RemappedDataset(DraftPickDataset(s["parquet"]), l2g), s["parquet"])
             for s, l2g in zip(train_specs, l2gs)]
    rating_specs = [{"manifest": s["manifest"], "ratings": s["ratings"]} for s in train_specs]

    # teacher per target (composite over the target's fields, on the merged ratings)
    teachers = {}
    for t in targets:
        cs, cm = composite_card_quality(rating_specs, l2gs, ginfo["n_cards"], fields=TARGETS[t])
        teachers[t] = WRSoftmaxTeacher(cs, cm, tau=a.wr_tau, device=dev)

    # holdout eval fields: GIH (old metric) and deck_value (estimated deck-WR)
    wr_gih = align_winrates(hold["manifest"], hold["ratings"], field=GIH)
    wr_val = align_winrates(hold["manifest"], hold["ratings"], field="deck_value")
    novel = novel_mask_for_holdout(hold["manifest"], set(key_to_idx))
    hp = dict(emb_dim=a.emb_dim, enc_hidden=a.enc_hidden, enc_layers=a.enc_layers, dropout=0.1,
              pool="set_transformer", n_heads=4, n_sab=a.n_sab)
    common = dict(checkpoint_dir="data/checkpoints", checkpoint_every=0, n_cards=ginfo["n_cards"],
                  epochs=a.epochs, lr=1e-3, warmup_frac=0.1, grad_clip=1.0)

    def train_arm(arm, seed, teacher=None):
        print(f"\n-- {arm} (seed {seed}) --", flush=True)
        torch.manual_seed(seed)
        tdl, vdl = _loaders(bases, 0.05, split_seed=seed, batch_size=512, skill=skill)
        m = _build_model(gmat, dev, **hp)
        extra = (dict(teacher=teacher, distill_lambda=a.distill_lambda, distill_temp=a.distill_temp)
                 if teacher is not None else {})
        train_loop(m, tdl, vdl, dev, tag=arm, ckpt_prefix=arm, **common, **extra)
        return m

    def evaluate(m):
        m.set_content(torch.from_numpy(hmat))
        g = evaluate_on_set(m, hold["parquet"], dev, novel_card=novel, card_wr=wr_gih)
        v = evaluate_on_set(m, hold["parquet"], dev, novel_card=novel, card_wr=wr_val)
        return {"top1": g["top1"], "novel_top1": g.get("novel_top1"),
                "wr_agree_gih": g.get("wr_agreement_model"), "wr_agree_value": v.get("wr_agreement_model"),
                "human_gih": g.get("wr_agreement_human"), "human_value": v.get("wr_agreement_human")}

    runs = []
    for seed in seeds:
        arms = {"good/base": train_arm("good_base", seed, teacher=None)}
        for t in targets:
            arms[f"good/comp:{t}"] = train_arm(f"good_comp_{t}", seed, teacher=teachers[t])
        runs.append({"seed": seed, "results": {name: evaluate(m) for name, m in arms.items()}})

    # report
    print("\n=== Step 2 — WR-agreement vs GIH and vs deck_value (estimated deck-WR), DSK holdout ===")
    print(f"  {'arm':<18}{'top1':>8}{'WR-agree:GIH':>14}{'WR-agree:value':>16}")
    agg = _aggregate(runs)
    for name, m in agg.items():
        print(f"  {name:<18}{m['top1']:>8.4f}{m['wr_agree_gih']:>14.4f}{m['wr_agree_value']:>16.4f}")
    hum = runs[0]["results"]["good/base"]
    print(f"  {'human':<18}{'':>8}{hum['human_gih']:>14.4f}{hum['human_value']:>16.4f}")
    print("\n  Headline deltas vs the GIH-target best (good/comp:gih):")
    base = agg["good/comp:gih"]
    for t in [f"good/comp:{x}" for x in targets if x != "gih"]:
        if t in agg:
            print(f"    {t} − gih:  WR-agree:value {agg[t]['wr_agree_value']-base['wr_agree_value']:+.4f}"
                  f"   WR-agree:GIH {agg[t]['wr_agree_gih']-base['wr_agree_gih']:+.4f}")

    out = pathlib.Path(a.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {"train_sets": train_sets, "holdout": a.holdout,
                   "net": f"emb{a.emb_dim}h{a.enc_hidden}L{a.enc_layers}", "targets": targets,
                   "seeds": seeds}, "runs": runs, "aggregate": agg}, indent=2, default=float))
    print(f"\nwrote {out}")


def _aggregate(runs):
    import statistics
    names = list(runs[0]["results"])
    agg = {}
    for name in names:
        agg[name] = {}
        for k in ("top1", "wr_agree_gih", "wr_agree_value"):
            vals = [r["results"][name][k] for r in runs if r["results"][name][k] is not None]
            agg[name][k] = statistics.mean(vals) if vals else float("nan")
            if len(vals) > 1:
                agg[name][k + "_std"] = statistics.pstdev(vals)
    return agg


if __name__ == "__main__":
    main()
