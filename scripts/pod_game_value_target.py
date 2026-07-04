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
import re

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

# Fixed export shapes used when --export-onnx-dir is set.
# pool = up to 45 cards (3 packs × 15 picks); pack = 15 cards.
_ONNX_MAX_POOL = 45
_ONNX_MAX_PACK = 15

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
    # some older sets (e.g. MID) have no game_data deck_value — fall back to plain ratings (their
    # cards just contribute no deck_value to the composite, fine for a 4-field blend).
    gv = pathlib.Path(base["gamevalue"])
    base["ratings"] = (merged_ratings_with_value(base["ratings_orig"], base["gamevalue"],
                                                 f"{merged_dir}/{s}.merged.json")
                       if gv.exists() else base["ratings_orig"])
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
    ap.add_argument("--export-onnx-dir", default=None, metavar="DIR",
                    help="if set, export each trained arm to ONNX at DIR/<arm-slug>.seed<seed>.onnx "
                         "and write DIR/meta.json with model config")
    a = ap.parse_args(argv)

    merged_dir = "data/ratings_merged"
    pathlib.Path(merged_dir).mkdir(parents=True, exist_ok=True)
    train_sets = [s.strip() for s in a.train_sets.split(",")]
    targets = [t.strip() for t in a.targets.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    skill = {"min_winrate": a.min_winrate, "min_games": a.min_games, "ranks": None}

    onnx_dir = pathlib.Path(a.export_onnx_dir) if a.export_onnx_dir else None
    if onnx_dir is not None:
        onnx_dir.mkdir(parents=True, exist_ok=True)

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
        if onnx_dir is not None:
            slug = _arm_slug(arm)
            onnx_path = onnx_dir / f"{slug}.seed{seed}.onnx"
            _export_onnx(m, hmat, onnx_path)
            _write_meta(onnx_dir, a, train_sets)
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


def _arm_slug(arm: str) -> str:
    """Convert an arm name to a filesystem-safe slug.

    Examples:
        "good_base"       -> "good_base"
        "good_comp_+value" -> "good_comp_plusvalue"
        "good_comp:gih"   -> "good_comp_gih"
    """
    s = arm.replace("+", "plus")
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _export_onnx(model, holdout_mat: "np.ndarray", path: pathlib.Path) -> None:
    """Export `model` to ONNX at `path` with the HOLDOUT set's content table baked in.

    Mirrors export_webapp._export_onnx (scripts/export_webapp.py:163) exactly:
      - Retargets the content buffer to `holdout_mat` so card indices in the graph are
        HOLDOUT-set indices (e.g. DSK card indices matching webapp/data/DSK.cards.json `i`).
      - Fixed MAX_POOL/MAX_PACK shapes; only the batch axis is dynamic.
      - Inputs: pool int64[B,45], pool_mask bool[B,45], pack int64[B,15], pack_mask bool[B,15]
      - Output: logits float32[B,15]
      - No wrapper needed — ContentDraftModel.forward() already takes index inputs.
      - Runs the padding-inertness check after export (mirrors export_webapp:176).
    The original model's device, training state, and content table are restored after export.
    """
    import copy
    import numpy as np

    orig_training = model.training
    orig_content = model.content.clone()
    orig_device = orig_content.device

    # Deep-copy to CPU; retarget to the holdout set's content table.
    cpu_model = copy.deepcopy(model).eval().to("cpu")
    cpu_model.set_content(torch.from_numpy(holdout_mat))   # bake DSK indices into graph

    # Fixed-shape export — mirrors export_webapp._export_onnx (scripts/export_webapp.py:163-175)
    pool      = torch.zeros((1, _ONNX_MAX_POOL), dtype=torch.long)
    pool_mask = torch.zeros((1, _ONNX_MAX_POOL), dtype=torch.bool)
    pack      = torch.zeros((1, _ONNX_MAX_PACK), dtype=torch.long)
    pack_mask = torch.ones( (1, _ONNX_MAX_PACK), dtype=torch.bool)
    torch.onnx.export(
        cpu_model, (pool, pool_mask, pack, pack_mask), str(path),
        input_names=["pool", "pool_mask", "pack", "pack_mask"],
        output_names=["logits"],
        dynamic_axes={"pool": {0: "B"}, "pool_mask": {0: "B"}, "pack": {0: "B"},
                      "pack_mask": {0: "B"}, "logits": {0: "B"}},
        opset_version=17, dynamo=False,
    )
    _assert_padding_inert(cpu_model, path)

    # Restore original model state (deepcopy means original is untouched, but be explicit)
    model.train(orig_training)
    model.set_content(orig_content.to(orig_device))
    print(f"  exported ONNX → {path} ({path.stat().st_size / 1e6:.1f} MB)")


def _assert_padding_inert(model, path: pathlib.Path) -> None:
    """Verify that padding with mask=False is numerically identical to an unpadded forward pass.

    Mirrors export_webapp._assert_padding_inert (scripts/export_webapp.py:179-197).
    Uses card indices that are valid for the baked-in content table (indices < n_cards).
    """
    import numpy as np
    import onnxruntime as ort

    n_cards = model.content.shape[0]
    # Use small safe indices (well within any set's vocab)
    rp = [i % n_cards for i in [3, 10, 7, 4, 2]]
    rk = [i % n_cards for i in [1, 5, 6, 8, 9]]
    with torch.no_grad():
        ref = model(torch.tensor([rp]), torch.ones(1, len(rp), dtype=torch.bool),
                    torch.tensor([rk]), torch.ones(1, len(rk), dtype=torch.bool)).numpy()[0]
    pool = np.zeros((1, _ONNX_MAX_POOL), np.int64); pool[0, :len(rp)] = rp
    pm   = np.zeros((1, _ONNX_MAX_POOL), bool);     pm[0,   :len(rp)] = True
    pk   = np.zeros((1, _ONNX_MAX_PACK), np.int64); pk[0,   :len(rk)] = rk
    km   = np.zeros((1, _ONNX_MAX_PACK), bool);     km[0,   :len(rk)] = True
    got = ort.InferenceSession(str(path)).run(
        None, {"pool": pool, "pool_mask": pm, "pack": pk, "pack_mask": km})[0][0, :len(rk)]
    diff = float(np.abs(ref - got).max())
    assert diff < 1e-4, f"padded ONNX != unpadded torch (max diff {diff}) — masking not inert"
    print(f"  padding-inert parity OK (max diff {diff:.2e})")


def _write_meta(onnx_dir: pathlib.Path, a, train_sets: list) -> None:
    """Write (or overwrite) meta.json in onnx_dir with the run's global config."""
    meta = {
        "max_pool": _ONNX_MAX_POOL,
        "max_pack": _ONNX_MAX_PACK,
        "emb_dim": a.emb_dim,
        "enc_hidden": a.enc_hidden,
        "enc_layers": a.enc_layers,
        "train_sets": train_sets,
        "holdout": a.holdout,
        "input_signature": (
            "pool int64[B,max_pool], pool_mask bool[B,max_pool], "
            "pack int64[B,max_pack], pack_mask bool[B,max_pack] -> logits float32[B,max_pack]. "
            "Card indices are HOLDOUT-set indices (baked-in content table). "
            "Pad unused positions with index=0 and mask=False."
        ),
    }
    (onnx_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
