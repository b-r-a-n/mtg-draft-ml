"""WR-softmax soft-label distillation (DD-004 #1) — good-not-just-human as a *dense* target.

`pick_advantage_weights` already biases toward winning picks, but only by *reweighting* the hard-label
CE example by a scalar (its docstring: "WITHOUT ever changing the target"). This instead reshapes the
**target**: the teacher distribution over the pack is softmax(card win-rate / tau), so the student is
trained to match the full win-rate *ranking*, not just to upweight imitation of the human's pick.

The headline comparison is dense-vs-scalar on the same held-out set: baseline CE vs CE + WR-softmax KD
(this) vs CE + IWD advantage-weighting (the existing scalar method). Judge on WR-agreement and
avg-pick-WR (does the model take the higher-win-rate card in the pack), measured on an unseen set —
i.e. does a win-rate target injected on the training sets generalize through the content encoder.

The teacher uses TRAINING-set ratings (those sets have 17lands data); it reuses the exact same KD term
and `train_loop` hook as the ensemble teacher — only `mean_probs` differs.
"""
from __future__ import annotations

import json

import torch

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, RemappedDataset
from ..eval.generalization import evaluate_on_set, novel_mask_for_holdout
from ..eval.winrate import align_winrates, build_global_wr_targets, composite_card_quality
from ..training.train import pick_device
from ..training.train_content import train_loop
from .ensemble import _build_model, _loaders  # shared train helpers (same package)


class WRSoftmaxTeacher:
    """Per-pack soft target ∝ softmax(card win-rate / tau) — "prefer the higher-WR card", densely.

    `score`/`mask` are per-global-card arrays (z-scored relative win rate + a rated mask, e.g. from
    `eval.winrate.build_global_wr_targets`). `tau` sets target sharpness (small = peakier). Unrated
    real pack cards sit at the pack-mean score (neutral — unrated ≠ weak); packs with <2 rated cards
    carry no win-rate signal and fall back to uniform. Same `mean_probs` interface as
    `EnsembleTeacher`, so it drops straight into the `train_loop` distillation hook.
    """

    def __init__(self, score, mask, tau: float = 1.0, device: str = "cpu"):
        self.score = torch.as_tensor(score, dtype=torch.float32, device=device)
        self.mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        self.tau = tau

    @torch.no_grad()
    def mean_probs(self, pool, pool_mask, pack, pack_mask, temp=None):
        # temp is ignored: the teacher's sharpness is set by tau (WR values have their own scale),
        # while the KD loss still softens the *student* at distill_temp.
        s = self.score[pack]                                  # [B,P]
        rated = self.mask[pack] & pack_mask
        n_rated = rated.sum(-1, keepdim=True)
        mean = torch.where(rated, s, torch.zeros_like(s)).sum(-1, keepdim=True) / n_rated.clamp_min(1)
        s = torch.where(rated, s, mean.expand_as(s)).masked_fill(~pack_mask, float("-inf"))
        probs = torch.softmax(s / self.tau, dim=-1)
        degen = (n_rated < 2).expand_as(probs)                # no real choice among rated cards
        if bool(degen.any()):
            uni = pack_mask.float()
            uni = uni / uni.sum(-1, keepdim=True).clamp_min(1)
            probs = torch.where(degen, uni, probs)
        return probs


def _metrics(m: dict) -> dict:
    return {k: m.get(k) for k in ("top1", "top5", "mtpd", "wr_agreement_model", "avg_pick_wr_model",
                                  "wr_agreement_human", "avg_pick_wr_human", "n")}


def run_wr_distill(
    train_specs: list[dict], holdout_spec: dict, *, holdout_ratings: str,
    wr_field: str = "ever_drawn_win_rate", wr_tau: float = 1.0,
    quality_fields: list[str] | None = None,
    distill_lambda: float = 0.5, distill_temp: float = 1.0, distill_topk: int = 0,
    adv_tau: float = 0.03,
    embedder: str = "all-MiniLM-L6-v2", text: bool = True,
    emb_dim: int = 256, enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
    pool: str = "set_transformer", n_heads: int = 4, n_sab: int = 1,
    warmup_frac: float = 0.1, grad_clip: float = 1.0,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
    device: str = "auto", checkpoint_dir: str = "data/checkpoints", seed: int = 0,
    out_json: str | None = None,
) -> dict:
    """Compare CE baseline vs CE+WR-softmax-KD (dense) vs CE+IWD-advantage (scalar) on a holdout.

    Each train spec must carry a "ratings" path (17lands ratings for that set). The dense and scalar
    methods use the SAME win-rate field; the question is whether reshaping the target beats reweighting
    the example for taking winning picks on an unseen set.
    """
    emb = get_embedder(embedder)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=text)
    hmat, hinfo = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                       embedder=emb, text=text)
    assert gmat.shape[1] == hmat.shape[1], (gmat.shape, hmat.shape)
    dev = pick_device(device)

    bases = [(RemappedDataset(DraftPickDataset(s["parquet"]), l2g), s["parquet"])
             for s, l2g in zip(train_specs, l2gs)]
    rating_specs = [{"manifest": s["manifest"], "ratings": s["ratings"]} for s in train_specs]
    # z-scored relative WR for the dense teacher target; raw WR for the scalar advantage comparison.
    wr_score, wr_mask = build_global_wr_targets(rating_specs, l2gs, ginfo["n_cards"],
                                                field=wr_field, standardize=True)
    adv_target, adv_mask = build_global_wr_targets(rating_specs, l2gs, ginfo["n_cards"],
                                                   field=wr_field, standardize=False)
    teacher = WRSoftmaxTeacher(wr_score, wr_mask, tau=wr_tau, device=dev)
    # optional: a richer composite target from MULTIPLE 17lands fields (GIH-WR + IWD + ALSA + …),
    # confidence-shrunk — vs the single-field `wr_field` teacher above.
    comp_teacher = None
    if quality_fields:
        cscore, cmask = composite_card_quality(rating_specs, l2gs, ginfo["n_cards"], fields=quality_fields)
        comp_teacher = WRSoftmaxTeacher(cscore, cmask, tau=wr_tau, device=dev)
    print(f"device={dev}  train {tuple(gmat.shape)} from {ginfo['n_sets']} sets  "
          f"rated={int(wr_mask.sum())}/{ginfo['n_cards']}  wr_tau={wr_tau} lambda={distill_lambda} "
          f"temp={distill_temp} topk={distill_topk}  adv_tau={adv_tau}  "
          f"quality_fields={list(quality_fields) if quality_fields else None}")

    hp = dict(emb_dim=emb_dim, enc_hidden=enc_hidden, enc_layers=enc_layers, dropout=dropout,
              pool=pool, n_heads=n_heads, n_sab=n_sab)
    common = dict(checkpoint_dir=checkpoint_dir, checkpoint_every=0, n_cards=ginfo["n_cards"],
                  epochs=epochs, lr=lr, warmup_frac=warmup_frac, grad_clip=grad_clip)

    def _train(tag, **extra):
        print(f"\n-- {tag} student (seed {seed}) --")
        torch.manual_seed(seed)
        tdl, vdl = _loaders(bases, val_frac, split_seed=seed, batch_size=batch_size)
        m = _build_model(gmat, dev, **hp)
        train_loop(m, tdl, vdl, dev, tag=tag, ckpt_prefix=tag, **common, **extra)
        return m

    base = _train("base")
    wr_kd = _train("wr_kd", teacher=teacher, distill_lambda=distill_lambda,
                   distill_temp=distill_temp, distill_topk=distill_topk)
    wr_kd_comp = _train("wr_kd_comp", teacher=comp_teacher, distill_lambda=distill_lambda,
                        distill_temp=distill_temp, distill_topk=distill_topk) if comp_teacher else None
    adv = _train("adv", adv_target=adv_target, adv_mask=adv_mask, adv_tau=adv_tau)

    novel = novel_mask_for_holdout(holdout_spec["manifest"], set(key_to_idx))
    card_wr = align_winrates(holdout_spec["manifest"], holdout_ratings, field=wr_field)

    def _eval(model):
        model.set_content(torch.from_numpy(hmat))
        return _metrics(evaluate_on_set(model, holdout_spec["parquet"], dev,
                                        novel_card=novel, card_wr=card_wr))

    results = {
        "mode": "wr_distill", "embedder": embedder, "pool": pool, "wr_field": wr_field,
        "wr_tau": wr_tau, "distill_lambda": distill_lambda, "distill_temp": distill_temp,
        "distill_topk": distill_topk, "adv_tau": adv_tau,
        "quality_fields": list(quality_fields) if quality_fields else None,
        "n_train_sets": len(train_specs), "train_cards": ginfo["n_cards"],
        "holdout_cards": hinfo["n_cards"], "frac_cards_novel": float(novel.float().mean()),
        "baseline": _eval(base), "wr_kd": _eval(wr_kd), "adv": _eval(adv),
        "wr_kd_comp": _eval(wr_kd_comp) if wr_kd_comp is not None else None,
    }
    _print_report(results)
    if out_json:
        json.dump(results, open(out_json, "w"), indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _print_report(r: dict):
    rows = [("baseline (CE)", r["baseline"]), ("WR-KD (single)", r["wr_kd"])]
    if r.get("wr_kd_comp"):
        rows.append(("WR-KD (composite)", r["wr_kd_comp"]))
    rows.append(("advantage (scalar)", r["adv"]))
    print("\n=== WR-softmax distillation report (good-not-just-human: dense target vs scalar reweight) ===")
    print(f"wr_field={r['wr_field']} wr_tau={r['wr_tau']} lambda={r['distill_lambda']} "
          f"topk={r['distill_topk']} adv_tau={r['adv_tau']}  "
          f"quality_fields={r.get('quality_fields')}  ({r['frac_cards_novel']*100:.0f}% holdout novel)")
    print(f"  {'policy':<22}{'WR-agree':>10}{'avg_pickWR':>12}{'top1':>8}{'top5':>8}{'mtpd':>8}")
    for name, m in rows:
        print(f"  {name:<22}{(m['wr_agreement_model'] or 0):>10.4f}{(m['avg_pick_wr_model'] or 0):>12.4f}"
              f"{m['top1']:>8.4f}{(m['top5'] or 0):>8.4f}{m['mtpd']:>8.3f}")
    h = r["baseline"]
    print(f"  {'human (reference)':<22}{(h['wr_agreement_human'] or 0):>10.4f}"
          f"{(h['avg_pick_wr_human'] or 0):>12.4f}")
    d_wr = (r["wr_kd"]["wr_agreement_model"] or 0) - (r["adv"]["wr_agreement_model"] or 0)
    print(f"  dense - scalar:  WR-agree {d_wr:+.4f}  (>0 => reshaping the target beats reweighting)")
    if r.get("wr_kd_comp"):
        d_comp = (r["wr_kd_comp"]["wr_agreement_model"] or 0) - (r["wr_kd"]["wr_agreement_model"] or 0)
        print(f"  composite - single:  WR-agree {d_comp:+.4f}  (>0 => the richer fields help the target)")


def _spec(s: str) -> dict:
    p = [x.strip() for x in s.split(",")]
    out = {"parquet": p[0], "manifest": p[1], "scryfall": p[2]}
    if len(p) > 3:
        out["ratings"] = p[3]
    return out


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="WR-softmax distillation: CE vs CE+WR-KD (dense) vs CE+advantage (scalar). "
                    "Each --train is 'parquet,manifest,scryfall,ratings'; --holdout omits ratings.")
    ap.add_argument("--train", action="append", required=True, metavar="PARQUET,MANIFEST,SCRYFALL,RATINGS")
    ap.add_argument("--holdout", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--holdout-ratings", required=True, help="17lands ratings JSON for the holdout")
    ap.add_argument("--wr-field", default="ever_drawn_win_rate")
    ap.add_argument("--wr-tau", type=float, default=1.0)
    ap.add_argument("--quality-fields", default=None,
                    help="comma list of 17lands fields for a composite target arm, e.g. "
                         "ever_drawn_win_rate,drawn_improvement_win_rate,avg_pick")
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=1.0)
    ap.add_argument("--distill-topk", type=int, default=0)
    ap.add_argument("--adv-tau", type=float, default=0.03)
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--pool", default="set_transformer", choices=["mean", "set_transformer"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_wr_distill(
        [_spec(t) for t in a.train], _spec(a.holdout), holdout_ratings=a.holdout_ratings,
        wr_field=a.wr_field, wr_tau=a.wr_tau,
        quality_fields=[s.strip() for s in a.quality_fields.split(",")] if a.quality_fields else None,
        distill_lambda=a.distill_lambda,
        distill_temp=a.distill_temp, distill_topk=a.distill_topk, adv_tau=a.adv_tau,
        embedder=a.embedder, text=not a.no_text, pool=a.pool,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device, out_json=a.out_json,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
