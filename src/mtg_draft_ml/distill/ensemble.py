"""Ensemble-of-seeds soft-label distillation (DD-004 #1) — better sample efficiency, same data.

The one-hot human pick collapses an 8-15 card pack to a single bit and is itself noisy (individual
drafter disagreement is the ~0.58 top-1 ceiling). This trains K independently-seeded teachers on the
same data, averages their per-pack distributions into a denoised soft target, and distills that into
one student via KL on top of the hard label. No new data, no leaky features — the purest test of
"is the bottleneck the lossy objective rather than the data?".

The teacher runs **on the fly** inside the student's training step (`EnsembleTeacher.mean_probs`):
the models are small (~2-15M params) and frozen, so K extra forward passes per batch is cheap, and
it avoids caching per-pick distributions and the example-id join that would require.

Headline comparison on a held-out set, all on the same trained content model:
  - baseline   — single seed, plain CE
  - distilled  — single seed, CE + ensemble-KD (the candidate)
  - ensemble   — the K-teacher average itself (the target / upper reference)
Judge on WR-agreement and ranking quality (top-3/5, MTPD), NOT just top-1 — if the residual is
genuine label noise, top-1 stays ceilinged even when the picks get better (the same reason the
project concluded 0.58 is a ceiling). `train_frac < 1` runs the actual sample-efficiency test: does
KD on a fraction of the data match plain CE on all of it?
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, RemappedDataset, collate_picks, skill_filter_indices
from ..eval.generalization import evaluate_on_set, novel_mask_for_holdout
from ..models.draft_model import ContentDraftModel
from ..training.train import draft_level_split, pick_device
from ..training.train_content import train_loop


class EnsembleTeacher:
    """K frozen models -> one denoised pack distribution (the mean of their softmaxes)."""

    def __init__(self, models: list[ContentDraftModel]):
        self.models = [m.eval() for m in models]

    @torch.no_grad()
    def mean_probs(self, pool, pool_mask, pack, pack_mask, temp: float = 2.0) -> torch.Tensor:
        """Mean over models of softmax(logits / temp) — a valid pack distribution (0 at pads)."""
        acc = None
        for m in self.models:
            p = F.softmax(m(pool, pool_mask, pack, pack_mask) / temp, dim=-1)  # -inf pads -> 0
            acc = p if acc is None else acc + p
        return acc / len(self.models)


class CompositeTeacher:
    """Weighted average of several teachers' pack distributions — compose KD signals.

    Lets you combine denoising (ensemble) with good-not-just-human (WR-softmax) or a leaky-feature
    teacher in one target: `CompositeTeacher([ensemble, wr_teacher], weights=[1, 2])`. The mean of
    valid distributions is a valid distribution; same `mean_probs` interface, so it drops into the
    same `train_loop` hook.
    """

    def __init__(self, teachers: list, weights: list[float] | None = None):
        self.teachers = teachers
        self.weights = weights if weights is not None else [1.0] * len(teachers)

    @torch.no_grad()
    def mean_probs(self, pool, pool_mask, pack, pack_mask, temp: float = 2.0) -> torch.Tensor:
        acc, wsum = None, 0.0
        for t, w in zip(self.teachers, self.weights):
            p = t.mean_probs(pool, pool_mask, pack, pack_mask, temp=temp) * w
            acc = p if acc is None else acc + p
            wsum += w
        return acc / wsum


class EnsembleModel:
    """Eval-time wrapper: makes the ensemble look like a model for `eval.evaluate_on_set`.

    Returns log(mean pack-prob) as pseudo-logits (argmax / top-k behave correctly; pads are -inf).
    """

    def __init__(self, teacher: EnsembleTeacher):
        self.teacher = teacher

    def eval(self):
        return self

    def __call__(self, pool, pool_mask, pack, pack_mask, neg_idx=None):
        return self.teacher.mean_probs(pool, pool_mask, pack, pack_mask, temp=1.0).clamp_min(1e-12).log()


def _build_model(gmat, dev, *, emb_dim, enc_hidden, enc_layers, dropout, pool, n_heads, n_sab):
    return ContentDraftModel(torch.from_numpy(gmat), emb_dim=emb_dim, enc_hidden=enc_hidden,
                             enc_layers=enc_layers, dropout=dropout, pool=pool,
                             n_heads=n_heads, n_sab=n_sab).to(dev)


def _loaders(bases, val_frac, split_seed, batch_size, train_frac=1.0, skill=None):
    """Build train/val loaders over the remapped sets. train_frac<1 subsamples train picks.

    skill (a dict of skill_filter_indices kwargs) restricts BOTH train and val to good-player picks.
    """
    train_subsets, val_subsets = [], []
    rng = np.random.default_rng(split_seed)
    for rds, pq in bases:
        tr, va = draft_level_split(pq, val_frac, split_seed)
        if skill:
            keep = set(skill_filter_indices(pq, **skill))
            tr = [i for i in tr if i in keep]
            va = [i for i in va if i in keep]
        if train_frac < 1.0:
            tr = np.asarray(tr)[rng.permutation(len(tr))[: int(train_frac * len(tr))]]
        train_subsets.append(Subset(rds, list(tr)))
        val_subsets.append(Subset(rds, list(va)))
    tdl = DataLoader(ConcatDataset(train_subsets), batch_size=batch_size, shuffle=True,
                     collate_fn=collate_picks)
    vdl = DataLoader(ConcatDataset(val_subsets), batch_size=batch_size, shuffle=False,
                     collate_fn=collate_picks)
    return tdl, vdl


def run_ensemble_distill(
    train_specs: list[dict], holdout_spec: dict, *,
    n_teachers: int = 3, distill_lambda: float = 0.5, distill_temp: float = 2.0, distill_topk: int = 0,
    train_frac: float = 1.0, holdout_ratings: str | None = None,
    wr_field: str = "ever_drawn_win_rate",
    embedder: str = "all-MiniLM-L6-v2", text: bool = True,
    emb_dim: int = 256, enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
    pool: str = "set_transformer", n_heads: int = 4, n_sab: int = 1,
    warmup_frac: float = 0.1, grad_clip: float = 1.0,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
    device: str = "auto", checkpoint_dir: str = "data/checkpoints", seed: int = 0,
    out_json: str | None = None,
) -> dict:
    """Train K seed teachers + a CE baseline + a KD-distilled student; compare on holdout."""
    emb = get_embedder(embedder)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=text)
    hmat, hinfo = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                       embedder=emb, text=text)
    assert gmat.shape[1] == hmat.shape[1], (gmat.shape, hmat.shape)
    dev = pick_device(device)
    print(f"device={dev}  train matrix {tuple(gmat.shape)} from {ginfo['n_sets']} sets  "
          f"teachers={n_teachers} lambda={distill_lambda} temp={distill_temp} topk={distill_topk} "
          f"train_frac={train_frac}")

    bases = [(RemappedDataset(DraftPickDataset(spec["parquet"]), l2g), spec["parquet"])
             for spec, l2g in zip(train_specs, l2gs)]
    hp = dict(emb_dim=emb_dim, enc_hidden=enc_hidden, enc_layers=enc_layers, dropout=dropout,
              pool=pool, n_heads=n_heads, n_sab=n_sab)
    common = dict(checkpoint_dir=checkpoint_dir, checkpoint_every=0, n_cards=ginfo["n_cards"],
                  epochs=epochs, lr=lr, warmup_frac=warmup_frac, grad_clip=grad_clip)

    # K teachers on full data, distinct seeds (init + split + shuffle) for ensemble diversity.
    teachers = []
    for k in range(1, n_teachers + 1):
        print(f"\n-- teacher {k}/{n_teachers} (seed {k}) --")
        torch.manual_seed(k)
        tdl, vdl = _loaders(bases, val_frac, split_seed=k, batch_size=batch_size)
        m = _build_model(gmat, dev, **hp)
        train_loop(m, tdl, vdl, dev, tag="teacher", ckpt_prefix=f"teacher{k}", **common)
        teachers.append(m)
    ensemble = EnsembleTeacher(teachers)

    # baseline (CE) and distilled (CE + KD) students share seed 0 + the same (possibly subsampled) data
    print("\n-- baseline student (CE, seed 0) --")
    torch.manual_seed(seed)
    tdl, vdl = _loaders(bases, val_frac, split_seed=seed, batch_size=batch_size, train_frac=train_frac)
    base = _build_model(gmat, dev, **hp)
    base_best = train_loop(base, tdl, vdl, dev, tag="base", ckpt_prefix="base", **common)

    print("\n-- distilled student (CE + ensemble-KD, seed 0) --")
    torch.manual_seed(seed)
    tdl, vdl = _loaders(bases, val_frac, split_seed=seed, batch_size=batch_size, train_frac=train_frac)
    dist = _build_model(gmat, dev, **hp)
    dist_best = train_loop(dist, tdl, vdl, dev, tag="distill", ckpt_prefix="distill",
                           teacher=ensemble, distill_lambda=distill_lambda,
                           distill_temp=distill_temp, distill_topk=distill_topk, **common)

    # ---- held-out evaluation ----
    novel = novel_mask_for_holdout(holdout_spec["manifest"], set(key_to_idx))
    card_wr = None
    if holdout_ratings is not None:
        from ..eval.winrate import align_winrates
        card_wr = align_winrates(holdout_spec["manifest"], holdout_ratings, field=wr_field)

    def _eval(model):
        model.set_content(torch.from_numpy(hmat))         # retarget the content head to the new set
        return _pick(evaluate_on_set(model, holdout_spec["parquet"], dev,
                                     novel_card=novel, card_wr=card_wr))

    base_m = _eval(base)
    dist_m = _eval(dist)
    for t in teachers:                                    # retarget each teacher for the ensemble eval
        t.set_content(torch.from_numpy(hmat))
    ens_m = _pick(evaluate_on_set(EnsembleModel(ensemble), holdout_spec["parquet"], dev,
                                  novel_card=novel, card_wr=card_wr))

    results = {
        "mode": "ensemble_distill", "embedder": embedder, "pool": pool,
        "n_teachers": n_teachers, "distill_lambda": distill_lambda, "distill_temp": distill_temp,
        "distill_topk": distill_topk, "train_frac": train_frac,
        "n_train_sets": len(train_specs), "train_cards": ginfo["n_cards"],
        "holdout_cards": hinfo["n_cards"], "frac_cards_novel": float(novel.float().mean()),
        "baseline": base_m, "distilled": dist_m, "ensemble": ens_m,
        "baseline_val_top1": base_best["top1"], "distilled_val_top1": dist_best["top1"],
    }
    _print_report(results)
    if out_json:
        json.dump(results, open(out_json, "w"), indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _pick(m: dict) -> dict:
    return {k: m.get(k) for k in ("top1", "top3", "top5", "mtpd", "novel_top1",
                                  "wr_agreement_model", "avg_pick_wr_model", "n")}


def _gap_closed(base, dist, ens, key) -> float:
    b, d, e = base.get(key), dist.get(key), ens.get(key)
    if None in (b, d, e) or abs(e - b) < 1e-9:
        return float("nan")
    return (d - b) / (e - b)


def _print_report(r: dict):
    rows = [("baseline (CE)", r["baseline"]), ("distilled (CE+KD)", r["distilled"]),
            ("ensemble (target)", r["ensemble"])]
    has_wr = r["baseline"].get("wr_agreement_model") is not None
    print("\n=== Ensemble-of-seeds distillation report ===")
    print(f"teachers={r['n_teachers']} lambda={r['distill_lambda']} temp={r['distill_temp']} "
          f"topk={r['distill_topk']} train_frac={r['train_frac']}  "
          f"({r['frac_cards_novel']*100:.0f}% holdout cards novel)")
    head = f"  {'policy':<20}{'top1':>8}{'top3':>8}{'top5':>8}{'mtpd':>8}"
    if has_wr:
        head += f"{'WR-agree':>10}{'avg_pickWR':>12}"
    print(head)
    for name, m in rows:
        line = (f"  {name:<20}{m['top1']:>8.4f}{(m['top3'] or 0):>8.4f}{(m['top5'] or 0):>8.4f}"
                f"{m['mtpd']:>8.3f}")
        if has_wr:
            line += f"{(m['wr_agreement_model'] or 0):>10.4f}{(m['avg_pick_wr_model'] or 0):>12.4f}"
        print(line)
    print(f"  distilled - baseline:  top1 {r['distilled']['top1'] - r['baseline']['top1']:+.4f}"
          f"   top5 {(r['distilled']['top5'] or 0) - (r['baseline']['top5'] or 0):+.4f}"
          f"   mtpd {r['distilled']['mtpd'] - r['baseline']['mtpd']:+.3f}")
    print(f"  fraction of ensemble gap closed: top5 {_gap_closed(r['baseline'], r['distilled'], r['ensemble'], 'top5')*100:.0f}%"
          + (f"   WR-agree {_gap_closed(r['baseline'], r['distilled'], r['ensemble'], 'wr_agreement_model')*100:.0f}%"
             if has_wr else ""))


def _spec(s: str) -> dict:
    p = [x.strip() for x in s.split(",")]
    return {"parquet": p[0], "manifest": p[1], "scryfall": p[2]}


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="Ensemble-of-seeds soft-label distillation. Train --teachers seed models, then "
                    "compare CE baseline vs CE+KD student on --holdout. Specs are 'parquet,manifest,scryfall'.")
    ap.add_argument("--train", action="append", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--holdout", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--teachers", type=int, default=3)
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-topk", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=1.0, help="<1 = sample-efficiency test")
    ap.add_argument("--holdout-ratings", default=None, help="17lands ratings JSON for WR-agreement")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--pool", default="set_transformer", choices=["mean", "set_transformer"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_ensemble_distill(
        [_spec(t) for t in a.train], _spec(a.holdout),
        n_teachers=a.teachers, distill_lambda=a.distill_lambda, distill_temp=a.distill_temp,
        distill_topk=a.distill_topk, train_frac=a.train_frac, holdout_ratings=a.holdout_ratings,
        embedder=a.embedder, text=not a.no_text, pool=a.pool,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device, out_json=a.out_json,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
