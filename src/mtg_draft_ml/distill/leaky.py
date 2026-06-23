"""Leaky-feature → release-day distillation (DD-004 #3) — smuggle privileged knowledge into a student.

Train a TEACHER whose card features include per-card win rate (powerful, but a post-hoc aggregate
that does not exist on a new set's release day and leaks outcome information). Distill its pack
rankings into a STUDENT that sees only release-day inputs (text + stats, no win rate). The teacher's
win-rate-informed *contextual* policy — "high-WR AND fits your pool" — gets baked into the student's
weights, so the student needs no win-rate features at inference.

This is distinct from the WR-softmax teacher: that target is the pure win-rate ordering (ignores
pool synergy); here the teacher *learns* a policy that blends win rate with the pool context, a
richer target. It reuses everything: the teacher is a `ContentDraftModel` on a win-rate-augmented
content matrix, wrapped by `EnsembleTeacher([teacher])` to expose the standard `mean_probs`.

Comparison on a held-out set:
  - baseline (CE)        — release-day student, hard label only (the floor)
  - distilled (CE+KD)    — release-day student + leaky-teacher target (the candidate)
  - teacher (leaky)      — the win-rate-augmented teacher itself (the ceiling the student chases;
                           uses the holdout's real win rate, which a real release-day deploy lacks)
Read WR-agreement / avg-pick-WR and ranking quality; the report prints the fraction of the
teacher−baseline gap the student closes.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, RemappedDataset
from ..eval.generalization import evaluate_on_set, novel_mask_for_holdout
from ..eval.winrate import align_winrates, build_global_wr_targets, zscore_ignore_nan
from ..training.train import pick_device
from ..training.train_content import train_loop
from .ensemble import EnsembleTeacher, _build_model, _loaders


def augment_with_winrate(base: np.ndarray, wr_z: np.ndarray, wr_mask: np.ndarray) -> np.ndarray:
    """Append two columns to the content matrix: z-scored win rate (0 where unrated) + a rated flag.

    The rated flag lets the encoder tell "WR≈0 because average" from "0 because unrated" (z-scored WR
    has mean 0, so 0-fill is ambiguous without it).
    """
    cols = np.stack([np.nan_to_num(wr_z, nan=0.0), wr_mask.astype(np.float32)], axis=1)
    return np.concatenate([base, cols.astype(np.float32)], axis=1)


def _wr_columns_for_holdout(manifest, ratings, field):
    """Per-holdout-card (z-scored WR, rated mask), matching the train-side standardization."""
    raw = align_winrates(manifest, ratings, field=field)      # NaN where unrated
    return zscore_ignore_nan(raw), ~np.isnan(raw)


def _metrics(m: dict) -> dict:
    return {k: m.get(k) for k in ("top1", "top5", "mtpd", "wr_agreement_model", "avg_pick_wr_model",
                                  "wr_agreement_human", "avg_pick_wr_human", "n")}


def run_leaky_distill(
    train_specs: list[dict], holdout_spec: dict, *, holdout_ratings: str,
    wr_field: str = "ever_drawn_win_rate",
    distill_lambda: float = 0.5, distill_temp: float = 2.0, distill_topk: int = 0,
    embedder: str = "all-MiniLM-L6-v2", text: bool = True,
    emb_dim: int = 256, enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
    pool: str = "set_transformer", n_heads: int = 4, n_sab: int = 1,
    warmup_frac: float = 0.1, grad_clip: float = 1.0,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
    device: str = "auto", checkpoint_dir: str = "data/checkpoints", seed: int = 0,
    teacher_seed: int = 1, out_json: str | None = None,
) -> dict:
    """Train a win-rate-augmented teacher, distill into a release-day student, compare on holdout.

    Each train spec needs a "ratings" path. The student never sees win rate; the teacher does.
    """
    emb = get_embedder(embedder)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=text)
    hmat, _ = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                   embedder=emb, text=text)
    assert gmat.shape[1] == hmat.shape[1], (gmat.shape, hmat.shape)
    dev = pick_device(device)

    bases = [(RemappedDataset(DraftPickDataset(s["parquet"]), l2g), s["parquet"])
             for s, l2g in zip(train_specs, l2gs)]
    rating_specs = [{"manifest": s["manifest"], "ratings": s["ratings"]} for s in train_specs]
    wr_z, wr_mask = build_global_wr_targets(rating_specs, l2gs, ginfo["n_cards"],
                                            field=wr_field, standardize=True)
    aug_gmat = augment_with_winrate(gmat, wr_z, wr_mask)
    print(f"device={dev}  base dim={gmat.shape[1]} -> teacher dim={aug_gmat.shape[1]}  "
          f"rated={int(wr_mask.sum())}/{ginfo['n_cards']}  lambda={distill_lambda} temp={distill_temp}")

    hp = dict(emb_dim=emb_dim, enc_hidden=enc_hidden, enc_layers=enc_layers, dropout=dropout,
              pool=pool, n_heads=n_heads, n_sab=n_sab)
    common = dict(checkpoint_dir=checkpoint_dir, checkpoint_every=0, n_cards=ginfo["n_cards"],
                  epochs=epochs, lr=lr, warmup_frac=warmup_frac, grad_clip=grad_clip)

    def _train(gmatrix, split_seed, tag, **extra):
        print(f"\n-- {tag} (seed {split_seed}) --")
        torch.manual_seed(split_seed)
        tdl, vdl = _loaders(bases, val_frac, split_seed=split_seed, batch_size=batch_size)
        m = _build_model(gmatrix, dev, **hp)
        train_loop(m, tdl, vdl, dev, tag=tag, ckpt_prefix=tag, **common, **extra)
        return m

    teacher_model = _train(aug_gmat, teacher_seed, "leaky_teacher")     # sees win rate
    teacher = EnsembleTeacher([teacher_model])                         # standard mean_probs wrapper
    base = _train(gmat, seed, "base")                                  # release-day floor
    dist = _train(gmat, seed, "distilled", teacher=teacher, distill_lambda=distill_lambda,
                  distill_temp=distill_temp, distill_topk=distill_topk)

    # ---- held-out evaluation ----
    novel = novel_mask_for_holdout(holdout_spec["manifest"], set(key_to_idx))
    card_wr = align_winrates(holdout_spec["manifest"], holdout_ratings, field=wr_field)

    def _eval_base(model):
        model.set_content(torch.from_numpy(hmat))
        return _metrics(evaluate_on_set(model, holdout_spec["parquet"], dev,
                                        novel_card=novel, card_wr=card_wr))

    base_m = _eval_base(base)
    dist_m = _eval_base(dist)
    # teacher ceiling: it needs the holdout's win-rate columns (a real release-day deploy wouldn't
    # have these — this is the offline upper reference the student is chasing).
    hwr_z, hwr_mask = _wr_columns_for_holdout(holdout_spec["manifest"], holdout_ratings, wr_field)
    teacher_model.set_content(torch.from_numpy(augment_with_winrate(hmat, hwr_z, hwr_mask)))
    teach_m = _metrics(evaluate_on_set(teacher_model, holdout_spec["parquet"], dev,
                                       novel_card=novel, card_wr=card_wr))

    results = {
        "mode": "leaky_distill", "embedder": embedder, "pool": pool, "wr_field": wr_field,
        "distill_lambda": distill_lambda, "distill_temp": distill_temp, "distill_topk": distill_topk,
        "n_train_sets": len(train_specs), "train_cards": ginfo["n_cards"],
        "teacher_dim": aug_gmat.shape[1], "base_dim": gmat.shape[1],
        "frac_cards_novel": float(novel.float().mean()),
        "baseline": base_m, "distilled": dist_m, "teacher": teach_m,
    }
    _print_report(results)
    if out_json:
        json.dump(results, open(out_json, "w"), indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _gap_closed(base, dist, teach, key) -> float:
    b, d, t = base.get(key), dist.get(key), teach.get(key)
    if None in (b, d, t) or abs(t - b) < 1e-9:
        return float("nan")
    return (d - b) / (t - b)


def _print_report(r: dict):
    rows = [("baseline (CE)", r["baseline"]), ("distilled (CE+KD)", r["distilled"]),
            ("teacher (leaky)", r["teacher"])]
    print("\n=== Leaky-feature -> release-day distillation report ===")
    print(f"wr_field={r['wr_field']} base_dim={r['base_dim']} teacher_dim={r['teacher_dim']} "
          f"lambda={r['distill_lambda']} topk={r['distill_topk']}  "
          f"({r['frac_cards_novel']*100:.0f}% holdout cards novel)")
    print(f"  {'policy':<20}{'WR-agree':>10}{'avg_pickWR':>12}{'top1':>8}{'top5':>8}{'mtpd':>8}")
    for name, m in rows:
        print(f"  {name:<20}{(m['wr_agreement_model'] or 0):>10.4f}{(m['avg_pick_wr_model'] or 0):>12.4f}"
              f"{m['top1']:>8.4f}{(m['top5'] or 0):>8.4f}{m['mtpd']:>8.3f}")
    h = r["baseline"]
    print(f"  {'human (reference)':<20}{(h['wr_agreement_human'] or 0):>10.4f}"
          f"{(h['avg_pick_wr_human'] or 0):>12.4f}")
    print(f"  distilled - baseline:  WR-agree "
          f"{(r['distilled']['wr_agreement_model'] or 0) - (r['baseline']['wr_agreement_model'] or 0):+.4f}"
          f"   top5 {(r['distilled']['top5'] or 0) - (r['baseline']['top5'] or 0):+.4f}")
    print(f"  fraction of teacher gap closed: WR-agree "
          f"{_gap_closed(r['baseline'], r['distilled'], r['teacher'], 'wr_agreement_model')*100:.0f}%"
          f"   top5 {_gap_closed(r['baseline'], r['distilled'], r['teacher'], 'top5')*100:.0f}%")


def _spec(s: str) -> dict:
    p = [x.strip() for x in s.split(",")]
    out = {"parquet": p[0], "manifest": p[1], "scryfall": p[2]}
    if len(p) > 3:
        out["ratings"] = p[3]
    return out


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="Leaky-feature -> release-day distillation: a win-rate-augmented teacher distilled "
                    "into a release-day student. Each --train is 'parquet,manifest,scryfall,ratings'.")
    ap.add_argument("--train", action="append", required=True, metavar="PARQUET,MANIFEST,SCRYFALL,RATINGS")
    ap.add_argument("--holdout", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--holdout-ratings", required=True)
    ap.add_argument("--wr-field", default="ever_drawn_win_rate")
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-topk", type=int, default=0)
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--pool", default="set_transformer", choices=["mean", "set_transformer"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_leaky_distill(
        [_spec(t) for t in a.train], _spec(a.holdout), holdout_ratings=a.holdout_ratings,
        wr_field=a.wr_field, distill_lambda=a.distill_lambda, distill_temp=a.distill_temp,
        distill_topk=a.distill_topk, embedder=a.embedder, text=not a.no_text, pool=a.pool,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device, out_json=a.out_json,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
