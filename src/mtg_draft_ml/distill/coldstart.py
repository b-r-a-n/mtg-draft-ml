"""Cold-start distillation experiment: does an LLM teacher help before 17lands data exists?

The setup reuses the leave-one-set-out frame (`docs/roadmap.md` Phase 1): train the content model on
N sets, then treat a held-out set as "the new set on release day". We compare three pick policies on
that held-out set, all on the SAME trained model:

  1. no-data baseline   — pure content-encoder zero-shot (aggressiveness/blend = 0)
  2. LLM cold-start     — blend the LLM teacher's per-card quality into the pick logits
  3. real-data oracle   — blend the REAL 17lands quality (the upper bound the LLM is chasing)

We report top-1, WR-agreement, and avg-pick-WR (all measured against the real 17lands signal, which
we DO have for the held-out set — that's what makes this a clean offline test), plus the rank
correlation between teacher quality and real quality. The headline number is how much of the
(oracle − baseline) gap the LLM teacher closes. If it closes a meaningful fraction, day-one
deployment on a new set is justified; if not, the experiment cleanly rules it out.

This is self-contained (it reuses the public building blocks from `eval`/`cards`/`training` rather
than `run_loso`, which doesn't expose the trained model) so it adds no risk to the locked Phase-0..3
code path.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, RemappedDataset, collate_picks
from ..eval.generalization import evaluate_on_set, novel_mask_for_holdout
from ..eval.winrate import align_winrates, zscore_ignore_nan
from ..models.draft_model import ContentDraftModel
from ..training.train import draft_level_split, pick_device
from ..training.train_content import train_loop
from .teacher import RATING_FIELD


def _quality_tensor(manifest_path, ratings_path, field, device):
    """Aligned, z-scored, NaN-filled per-card quality (manifest-index order) for the blend dial.

    Z-scoring (ignoring missing) makes `blend_alpha` mean the same thing across quality sources, so
    the teacher and the real-data oracle are compared on one alpha grid (mirrors deploy.Drafter's
    scale-invariant dial). Missing cards get 0 = the post-zscore mean = neutral.
    """
    raw = align_winrates(manifest_path, ratings_path, field=field)   # [n_cards], NaN where missing
    std = np.nan_to_num(zscore_ignore_nan(raw), nan=0.0)
    return torch.as_tensor(std, dtype=torch.float32, device=device), raw


def _rank_corr(a: np.ndarray, b: np.ndarray) -> dict:
    """Spearman + Pearson over cards rated by BOTH sources (raw, pre-zscore)."""
    m = ~(np.isnan(a) | np.isnan(b))
    n = int(m.sum())
    if n < 3:
        return {"n": n, "spearman": float("nan"), "pearson": float("nan")}
    x, y = a[m], b[m]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return {"n": n,
            "spearman": float(np.corrcoef(rx, ry)[0, 1]),
            "pearson": float(np.corrcoef(x, y)[0, 1])}


def run_coldstart(
    train_specs: list[dict], holdout_spec: dict, coldstart_ratings: str, holdout_ratings: str,
    *, wr_field: str = "ever_drawn_win_rate", blend_alphas: list[float] | None = None,
    embedder: str = "all-MiniLM-L6-v2", text: bool = True,
    emb_dim: int = 256, enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
    pool: str = "set_transformer", n_heads: int = 4, n_sab: int = 1,
    warmup_frac: float = 0.1, grad_clip: float = 1.0,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
    device: str = "auto", checkpoint_dir: str = "data/checkpoints", seed: int = 0,
    out_json: str | None = None,
) -> dict:
    """Train on train_specs, then compare baseline / LLM-coldstart / real-oracle picks on holdout.

    `coldstart_ratings` is the teacher's ratings JSON (field=RATING_FIELD); `holdout_ratings` is the
    real 17lands ratings for the held-out set (field=`wr_field`) — used both as the upper-bound blend
    and as the truth the metrics are scored against.
    """
    blend_alphas = blend_alphas if blend_alphas is not None else [0.5, 1.0, 2.0, 4.0]
    emb = get_embedder(embedder)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=text)
    hmat, hinfo = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                       embedder=emb, text=text)
    assert gmat.shape[1] == hmat.shape[1], (gmat.shape, hmat.shape)

    torch.manual_seed(seed)
    dev = pick_device(device)
    print(f"device={dev}  train matrix {tuple(gmat.shape)} from {ginfo['n_sets']} sets")

    train_subsets, val_subsets = [], []
    for spec, l2g in zip(train_specs, l2gs):
        rds = RemappedDataset(DraftPickDataset(spec["parquet"]), l2g)
        tr, va = draft_level_split(spec["parquet"], val_frac, seed)
        train_subsets.append(Subset(rds, tr))
        val_subsets.append(Subset(rds, va))
    train_dl = DataLoader(ConcatDataset(train_subsets), batch_size=batch_size, shuffle=True,
                          collate_fn=collate_picks)
    val_dl = DataLoader(ConcatDataset(val_subsets), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_picks)

    model = ContentDraftModel(torch.from_numpy(gmat), emb_dim=emb_dim, enc_hidden=enc_hidden,
                              enc_layers=enc_layers, dropout=dropout, pool=pool,
                              n_heads=n_heads, n_sab=n_sab).to(dev)
    best = train_loop(model, train_dl, val_dl, dev, epochs=epochs, lr=lr,
                      checkpoint_dir=checkpoint_dir, checkpoint_every=0,
                      n_cards=ginfo["n_cards"], tag="coldstart", ckpt_prefix="coldstart",
                      warmup_frac=warmup_frac, grad_clip=grad_clip)

    # Retarget the content head to the held-out set; everything below is its release-day evaluation.
    model.set_content(torch.from_numpy(hmat))
    novel = novel_mask_for_holdout(holdout_spec["manifest"], set(key_to_idx))
    real_wr = align_winrates(holdout_spec["manifest"], holdout_ratings, field=wr_field)  # the truth
    teacher_q, teacher_raw = _quality_tensor(holdout_spec["manifest"], coldstart_ratings,
                                             RATING_FIELD, dev)
    oracle_q, _ = _quality_tensor(holdout_spec["manifest"], holdout_ratings, wr_field, dev)

    def ev(quality=None, alpha=0.0):
        m = evaluate_on_set(model, holdout_spec["parquet"], dev, novel_card=novel,
                            card_wr=real_wr, quality=quality, blend_alpha=alpha)
        return {"alpha": alpha, "top1": m["top1"], "novel_top1": m.get("novel_top1"),
                "wr_agreement": m.get("wr_agreement_model"), "avg_pick_wr": m.get("avg_pick_wr_model")}

    baseline = ev()                                                  # alpha=0, no quality blend
    coldstart = [ev(teacher_q, a) for a in blend_alphas]             # LLM teacher signal
    oracle = [ev(oracle_q, a) for a in blend_alphas]                 # real-data upper bound
    corr = _rank_corr(teacher_raw, real_wr)

    results = {
        "mode": "coldstart", "embedder": embedder, "pool": pool,
        "n_train_sets": len(train_specs), "train_cards": ginfo["n_cards"],
        "holdout": {"parquet": holdout_spec["parquet"], "n_cards": hinfo["n_cards"],
                    "frac_cards_novel": float(novel.float().mean()),
                    "human_wr_agreement": None},
        "teacher_vs_real": corr,
        "baseline": baseline, "coldstart_sweep": coldstart, "oracle_sweep": oracle,
        "train_in_set_top1": best["top1"],
    }
    # human WR-agreement reference: the imitation floor the model+teacher must beat on winning picks
    href = evaluate_on_set(model, holdout_spec["parquet"], dev, card_wr=real_wr)
    results["holdout"]["human_wr_agreement"] = href.get("wr_agreement_human")
    _print_coldstart(results)
    if out_json:
        json.dump(results, open(out_json, "w"), indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _best(sweep: list[dict], key: str) -> dict:
    vals = [s for s in sweep if s.get(key) is not None]
    return max(vals, key=lambda s: s[key]) if vals else {"alpha": None, key: float("nan")}


def _print_coldstart(r: dict):
    h, c = r["holdout"], r["teacher_vs_real"]
    b = r["baseline"]
    print("\n=== Cold-start distillation report ===")
    print(f"embedder={r['embedder']} pool={r['pool']} train_sets={r['n_train_sets']} "
          f"train_cards={r['train_cards']}")
    print(f"held-out new set: {h['n_cards']} cards ({h['frac_cards_novel']*100:.0f}% novel)")
    print(f"teacher vs real 17lands: spearman={c['spearman']:.3f} pearson={c['pearson']:.3f} "
          f"(n={c['n']} cards rated by both)")
    print(f"\n  {'policy':<22}{'best_alpha':>11}{'WR-agree':>10}{'avg_pick_WR':>13}{'top1':>8}")
    bw = _best(r["coldstart_sweep"], "wr_agreement")
    ow = _best(r["oracle_sweep"], "wr_agreement")
    print(f"  {'no-data baseline':<22}{'0':>11}{b['wr_agreement']:>10.4f}"
          f"{b['avg_pick_wr']:>13.4f}{b['top1']:>8.4f}")
    print(f"  {'LLM cold-start':<22}{str(bw['alpha']):>11}{bw['wr_agreement']:>10.4f}"
          f"{bw['avg_pick_wr']:>13.4f}{bw['top1']:>8.4f}")
    print(f"  {'real-data oracle':<22}{str(ow['alpha']):>11}{ow['wr_agreement']:>10.4f}"
          f"{ow['avg_pick_wr']:>13.4f}{ow['top1']:>8.4f}")
    gap = ow["wr_agreement"] - b["wr_agreement"]
    closed = (bw["wr_agreement"] - b["wr_agreement"]) / gap if gap > 1e-9 else float("nan")
    print(f"  human WR-agreement floor: {h['human_wr_agreement']:.4f}")
    print(f"  => LLM closes {closed*100:.0f}% of the oracle gap on WR-agreement "
          f"(oracle - baseline = {gap:+.4f})")


def _spec(s: str) -> dict:
    parts = [p.strip() for p in s.split(",")]
    return {"parquet": parts[0], "manifest": parts[1], "scryfall": parts[2]}


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="Cold-start distillation: train on --train sets, evaluate baseline vs "
                    "LLM-teacher vs real-data blends on --holdout. Specs are 'parquet,manifest,scryfall'.")
    ap.add_argument("--train", action="append", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--holdout", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--coldstart-ratings", required=True, help="LLM-teacher ratings JSON (build via distill.teacher)")
    ap.add_argument("--holdout-ratings", required=True, help="real 17lands ratings JSON for the holdout")
    ap.add_argument("--wr-field", default="ever_drawn_win_rate")
    ap.add_argument("--blend-alphas", default="0.5,1,2,4")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--pool", default="set_transformer", choices=["mean", "set_transformer"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_coldstart(
        [_spec(t) for t in a.train], _spec(a.holdout), a.coldstart_ratings, a.holdout_ratings,
        wr_field=a.wr_field, blend_alphas=[float(x) for x in a.blend_alphas.split(",")],
        embedder=a.embedder, text=not a.no_text, pool=a.pool,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device, out_json=a.out_json,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
