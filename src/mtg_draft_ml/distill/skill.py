"""Train on GOOD PLAYERS, not the average drafter — attack both ceilings from the data side.

The ~0.58 top-1 ceiling is human *disagreement* (worst mid-pack, ~0.68 agreement among all drafters).
Good players disagree less (a cleaner label) and take higher-win-rate cards. 17lands tags every pick
with the drafter's skill (`user_game_win_rate_bucket`, `user_n_games_bucket`, `rank`), which our
parquets already carry — so we can simply *filter* to good players. The depth result de-risks this:
since more picks/set doesn't help, we can afford to drop ~80% of the data for higher-quality labels.

Experiment: 2×2 — {all players, good players} × {CE baseline, CE + composite-WR KD}, LOSO holdout.
Each model is evaluated on (a) the FULL holdout — top-1 here may *drop* for good-player models (they
pick like good players, not the average human, which is the point) and WR-agreement is the clean
"is it picking better" metric; and (b) the GOOD-PLAYER holdout — the fair top-1 test (did the model
get better at predicting *good* drafters).
"""
from __future__ import annotations

import json

import torch

from ..cards.content_table import build_content_matrix, build_multiset_content
from ..cards.text_embed import get_embedder
from ..data.dataset import DraftPickDataset, RemappedDataset, skill_filter_indices
from ..eval.generalization import evaluate_on_set, novel_mask_for_holdout
from ..eval.winrate import DEFAULT_QUALITY_FIELDS, align_winrates, composite_card_quality
from ..training.train import pick_device
from ..training.train_content import train_loop
from .ensemble import _build_model, _loaders
from .wr import WRSoftmaxTeacher


def _metrics(m: dict) -> dict:
    return {k: m.get(k) for k in ("top1", "top5", "wr_agreement_model", "avg_pick_wr_model",
                                  "wr_agreement_human", "avg_pick_wr_human", "n")}


def run_skill_experiment(
    train_specs: list[dict], holdout_spec: dict, *, holdout_ratings: str,
    min_winrate: float = 0.55, min_games: float = 50, ranks: set[str] | None = None,
    wr_field: str = "ever_drawn_win_rate", quality_fields: list[str] | None = None,
    wr_tau: float = 1.0, distill_lambda: float = 0.5, distill_temp: float = 1.0, distill_topk: int = 0,
    volume_control: bool = False,
    embedder: str = "all-MiniLM-L6-v2", text: bool = True,
    emb_dim: int = 256, enc_hidden: int = 512, enc_layers: int = 3, dropout: float = 0.1,
    pool: str = "set_transformer", n_heads: int = 4, n_sab: int = 1,
    warmup_frac: float = 0.1, grad_clip: float = 1.0,
    epochs: int = 10, batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.05,
    device: str = "auto", checkpoint_dir: str = "data/checkpoints", seed: int = 0,
    out_json: str | None = None,
) -> dict:
    """2x2: {all, good players} x {CE, CE+composite-WR}. Each evaluated on full + good-player holdout."""
    quality_fields = quality_fields if quality_fields is not None else list(DEFAULT_QUALITY_FIELDS)
    skill = {"min_winrate": min_winrate, "min_games": min_games, "ranks": ranks}
    emb = get_embedder(embedder)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=text)
    hmat, _ = build_content_matrix(holdout_spec["manifest"], holdout_spec["scryfall"],
                                   embedder=emb, text=text)
    assert gmat.shape[1] == hmat.shape[1], (gmat.shape, hmat.shape)
    dev = pick_device(device)

    bases = [(RemappedDataset(DraftPickDataset(s["parquet"]), l2g), s["parquet"])
             for s, l2g in zip(train_specs, l2gs)]
    rating_specs = [{"manifest": s["manifest"], "ratings": s["ratings"]} for s in train_specs]
    cscore, cmask = composite_card_quality(rating_specs, l2gs, ginfo["n_cards"], fields=quality_fields)
    teacher = WRSoftmaxTeacher(cscore, cmask, tau=wr_tau, device=dev)

    # report how much data the good-player filter keeps (confirms it actually fired)
    kept = sum(len(skill_filter_indices(s["parquet"], **skill)) for s in train_specs)
    total = sum(len(DraftPickDataset(s["parquet"])) for s in train_specs)
    kept_frac = kept / max(total, 1)
    print(f"device={dev}  good-player filter (winrate>={min_winrate}, games>={min_games}, ranks={ranks}): "
          f"keeps {kept}/{total} train picks ({kept_frac*100:.0f}%)  quality_fields={quality_fields}"
          + (f"  [volume_control: random arm at {kept_frac:.3f}]" if volume_control else ""))

    hp = dict(emb_dim=emb_dim, enc_hidden=enc_hidden, enc_layers=enc_layers, dropout=dropout,
              pool=pool, n_heads=n_heads, n_sab=n_sab)
    common = dict(checkpoint_dir=checkpoint_dir, checkpoint_every=0, n_cards=ginfo["n_cards"],
                  epochs=epochs, lr=lr, warmup_frac=warmup_frac, grad_clip=grad_clip)

    def _train(pop, arm):
        tag = f"{pop}_{arm}"
        print(f"\n-- {tag} (seed {seed}) --")
        torch.manual_seed(seed)
        # good = skill-filtered (~kept_frac of picks); rand = a RANDOM kept_frac subsample (same
        # volume, mixed quality — the control that isolates label quality from data quantity).
        tdl, vdl = _loaders(bases, val_frac, split_seed=seed, batch_size=batch_size,
                            train_frac=kept_frac if pop == "rand" else 1.0,
                            skill=skill if pop == "good" else None)
        m = _build_model(gmat, dev, **hp)
        extra = dict(teacher=teacher, distill_lambda=distill_lambda, distill_temp=distill_temp,
                     distill_topk=distill_topk) if arm == "comp" else {}
        train_loop(m, tdl, vdl, dev, tag=tag, ckpt_prefix=tag, **common, **extra)
        return m

    pops = ("all", "good", "rand") if volume_control else ("all", "good")
    models = {(pop, arm): _train(pop, arm) for pop in pops for arm in ("base", "comp")}

    novel = novel_mask_for_holdout(holdout_spec["manifest"], set(key_to_idx))
    card_wr = align_winrates(holdout_spec["manifest"], holdout_ratings, field=wr_field)
    good_holdout = skill_filter_indices(holdout_spec["parquet"], **skill)
    print(f"good-player holdout: {len(good_holdout)}/{len(DraftPickDataset(holdout_spec['parquet']))} picks")

    def ev(m, subset=None):
        return _metrics(evaluate_on_set(m, holdout_spec["parquet"], dev, novel_card=novel,
                                        card_wr=card_wr, subset_indices=subset))

    results = {
        "mode": "skill", "min_winrate": min_winrate, "min_games": min_games, "ranks": list(ranks) if ranks else None,
        "quality_fields": quality_fields, "n_train_sets": len(train_specs), "volume_control": volume_control,
        "kept_frac": kept_frac, "good_holdout_frac": len(good_holdout) / max(len(DraftPickDataset(holdout_spec["parquet"])), 1),
    }
    for (pop, arm), m in models.items():
        m.set_content(torch.from_numpy(hmat))
        results[f"{pop}_{arm}"] = {"full": ev(m), "good_holdout": ev(m, good_holdout)}
    _print_report(results)
    if out_json:
        json.dump(results, open(out_json, "w"), indent=2, default=float)
        print(f"\nwrote {out_json}")
    return results


def _print_report(r: dict):
    print("\n=== Good-players experiment report ===")
    print(f"filter winrate>={r['min_winrate']} games>={r['min_games']} ranks={r['ranks']}  "
          f"keeps {r['kept_frac']*100:.0f}% of train picks; good-player holdout = {r['good_holdout_frac']*100:.0f}%")
    print(f"  {'model':<16}{'WR-agree':>10}{'avgPWR':>9}{'top1(all)':>11}{'top1(good)':>12}")
    pops = ("all", "good", "rand") if r.get("volume_control") else ("all", "good")
    for pop in pops:
        for arm in ("base", "comp"):
            k = f"{pop}_{arm}"; f = r[k]["full"]; g = r[k]["good_holdout"]
            print(f"  {pop+'/'+arm:<16}{(f['wr_agreement_model'] or 0):>10.4f}{(f['avg_pick_wr_model'] or 0):>9.4f}"
                  f"{f['top1']:>11.4f}{g['top1']:>12.4f}")
    fa, ga = r["all_base"]["full"], r["all_base"]["good_holdout"]
    print(f"  {'human (all)':<16}{(fa['wr_agreement_human'] or 0):>10.4f}{(fa['avg_pick_wr_human'] or 0):>9.4f}")
    print(f"  {'human (good)':<16}{'':>10}{'':>9}{'':>11}{(ga['wr_agreement_human'] or 0):>12.4f}  <- WR-agree among good drafters")
    d_wr = (r["good_base"]["full"]["wr_agreement_model"] or 0) - (r["all_base"]["full"]["wr_agreement_model"] or 0)
    d_t1g = r["good_base"]["good_holdout"]["top1"] - r["all_base"]["good_holdout"]["top1"]
    print(f"  good - all (baseline):  WR-agree {d_wr:+.4f}   top1-on-good {d_t1g:+.4f}")
    if r.get("volume_control"):       # the clean test: same volume, only quality differs
        q_wr = (r["good_base"]["full"]["wr_agreement_model"] or 0) - (r["rand_base"]["full"]["wr_agreement_model"] or 0)
        q_t1 = r["good_base"]["good_holdout"]["top1"] - r["rand_base"]["good_holdout"]["top1"]
        print(f"  good - rand (SAME volume):  WR-agree {q_wr:+.4f}   top1-on-good {q_t1:+.4f}  "
              f"(>0 => good-player LABELS are genuinely cleaner, beyond volume)")


def _spec(s: str) -> dict:
    p = [x.strip() for x in s.split(",")]
    out = {"parquet": p[0], "manifest": p[1], "scryfall": p[2]}
    if len(p) > 3:
        out["ratings"] = p[3]
    return out


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Good-players-vs-all experiment (skill-filtered training).")
    ap.add_argument("--train", action="append", required=True, metavar="PARQUET,MANIFEST,SCRYFALL,RATINGS")
    ap.add_argument("--holdout", required=True, metavar="PARQUET,MANIFEST,SCRYFALL")
    ap.add_argument("--holdout-ratings", required=True)
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--min-games", type=float, default=50)
    ap.add_argument("--ranks", default=None, help="comma list e.g. mythic,diamond,platinum (optional)")
    ap.add_argument("--volume-control", action="store_true",
                    help="also train a random subsample at the good-player fraction (isolates quality vs quantity)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    run_skill_experiment(
        [_spec(t) for t in a.train], _spec(a.holdout), holdout_ratings=a.holdout_ratings,
        min_winrate=a.min_winrate, min_games=a.min_games,
        ranks=set(s.strip() for s in a.ranks.split(",")) if a.ranks else None,
        volume_control=a.volume_control,
        epochs=a.epochs, batch_size=a.batch_size, device=a.device, out_json=a.out_json,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
