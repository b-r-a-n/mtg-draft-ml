"""Step 1 — build the per-set game_data value field (`deck_value`) for all train sets.

See docs/game-data-plan.md (Step 1) + docs/results/game-data-value-model.md. For each set: download a
sample of per-game data, fit the L2 logistic value model (won ~ Sigma beta_c*deck_count_c + controls),
and emit `<SET>.<EVENT>.gamevalue.json` — a 17lands-ratings-shaped file with a `deck_value` field
(+ `deck_value_support`) that drops into `align_winrates(field="deck_value")` / `composite_card_quality`
and the WR-softmax teacher.

Validation per set: face-plausibility (top/bottom cards), correlation vs that set's GIH-WR/IWD, and
**split-half stability** (fit on the first vs second half of games — Spearman of the two beta vectors,
the signal-vs-noise check). Across sets: **reprint consistency** (cards in >=2 sets — do they get a
similar deck_value?). This is CPU/network work — no GPU needed.

    uv run python scripts/game_value_build.py --sample-rows 150000 --l2 30   # --sample-rows 0 = full CSV
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from mtg_draft_ml.data.download import download_17lands_game
from mtg_draft_ml.data.game_preprocess import preprocess_game_set
from mtg_draft_ml.eval.game_value import (
    _rank_corr,
    build_value_ratings,
    compare_to_ratings,
    fit_card_values,
)

DEFAULT_SETS = ["BLB", "OTJ", "WOE", "MKM", "DSK", "LCI", "MH3", "MOM"]


def _resolve_manifest(dir_path: pathlib.Path, tag: str) -> pathlib.Path:
    exact = dir_path / f"{tag}.json"
    if exact.exists():
        return exact
    cands = sorted(dir_path.glob(f"{tag}.sample*.json"))
    if cands:
        return cands[0]
    raise SystemExit(f"no manifest for {tag} under {dir_path}")


def build_set(set_code, event_type, sample_rows, l2, hf_dir, raw_dir, cache_dir, out_dir):
    tag = f"{set_code}.{event_type}"
    manifest = _resolve_manifest(pathlib.Path(hf_dir) / "manifests", tag)
    ratings = pathlib.Path(hf_dir) / "ratings" / f"{tag}.ratings.json"

    full = sample_rows <= 0  # --sample-rows 0 => the full game_data CSV
    csv = download_17lands_game(set_code, event_type, raw_dir,
                                sample_rows=None if full else sample_rows)
    npz = pathlib.Path(cache_dir) / (f"game.{tag}.full.npz" if full
                                     else f"game.{tag}.sample{sample_rows}.npz")
    data = preprocess_game_set(csv, manifest, out_npz=npz)
    X, y, C = data["X"], data["y"], data["C"]
    card_names = list(data["card_names"])
    support = X.sum(axis=0)

    fit = fit_card_values(X, y, C, l2=l2)
    beta = fit["beta"]

    # split-half stability: fit on the first vs second half of games, correlate the two betas
    h = X.shape[0] // 2
    b1 = fit_card_values(X[:h], y[:h], C[:h], l2=l2)["beta"]
    b2 = fit_card_values(X[h:], y[h:], C[h:], l2=l2)["beta"]
    well = support >= 1000.0
    split_rho = _rank_corr(np.where(well, b1, np.nan), np.where(well, b2, np.nan))

    cmp = compare_to_ratings(beta, manifest, ratings, card_names=card_names, support=support)
    recs = build_value_ratings(beta, support, card_names, set_code=set_code, event_type=event_type)

    out_path = pathlib.Path(out_dir) / f"{tag}.gamevalue.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(recs, indent=2))

    order = np.argsort(-beta)
    rated = [i for i in order if support[i] >= 1000.0]
    top = [(card_names[i], float(beta[i])) for i in rated[:5]]
    bot = [(card_names[i], float(beta[i])) for i in rated[-5:]]
    n_valued = sum(r["deck_value"] is not None for r in recs)
    print(f"  {set_code}: games={X.shape[0]} cards={X.shape[1]} valued={n_valued}  "
          f"win_rate={float(y.mean()):.3f}  train_acc={fit['train_acc']:.3f}")
    print(f"     Sp(beta,IWD)={cmp['spearman_iwd']:.3f} Sp(beta,GIH)={cmp['spearman_gih']:.3f} "
          f"split-half_rho={split_rho:.3f}")
    print(f"     top:  {', '.join(f'{n}({v:+.2f})' for n, v in top)}")
    print(f"     bot:  {', '.join(f'{n}({v:+.2f})' for n, v in bot)}")
    return {
        "set": set_code, "n_games": int(X.shape[0]), "n_valued": n_valued,
        "win_rate": float(y.mean()), "train_acc": fit["train_acc"],
        "spearman_iwd": cmp["spearman_iwd"], "spearman_gih": cmp["spearman_gih"],
        "spearman_iwd_well": cmp["spearman_iwd_well"], "split_half_rho": split_rho,
        "out": str(out_path),
        "value_by_name": {card_names[i]: float(beta[i]) for i in range(len(card_names))
                          if support[i] >= 1000.0},
    }


def reprint_consistency(summaries, min_sets=2):
    """For cards appearing (well-sampled) in >=min_sets sets, correlate their deck_value across the
    two sets where they're most/least valued — a cross-set stability check on reprints."""
    by_name: dict[str, list[float]] = {}
    for s in summaries:
        for name, v in s["value_by_name"].items():
            by_name.setdefault(name, []).append(v)
    shared = {n: vs for n, vs in by_name.items() if len(vs) >= min_sets}
    if len(shared) < 3:
        return {"n_reprints": len(shared), "spread_mean": None, "corr_first_pair": None}
    spreads = [max(vs) - min(vs) for vs in shared.values()]
    # correlation between each reprint's first two observed set-values (order-independent proxy)
    a = np.array([vs[0] for vs in shared.values()])
    b = np.array([vs[1] for vs in shared.values()])
    return {
        "n_reprints": len(shared),
        "spread_mean": float(np.mean(spreads)),
        "corr_first_pair": _rank_corr(a, b),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", default=",".join(DEFAULT_SETS))
    ap.add_argument("--event", dest="event_type", default="PremierDraft")
    ap.add_argument("--sample-rows", type=int, default=150_000,
                    help="rows per set; <=0 = full game_data CSV")
    ap.add_argument("--l2", type=float, default=30.0)
    ap.add_argument("--hf-dir", default="data/hf")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cache-dir", default="data/game")
    ap.add_argument("--out-dir", default="data/hf/gamevalue")
    ap.add_argument("--summary", default="docs/results/game-value-step1.json")
    a = ap.parse_args(argv)

    sets = [s.strip() for s in a.sets.split(",")]
    sample_label = "full" if a.sample_rows <= 0 else a.sample_rows
    print(f"Building game_data value field for {len(sets)} sets (sample={sample_label}, l2={a.l2})")
    summaries = []
    for s in sets:
        try:
            summaries.append(build_set(s, a.event_type, a.sample_rows, a.l2,
                                       a.hf_dir, a.raw_dir, a.cache_dir, a.out_dir))
        except Exception as e:  # one set's missing data shouldn't kill the run
            print(f"  {s}: SKIPPED ({type(e).__name__}: {e})")

    rep = reprint_consistency(summaries)
    rhos = [s["split_half_rho"] for s in summaries if s["split_half_rho"] == s["split_half_rho"]]
    iwd = [s["spearman_iwd"] for s in summaries]
    print("\n=== summary ===")
    print(f"  sets built: {len(summaries)}/{len(sets)}")
    print(f"  split-half stability rho: mean={np.mean(rhos):.3f} min={np.min(rhos):.3f} "
          f"max={np.max(rhos):.3f}")
    print(f"  Sp(beta,IWD): mean={np.mean(iwd):.3f}  (all < 0.95 => value field is its own signal)")
    print(f"  reprints (>=2 sets): n={rep['n_reprints']}  cross-set rho={rep['corr_first_pair']}  "
          f"mean spread={rep['spread_mean']}")

    out = pathlib.Path(a.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_rows": 0 if a.sample_rows <= 0 else a.sample_rows,  # 0 = full CSV
        "l2": a.l2, "event": a.event_type,
        "sets": [{k: v for k, v in s.items() if k != "value_by_name"} for s in summaries],
        "split_half_rho_mean": float(np.mean(rhos)),
        "spearman_iwd_mean": float(np.mean(iwd)), "reprint_consistency": rep,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote per-set fields to {a.out_dir}/  and summary to {out}")
    return payload


if __name__ == "__main__":
    main()
