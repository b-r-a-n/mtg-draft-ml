"""Step-0 go/no-go: does game_data give a less-confounded card-value signal than GIH-WR / IWD?

See docs/game-data-plan.md (Step 0). Cheap, one set, CPU:

  1. download a sample of <SET> per-game data (first N games),
  2. build the per-game deck-count matrix aligned to the set manifest,
  3. fit L2 logistic regression  won ~ decks + controls,  -> beta_c (marginal card value),
  4. compare beta_c to GIH-WR and IWD (Spearman/Pearson + biggest rank movers).

Decision rule (printed): beta ~ IWD (Spearman >= 0.95) -> STOP; meaningfully different -> PROCEED.

    uv run python scripts/game_value_gonogo.py --set DSK --sample-rows 80000 --l2 10
"""
from __future__ import annotations

import argparse
import json
import pathlib

from mtg_draft_ml.data.download import download_17lands_game
from mtg_draft_ml.data.game_preprocess import preprocess_game_set
from mtg_draft_ml.eval.game_value import compare_to_ratings, fit_card_values

STOP_THRESHOLD = 0.95  # Spearman(beta, IWD) at/above which game_data adds little de-confounding


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--event", dest="event_type", default="PremierDraft")
    ap.add_argument("--sample-rows", type=int, default=80_000)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--hf-dir", default="data/hf", help="dir holding manifests/ and ratings/")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cache-dir", default="data/game")
    ap.add_argument("--out", default="docs/results/game-data-value-model.json")
    a = ap.parse_args(argv)

    tag = f"{a.set_code}.{a.event_type}"
    hf = pathlib.Path(a.hf_dir)
    manifest = _resolve(hf / "manifests", tag)
    ratings = hf / "ratings" / f"{tag}.ratings.json"
    if not ratings.exists():
        raise SystemExit(f"missing ratings {ratings} — run the hf pull first")

    print(f"[1/4] download game_data sample ({a.sample_rows} games) for {tag} ...")
    csv = download_17lands_game(a.set_code, a.event_type, a.raw_dir, sample_rows=a.sample_rows)

    print(f"[2/4] preprocess -> deck matrix (manifest {manifest.name}) ...")
    npz = pathlib.Path(a.cache_dir) / f"game.{tag}.sample{a.sample_rows}.npz"
    data = preprocess_game_set(csv, manifest, out_npz=npz)
    X, y, C = data["X"], data["y"], data["C"]
    card_names = list(data["card_names"])
    print(f"      games={X.shape[0]}  cards={X.shape[1]}  present_in_game_data={int(data['n_cards_present'])}"
          f"  win_rate={float(y.mean()):.3f}")

    print(f"[3/4] fit L2 logistic regression (l2={a.l2}) ...")
    fit = fit_card_values(X, y, C, l2=a.l2)
    ctrl = dict(zip(list(data["control_names"]), fit["control_coef"].tolist()))
    print(f"      loss={fit['loss']:.4f}  train_acc={fit['train_acc']:.3f}  intercept={fit['intercept']:+.3f}")
    print(f"      control coefs (standardized): {{ {', '.join(f'{k}={v:+.3f}' for k, v in ctrl.items())} }}")

    print("[4/4] compare beta_c to GIH-WR and IWD ...")
    support = X.sum(axis=0)
    cmp = compare_to_ratings(fit["beta"], manifest, ratings, card_names=card_names, support=support)
    print(f"      Spearman(beta, GIH)={cmp['spearman_gih']:.3f}  Pearson={cmp['pearson_gih']:.3f}"
          f"  (n={cmp['n_compared_gih']})  well-sampled={cmp['spearman_gih_well']:.3f}")
    print(f"      Spearman(beta, IWD)={cmp['spearman_iwd']:.3f}  Pearson={cmp['pearson_iwd']:.3f}"
          f"  (n={cmp['n_compared_iwd']})  well-sampled={cmp['spearman_iwd_well']:.3f}"
          f"  (n_well={cmp['n_well_sampled']}, support>={int(cmp['min_support'])})")

    decision = "STOP" if cmp["spearman_iwd"] >= STOP_THRESHOLD else "PROCEED"
    print(f"\n  ==> DECISION: {decision}  (Spearman(beta,IWD)={cmp['spearman_iwd']:.3f} "
          f"vs stop-threshold {STOP_THRESHOLD})")

    print("\n  Promoted by beta vs IWD (well-sampled; genuinely impactful, IWD under-rates):")
    for m in cmp["promoted_vs_iwd"][:8]:
        print(f"    {m['name'][:34]:34s}  beta={m['beta']:+.3f}  iwd={m['iwd']:+.4f}  "
              f"gih={m['gih']:.3f}  n={m['support']}  drank={m['delta_rank']:+.2f}")
    print("  Demoted by beta vs IWD (well-sampled; ride-along, IWD over-rates):")
    for m in cmp["demoted_vs_iwd"][:8]:
        print(f"    {m['name'][:34]:34s}  beta={m['beta']:+.3f}  iwd={m['iwd']:+.4f}  "
              f"gih={m['gih']:.3f}  n={m['support']}  drank={m['delta_rank']:+.2f}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "set": a.set_code, "event": a.event_type, "sample_rows": a.sample_rows,
        "l2": a.l2, "n_games": int(X.shape[0]), "win_rate": float(y.mean()),
        "train_acc": fit["train_acc"], "intercept": fit["intercept"],
        "control_coef": ctrl, "decision": decision, "stop_threshold": STOP_THRESHOLD,
        **{k: cmp[k] for k in ("spearman_gih", "pearson_gih", "spearman_iwd", "pearson_iwd",
                               "spearman_gih_well", "spearman_iwd_well", "n_compared_gih",
                               "n_compared_iwd", "n_well_sampled", "min_support")},
        "promoted_vs_iwd": cmp["promoted_vs_iwd"], "demoted_vs_iwd": cmp["demoted_vs_iwd"],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    return payload


def _resolve(dir_path: pathlib.Path, tag: str) -> pathlib.Path:
    """Find <tag>.json or the sampled variant <tag>.sampleN.json under dir_path."""
    exact = dir_path / f"{tag}.json"
    if exact.exists():
        return exact
    cands = sorted(dir_path.glob(f"{tag}.sample*.json"))
    if cands:
        return cands[0]
    raise SystemExit(f"no manifest for {tag} under {dir_path}")


if __name__ == "__main__":
    main()
