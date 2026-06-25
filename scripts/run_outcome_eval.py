"""Outcome eval — do the model's drafted decks actually win more (by the game_data deck-value model)?

Replays held-out drafts: each policy drafts a seat (on the recorded pack sequence), and the resulting
pool is scored by the game_data logistic deck-value model (estimated deck win rate). Compares the
DEPLOYED webapp model (trained on the GIH-composite, NOT on deck_value — so this is non-circular) to
the humans who actually drafted those seats, plus references:
  - human            the recorded picks (baseline)
  - model            the webapp ONNX model (argmax pick)
  - gih_greedy       always take the highest GIH-WR card in the pack (the confounded-rating policy)
  - deckvalue_greedy always take the highest deck_value card (oracle on the scorer — an upper bound)
  - random           random legal pick (floor)

CPU only — uses the exported webapp model (no retrain).

    uv run python scripts/run_outcome_eval.py --set DSK --n-drafts 800
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from mtg_draft_ml.eval.game_value import fit_card_values
from mtg_draft_ml.eval.outcome import greedy_pick, human_pick, load_drafts, random_pick, run_eval
from mtg_draft_ml.eval.winrate import align_winrates


def _onnx_pick(session, max_pool, max_pack):
    def fn(pool, pack, step):
        pa = np.zeros((1, max_pool), np.int64); pm = np.zeros((1, max_pool), bool)
        for j, c in enumerate(pool[:max_pool]):
            pa[0, j] = c; pm[0, j] = True
        ka = np.zeros((1, max_pack), np.int64); km = np.zeros((1, max_pack), bool)
        for j, c in enumerate(pack[:max_pack]):
            ka[0, j] = c; km[0, j] = True
        o = session.run(None, {"pool": pa, "pool_mask": pm, "pack": ka, "pack_mask": km})[0][0, :len(pack)]
        return pack[int(np.argmax(o))]
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--n-drafts", type=int, default=800)
    ap.add_argument("--n-spells", type=int, default=23)
    ap.add_argument("--l2", type=float, default=30.0)
    ap.add_argument("--hf-dir", default="data/hf")
    ap.add_argument("--webapp-dir", default="webapp")
    ap.add_argument("--game-npz", default=None)
    ap.add_argument("--out", default="docs/results/outcome-eval.json")
    a = ap.parse_args(argv)
    import onnxruntime as ort

    hf = pathlib.Path(a.hf_dir)
    manifest = next(iter((hf / "manifests").glob(f"{a.set_code}.PremierDraft.sample*.json")))
    ratings = hf / "ratings" / f"{a.set_code}.PremierDraft.ratings.json"
    parquet = next(iter((hf / "draft").glob(f"{a.set_code}.PremierDraft.sample*.parquet")))
    npz = a.game_npz or str(next(iter(pathlib.Path("data/game").glob(f"game.{a.set_code}.*.npz"))))
    meta = json.load(open(f"{a.webapp_dir}/data/{a.set_code}.meta.json"))

    # 1. (re)fit the game_data deck-value model -> beta (+ intercept) in manifest-index order
    z = np.load(npz, allow_pickle=True)
    fit = fit_card_values(z["X"], z["y"], z["C"], l2=a.l2)
    beta = fit["beta"].astype(np.float64)
    intercept = fit["intercept"]
    print(f"deck-value fit: {len(beta)} cards, train_acc={fit['train_acc']:.3f}, intercept={intercept:+.3f}")

    # 2. the deployed model + the GIH rating
    sess = ort.InferenceSession(f"{a.webapp_dir}/model/{a.set_code}.onnx")
    gih = align_winrates(manifest, ratings, field="ever_drawn_win_rate")
    drafts = load_drafts(parquet, limit=a.n_drafts)
    print(f"replaying {len(drafts)} drafts on {a.set_code} (model trained on {len(meta['train_sets'])} "
          f"sets, teacher={meta['teacher']}, in-distribution={meta['target_in_train']})")

    policies = {
        "human": human_pick,
        "model": _onnx_pick(sess, meta["max_pool"], meta["max_pack"]),
        "gih_greedy": greedy_pick(gih),
        "deckvalue_greedy": greedy_pick(beta),
        "random": random_pick(np.random.default_rng(0)),
    }
    res = run_eval(drafts, policies, beta, intercept=0.0, n_spells=a.n_spells)

    # report: deck-value logit (sum top-N beta) is the honest scale; sigma is a monotone "win-ish" view
    print(f"\n=== estimated deck-WR ({a.set_code}, best-{a.n_spells} spells by deck_value, "
          f"{res['n_drafts']} drafts) ===")
    print(f"  {'policy':<18}{'est deck-WR':>12}{'Δ vs human':>12}{'beats human':>14}")
    order = ["deckvalue_greedy", "model", "gih_greedy", "human", "random"]
    for k in order:
        if k not in res["mean"]:
            continue
        d = res.get("delta_vs_human", {}).get(k)
        b = res.get("beats_human_frac", {}).get(k)
        ds = f"{d:+.4f}" if d is not None else "—"
        bs = f"{b*100:.0f}%" if b is not None else "—"
        print(f"  {k:<18}{res['mean'][k]:>12.4f}{ds:>12}{bs:>14}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": a.set_code, "n_drafts": res["n_drafts"], "n_spells": a.n_spells,
                   "train_sets": len(meta["train_sets"]), "teacher": meta["teacher"],
                   "in_distribution": meta["target_in_train"], **res}, indent=2, default=float))
    print(f"\nwrote {out}")
    return res


if __name__ == "__main__":
    main()
