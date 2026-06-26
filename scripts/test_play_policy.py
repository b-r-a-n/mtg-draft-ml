"""Does folding P(played | pool) into the pick policy improve drafting?

Capstone test for the buildability signal (docs/results/play-prob.md): at each pick, combine the draft
model's logit with the learned P(played | pool) — effective(c) = model_logit(c) + λ·logit P(played) —
so cards unlikely to make the deck (off-color, redundant) are down-weighted. Replay held-out drafts and
score each policy's pool by the outcome eval's constrained 2-color+curve deck-WR, vs the plain model and
humans. Diagnostic: pool color concentration (does it draft more coherent pools? does it degenerate?).

    uv run python scripts/test_play_policy.py --set DSK --n-drafts 600
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from mtg_draft_ml.eval.outcome import greedy_pick, human_pick, load_drafts, replay, run_eval
from mtg_draft_ml.eval.play_prob import train_play_model, train_play_model_partial
from mtg_draft_ml.eval.winrate import align_winrates


def _onnx_logits(session, mp, mk):
    def f(pool, pack):
        pa = np.zeros((1, mp), np.int64); pm = np.zeros((1, mp), bool)
        for j, c in enumerate(pool[:mp]):
            pa[0, j] = c; pm[0, j] = True
        ka = np.zeros((1, mk), np.int64); km = np.zeros((1, mk), bool)
        for j, c in enumerate(pack[:mk]):
            ka[0, j] = c; km[0, j] = True
        return session.run(None, {"pool": pa, "pool_mask": pm, "pack": ka, "pack_mask": km})[0][0, :len(pack)]
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--n-drafts", type=int, default=600)
    ap.add_argument("--lambdas", default="0,0.5,1,2")
    ap.add_argument("--partial", action="store_true",
                    help="train P(played|PARTIAL pool) so it's in-distribution at pick time (the fix "
                         "for the OOD failure of the full-pool model). See play-prob-pick-time.md.")
    ap.add_argument("--hf-dir", default="data/hf")
    ap.add_argument("--webapp-dir", default="webapp")
    ap.add_argument("--out", default="docs/results/play-policy.json")
    a = ap.parse_args(argv)
    import onnxruntime as ort

    hf = pathlib.Path(a.hf_dir)
    man = next(iter((hf / "manifests").glob(f"{a.set_code}.PremierDraft.sample*.json")))
    parquet = next(iter((hf / "draft").glob(f"{a.set_code}.PremierDraft.sample*.parquet")))
    csv = str(next(iter(pathlib.Path("data/raw").glob(f"game.{a.set_code}.*.csv"))))
    cards = json.load(open(f"{a.webapp_dir}/data/{a.set_code}.cards.json"))["cards"]
    meta = json.load(open(f"{a.webapp_dir}/data/{a.set_code}.meta.json"))
    n = len(json.load(open(man))["cards"])
    ci = [""] * n; cmc = np.zeros(n); types = ["spell"] * n; beta = np.full(n, np.nan)
    for c in cards:
        ci[c["i"]] = c.get("ci", "C"); cmc[c["i"]] = c.get("cmc", 0) or 0; types[c["i"]] = c.get("t", "spell")
        if c["deck_value"] is not None:
            beta[c["i"]] = c["deck_value"]

    print(f"training P(played | {'PARTIAL' if a.partial else 'full'} pool) …")
    pm = (train_play_model_partial if a.partial else train_play_model)(csv, str(man), cards)
    sess = ort.InferenceSession(f"{a.webapp_dir}/model/{a.set_code}.onnx")
    logit_fn = _onnx_logits(sess, meta["max_pool"], meta["max_pack"])
    gih = align_winrates(str(man), f"{hf}/ratings/{a.set_code}.PremierDraft.ratings.json",
                         field="ever_drawn_win_rate")
    drafts = load_drafts(parquet, limit=a.n_drafts)

    def build_pick(lam):
        def fn(pool, pack, step):
            lg = np.asarray(logit_fn(pool, pack), float)
            if lam:
                p = np.clip(pm.probs(pool, pack), 0.02, 0.98)
                lg = lg + lam * np.log(p / (1 - p))
            return pack[int(np.argmax(lg))]
        return fn

    lams = [float(x) for x in a.lambdas.split(",")]
    policies = {"human": human_pick}
    for lam in lams:
        policies[f"model+build@{lam}" if lam else "model"] = build_pick(lam)
    policies["gih_greedy"] = greedy_pick(gih)
    res = run_eval(drafts, policies, beta, intercept=0.0, n_spells=23, ci=ci, cmc=cmc, types=types)

    print(f"\n=== outcome eval: P(played|pool) reweighting (DSK, {res['n_drafts']} drafts, "
          f"constrained 2-color+curve deck) ===")
    print(f"  {'policy':<18}{'deck-WR':>9}{'Δ vs model':>12}{'Δ vs human':>12}{'top2-color%':>13}")
    base = res["mean"].get("model")
    for name in policies:
        # color concentration: fraction of the drafted pool in its top-2 colors
        pools = [replay(d, policies[name]) for d in drafts[:120]]
        conc = np.mean([_top2_frac(p, ci) for p in pools])
        dm = res["mean"][name] - base if base is not None else None
        dh = res.get("delta_vs_human", {}).get(name)
        print(f"  {name:<18}{res['mean'][name]:>9.4f}"
              f"{(f'{dm:+.4f}' if dm is not None else '—'):>12}"
              f"{(f'{dh:+.4f}' if dh is not None else '—'):>12}{conc*100:>12.0f}%")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps({"set": a.set_code, "lambdas": lams, **res}, indent=2, default=float))
    print(f"\nwrote {a.out}")


def _top2_frac(pool, ci):
    from collections import Counter
    cc = Counter()
    for i in pool:
        for L in set(ci[i]) - {"C"}:
            cc[L] += 1
    tot = sum(cc.values()) or 1
    return sum(v for _, v in cc.most_common(2)) / tot


if __name__ == "__main__":
    main()
