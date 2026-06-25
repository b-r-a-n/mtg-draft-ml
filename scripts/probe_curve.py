"""Mechanistic probe: does the draft model do contextual curve-completion?

A causal intervention on the deployed ONNX model (better than weight-staring): hold a pack fixed (a
cheap creature + an expensive card of the SAME color, **beta-matched** so card value isn't the
tiebreaker), then swap only the POOL between top-heavy (expensive cards) and low-curve (cheap cards),
both drawn from **mid-beta filler** so the pools are power-matched and differ only in CMC. Measure how
the model's preference for the cheap card shifts.

  shift = [score(cheap) - score(exp) | top-heavy pool] - [... | low-curve pool]
  shift > 0  =>  curve-aware (prefers the cheap card more when the pool is top-heavy / curve-starved)

Finding (DSK, deployed 20-set model): shift ≈ -0.024 logit, 39% positive — the model is curve-BLIND
(near-zero, slightly negative). It does not fix bad curves; its pool-conditioning is color/power/synergy,
not mana curve. This is why the outcome eval (eval/outcome.py) needs an EXTERNAL 2-color+curve build.

    uv run python scripts/probe_curve.py --set DSK
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--webapp-dir", default="webapp")
    ap.add_argument("--beta-match", type=float, default=0.04, help="max |Δβ| for the pack pair")
    ap.add_argument("--pool-size", type=int, default=5)
    a = ap.parse_args(argv)
    import onnxruntime as ort

    cards = json.load(open(f"{a.webapp_dir}/data/{a.set_code}.cards.json"))["cards"]
    meta = json.load(open(f"{a.webapp_dir}/data/{a.set_code}.meta.json"))
    mp, mk = meta["max_pool"], meta["max_pack"]
    sess = ort.InferenceSession(f"{a.webapp_dir}/model/{a.set_code}.onnx")
    beta = {c["i"]: c["deck_value"] for c in cards if c["deck_value"] is not None}
    cmc = {c["i"]: (c["cmc"] or 0) for c in cards}

    def logits(pool, pack):
        pa = np.zeros((1, mp), np.int64); pm = np.zeros((1, mp), bool)
        for j, c in enumerate(pool[:mp]):
            pa[0, j] = c; pm[0, j] = True
        ka = np.zeros((1, mk), np.int64); km = np.zeros((1, mk), bool)
        for j, c in enumerate(pack[:mk]):
            ka[0, j] = c; km[0, j] = True
        return sess.run(None, {"pool": pa, "pool_mask": pm, "pack": ka, "pack_mask": km})[0][0, :len(pack)]

    deltas, lo_b, hi_b = [], [], []
    for color in ["W", "U", "B", "R", "G"]:
        oc = [c["i"] for c in cards if c["ci"] == color and c["i"] in beta]
        if len(oc) < 10:
            continue
        p20, p80 = np.percentile([beta[i] for i in oc], [20, 80])
        filler = [i for i in oc if p20 <= beta[i] <= p80]           # mid-beta filler (no bombs/duds)
        fl_cheap = [i for i in filler if cmc[i] <= 2]; fl_exp = [i for i in filler if cmc[i] >= 4]
        if len(fl_cheap) < 4 or len(fl_exp) < 4:
            continue
        for ci_ in (i for i in oc if cmc[i] <= 2):
            for ei in (i for i in oc if cmc[i] >= 4):
                if abs(beta[ci_] - beta[ei]) > a.beta_match:        # beta-matched pack: curve is the tiebreaker
                    continue
                low = [x for x in fl_cheap if x != ci_][:a.pool_size]
                top = [x for x in fl_exp if x != ei][:a.pool_size]
                if len(low) < 4 or len(top) < 4:
                    continue
                lt = logits(top, [ci_, ei]); ll = logits(low, [ci_, ei])
                deltas.append((lt[0] - lt[1]) - (ll[0] - ll[1]))    # >0 => prefers cheap more when top-heavy
                lo_b.append(np.mean([beta[x] for x in low])); hi_b.append(np.mean([beta[x] for x in top]))

    d = np.array(deltas)
    se = d.std() / np.sqrt(len(d))
    print(f"set={a.set_code}  n beta-matched probes={len(d)}")
    print(f"power control — mean pool beta: low-curve {np.mean(lo_b):+.3f} vs top-heavy {np.mean(hi_b):+.3f}")
    print(f"shift toward CHEAP card when pool is top-heavy: {d.mean():+.4f} logit  ({d.mean()/se:+.1f}σ)")
    print(f"fraction favoring the cheap card more under top-heavy pool: {(d > 0).mean()*100:.0f}%  "
          f"(50% = curve-blind)")
    verdict = "curve-AWARE" if d.mean() > 0.05 else ("curve-BLIND" if abs(d.mean()) < 0.05 else "ANTI-curve")
    print(f"=> {verdict}")


if __name__ == "__main__":
    main()
