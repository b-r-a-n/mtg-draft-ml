"""Aggregate per-set decensor_curve.py results into ONE cross-set curve verdict.

A single set's LR chi2 (3-10) is within noise; the real (a) curve-matters / (b) flat conclusion needs
the curve descriptor to be CONSISTENTLY positive across >=5 sets (the workflow synthesis's bar). This
reads docs/results/decensor/<SET>.json and reports, per descriptor, the mean marginal coef + chi2, how
many sets clear chi2>3.84, and the (power-residualized) banded gap consistency.

    uv run python scripts/decensor_aggregate.py --dir docs/results/decensor
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="docs/results/decensor")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{a.dir}/*.json"))]
    if not rows:
        raise SystemExit(f"no per-set JSONs under {a.dir}")
    sets = [r["set"] for r in rows]
    n = len(rows)
    print(f"cross-set de-censoring aggregate over {n} sets: {sets}\n")
    print(f"  {'descriptor':<10}{'mean coef/SD':>13}{'mean chi2':>11}{'chi2>3.84':>11}"
          f"{'mean band':>11}{'band>0':>8}")
    summary = {}
    for d in ("cast", "pip_conc", "avg_cmc", "n_lands"):
        coefs = [r["marginal"]["descriptors"][d]["coef_after_controls"] for r in rows]
        chis = [r["marginal"]["descriptors"][d]["lr_chi2_1dof"] for r in rows]
        bands = [g for r in rows if d in r.get("banded", {})
                 for g in [r["banded"][d]["mean_gap"]] if g == g]   # drop NaN (cells too small)
        n_sig = sum(c > 3.84 for c in chis)
        n_bpos = sum(b > 0 for b in bands)
        summary[d] = {
            "mean_coef": statistics.mean(coefs), "mean_chi2": statistics.mean(chis),
            "n_sig": n_sig, "n_sets": n,
            "mean_band": statistics.mean(bands) if bands else None,
            "n_band_pos": n_bpos, "n_band": len(bands),
            "coef_sign_consistent": all(c > 0 for c in coefs) or all(c < 0 for c in coefs),
        }
        bg = f"{summary[d]['mean_band']:+.4f}" if bands else "  —"
        print(f"  {d:<10}{summary[d]['mean_coef']:>13.4f}{summary[d]['mean_chi2']:>11.1f}"
              f"{f'{n_sig}/{n}':>11}{bg:>11}{f'{n_bpos}/{len(bands)}':>8}")

    ca = summary["cast"]
    thr = max(3, 0.6 * n)
    consistent = (ca["n_sig"] >= thr and ca["mean_coef"] > 0 and ca["coef_sign_consistent"]
                  and ca["mean_band"] is not None and ca["mean_band"] > 0.003
                  and ca["n_band_pos"] >= 0.6 * ca["n_band"])
    verdict = ("(a) CURVE MATTERS once de-censored: castability has a consistent positive marginal effect "
               "across sets -> VALIDATES castability.py against real win/loss AND the bounded-out "
               "'pick-time curve is valueless' verdict is no longer safe (run the picker arm to fully reopen)."
               if consistent else
               "(b) FLAT / inconsistent: castability's marginal effect is noise-level or not sign-consistent "
               "across sets -> curve is dominated by card power even de-censored; castability stays a "
               "build-time tool (it still helps the deckbuilder, per play_prob's +0.049).")
    print(f"\nCROSS-SET VERDICT ({ca['n_sig']}/{n} sets sig, mean coef {ca['mean_coef']:+.4f}, "
          f"band {ca['n_band_pos']}/{ca['n_band']} positive):\n  {verdict}")

    if a.out:
        json.dump({"sets": sets, "summary": summary, "consistent_a": consistent, "verdict": verdict},
                  open(a.out, "w"), indent=2, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
