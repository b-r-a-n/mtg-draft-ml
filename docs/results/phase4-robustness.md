# Phase 4 — Robustness (seeds × rotating holdout) + the standardization A/B

Tightening the single-seed, single-holdout Phase-1/2/3 numbers before any GPU run.

**Setup:** best config — `set_transformer` + `ce` + aux-WR (λ=1.0), MiniLM text encoder. Every set
standardized to a 60k-pick sample. **3 seeds × 5 rotating holdouts = 15 runs**, 10 epochs each.
Run with `scripts/rotate_seeds.py` (add `--standardize` for the A/B).

## Robustness of the current recipe (raw features)

| Held-out | top-1 (mean ± std over seeds) | WR-agreement (model) |
|---|---|---|
| DSK | 0.5746 ± 0.0009 | 0.2568 |
| BLB | 0.5570 ± 0.0007 | 0.2896 |
| OTJ | 0.5333 ± 0.0023 | 0.2479 |
| MKM | 0.5281 ± 0.0016 | 0.2375 |
| WOE | 0.5096 ± 0.0023 | 0.2331 |
| **Overall (n=15)** | **0.5405 ± 0.0229** | model 0.2529 / human 0.2893 |

**Findings:**
1. **Seeds barely matter** — per-holdout std ≈ 0.001–0.003 (±0.1–0.3 pt). The model is stable; our
   single-seed numbers were trustworthy.
2. **The real spread is *which set* is held out** (overall std ±2.3 pt, range 0.510–0.575) — an
   order of magnitude larger than seed noise. Some sets are easier to generalize *to* (DSK, BLB)
   than others (WOE).
3. **The flagship 0.574 was DSK-specific**, and DSK is one of the *easiest* holdouts (reproduced
   exactly: 0.5746 ± 0.0009). The honest, representative number is **0.540 ± 0.023 (rotated)**, with
   DSK a favorable case. Rotating the holdout matters: a single holdout can flatter by ±2–3 pt.

## A/B: does feature standardization help? (No.)

Per-column z-score of the content matrix (train stats applied to holdout), same 15-run protocol.

| Held-out | raw | standardized | Δ |
|---|---|---|---|
| DSK | 0.5746 | 0.5524 | −0.022 |
| BLB | 0.5570 | 0.5484 | −0.009 |
| OTJ | 0.5333 | 0.5140 | −0.019 |
| MKM | 0.5281 | 0.5132 | −0.015 |
| WOE | 0.5096 | 0.5041 | −0.006 |
| **Overall** | **0.5405 ± 0.0229** | **0.5264 ± 0.0201** | **−0.014** |

WR-agreement essentially unchanged (0.2529 → 0.2542).

**Conclusion: standardization consistently *hurts* (~−1.4 pt, every holdout down). Do NOT adopt it.**

**Why (likely):** the blunt fix z-scored *all* columns, including the 384-dim MiniLM text block
which is already L2-normalized per row. Per-column z-scoring distorts that geometry and **inflates
low-variance (less-informative) dimensions to unit variance, amplifying noise.** Meanwhile the card
encoder's `LayerNorm` + learned first layer already handle the raw structured-feature scales, so
there was nothing to fix. The Phase-1 "unstandardized features" caveat turns out to be a non-issue.

**Untested variant:** standardizing *only* the 73 structured-feature columns (leaving the text block
alone) might behave differently — but given the encoder's LayerNorm already normalizes internally,
the expected upside is small. Not pursuing it unless a reason appears.

## Decision (locks the recipe for the GPU run)

- **Recipe:** content encoder → Set Transformer pool encoder → pointer head → in-pack CE +
  aux-WR head (λ≈1.0), **raw (un-standardized) features**, optional pick-time quality blend for the
  "good, not just human" dial.
- **Expected generalization:** ~**0.54 ± 0.02** rotated (vs 0.233 floor, ~0.55 published bar);
  stable across seeds.
- The standardization knob stays in the code (`--standardize`) but **off by default**.
