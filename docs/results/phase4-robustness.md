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

## Data-scaling curve — does more data help? (No, at this model size)

Fix holdout = DSK and the best recipe; train on the first N of [BLB, OTJ, WOE, MKM, LCI, MOM, MH3].
Single seed (`scripts/scaling_curve.py`).

| n_sets | train picks | train cards | held-out top-1 |
|---|---|---|---|
| 1 | 60k | 276 | 0.4668 |
| 2 | 120k | 652 | 0.5628 |
| 3 | 180k | 975 | 0.5709 |
| 4 | 240k | 1294 | 0.5744 |
| 5 | 300k | 1580 | 0.5705 |
| 6 | 360k | 1920 | 0.5743 |
| 7 | 420k | 2240 | 0.5759 |

**Saturates at ~3–4 sets.** The only large gains are 1→2 (+9.6 pt) and 2→3 (+0.8); from 4→7 sets
(nearly 2× the data, +950 cards) accuracy moves **+0.15 pt — within noise.** The small model is
**data-saturated, not data-limited** — it lacks the capacity to exploit more data.

## Depth-scaling test (the untested axis): 8× more picks/set also does nothing

The scaling curve above varied SET COUNT at a fixed 60k picks/set. The other axis — depth per set —
was untested (we always sampled to 60k; full sets are millions). Tested the landed recipe
(set_transformer + IWD advantage-weighting) at 60k vs 500k picks/set, 4 sets → DSK:

| depth (total picks) | top-1 | top-3 | top-5 | WR-agree (IWD) |
|---|---|---|---|---|
| 60k/set (250k) | 0.560 | 0.875 | 0.963 | 0.288 |
| 500k/set (2.0M) | 0.553 | 0.873 | 0.961 | 0.298 |

**Flat on every metric** (top-1/3/5 and win-rate), despite 8× the data. The model is data-saturated
on BOTH axes (set count and depth), on BOTH human-imitation and fair top-k/win-rate metrics.
Five independent levers now exhausted: more sets, more depth, more capacity, sequence inputs,
pool-conditioning. A laptop-trained model on sampled sets is at the frontier.

## Decision — the GPU full-corpus run is NOT justified (for accuracy)

- **More data won't help at this (deployable, ~10M-param) model size.** The full-corpus GPU run
  would cost time/money for ~0 accuracy gain. **Train the deployable model on the laptop** on a
  handful of diverse recent sets (≈4–7) with the locked recipe.
- **Locked recipe:** content encoder → Set Transformer → pointer head → in-pack CE + aux-WR head
  (λ≈1.0), **raw features**, optional pick-time quality blend. Generalization ~**0.54–0.57** to an
  unseen set (vs 0.233 floor, ~0.55 published bar); stable across seeds.
- Standardization knob stays in code (`--standardize`) but off by default.

## Capacity sweep — does a BIGGER model help? (No — and now properly tested)

Two attempts (both on rented RunPod GPUs, 7 train sets → holdout DSK):

**Attempt 1 (naive, fixed LR=1e-3, 12 epochs):** the M model *collapsed* — in-set 0.62 → 0.41.
That's an **optimization failure, not a capacity verdict** (a bigger model that can't even fit train
was mis-trained), so it didn't answer the question.

**Attempt 2 (tuned: per-size LR + 10% LR warmup + grad-clip=1.0, 20 epochs):** bigger models now
train correctly (no collapse). Definitive result:

| config | params | lr | in-set | held-out | novel |
|---|---|---|---|---|---|
| S | ~2M | 1e-3 | 0.6241 | **0.5793** | 0.5636 |
| M | ~8M | 5e-4 | 0.6263 | 0.5772 | 0.5612 |
| L | ~15M | 3e-4 | 0.6263 | 0.5772 | 0.5612 |

**Conclusion: capacity is NOT the lever.** With proper optimization, bigger models fit train
*slightly* better (in-set 0.624 → 0.626) but **held-out generalization is flat** (0.579 / 0.577 /
0.577 — S is marginally best, all within noise). We are at a **task/data ceiling (~0.58) for this
problem framing**, not capacity-limited. More parameters won't break it.

**Routing decision:** since neither more data (scaling curve) nor more capacity (this sweep) breaks
~0.58, the remaining lever is **inputs** — the Phase-5 sequence model that adds the signal-reading
information channel the memoryless model can't see ([../phase5-sequence-modeling.md](../phase5-sequence-modeling.md)).
That, not a bigger model, is the next thing to try if we want past ~0.58.

### Infra footnote (the debugging that made this run possible)
The capacity sweep crashed silently several times before this clean run. Root cause was **not**
flakiness: `uv run` auto-syncs the venv and re-resolved `torch>=2.2` to the latest PyPI wheel
(2.12.1+**cu130**, CUDA 13), too new for RunPod host drivers (e.g. 575 = CUDA 12.9) → GPU invisible →
silent crash on first CUDA use. Hosts with newer drivers (580) happened to work, masking it as
"flakiness." Fixed by pinning torch to the **cu124 index for Linux** in `pyproject.toml`
`[tool.uv.sources]` (macOS keeps the default MPS wheel) + a **fatal** GPU check in the bootstrap.
