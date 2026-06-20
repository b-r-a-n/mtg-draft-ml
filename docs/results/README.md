# Results

Index of experimental results. **Headline metric:** zero-shot top-1 pick accuracy on a *held-out
set* (cards the model never trained on), vs the **0.233 random floor** and the **~0.55 published
bar** (Bertram et al. 2024). All runs: MPS, sampled 17lands data, single seed unless noted.

## Cross-phase progression (held-out / unseen-set top-1)

| Phase | Config | Train sets | Held-out | top-1 | Notes |
|---|---|---|---|---|---|
| 0 | one-hot MLP baseline | single | — | n/a | fixed-vocabulary — **cannot** run cross-set |
| 1 | content encoder, MiniLM | BLB | DSK | 0.437 | single-set; text *underperforms* features here |
| 1 | content encoder, MiniLM, **LOSO** | BLB+OTJ+WOE+MKM | DSK | 0.552 | multi-set — matches the ~0.55 bar |
| 2 | + **Set Transformer** (in-pack CE) | BLB+OTJ+WOE+MKM | DSK | **0.563** | **current best** |
| 3 | + win-weighting (exp β=0.4) | BLB+OTJ+WOE+MKM | DSK | 0.562 | top-1 flat; WR-agreement ↑ |

Each step is a real, measured improvement on a ~98%-novel held-out set. The Phase-0 baseline is
omitted from the cross-set column because a fixed-vocabulary model has no parameters for unseen cards.
Phase 3 optimizes *pick quality* (WR-agreement), not human top-1 — see below.

## Phase 2 ablation (Set Transformer × loss) — held-out DSK

| pool | loss | in-set top-1 | **held-out top-1** | novel-only | MTPD |
|---|---|---|---|---|---|
| mean | ce *(Phase 1 baseline)* | 0.588 | 0.552 | 0.534 | 1.005 |
| **set_transformer** | **ce** | 0.614 | **0.563** | 0.546 | **0.919** |
| mean | infonce (512 global neg) | 0.436 | 0.349 | 0.324 | 2.341 |
| set_transformer | infonce (512 global neg) | 0.441 | 0.350 | 0.324 | 2.177 |

- **Set Transformer wins** (synergy modeling): +0.011 held-out, +0.026 in-set, better MTPD.
- **Global-negative InfoNCE hurts** (−0.20): the *pack* is the correct negative set, so `ce`
  (≡ in-pack InfoNCE) is the right objective; diluting with random negatives wrecks within-pack
  ranking. Recommended config: **`set_transformer` + `ce`**.

## Phase 3 — win-rate weighting (WR-agreement, held-out DSK)

| win-weight | held-out top-1 | WR-agreement (model) | avg pick GIH-WR |
|---|---|---|---|
| none | 0.5626 | 0.2553 | 0.5465 |
| **exp β=0.4** | 0.5623 | **0.2632** | 0.5471 |
| exp β=0.8 (too strong) | 0.5393 | 0.2520 | 0.5461 |
| *human reference* | — | 0.2990 | 0.5509 |

Gentle win-weighting nudges picks toward higher-WR cards at no top-1 cost; aggressive weighting
backfires (data concentration). The model still trails humans on WR-agreement — deck-level
`event_match_wins` is a weak lever; a card-level adjusted-WR head is the next step.

## Detailed docs

- [phase1-generalization.md](phase1-generalization.md) — single-set vs multi-set LOSO; the
  encoder ablation (features / hashing / MiniLM) and why text needs set diversity.
- [phase2-set-transformer-infonce.md](phase2-set-transformer-infonce.md) — full Phase 2 analysis.
- [phase3-winrate.md](phase3-winrate.md) — win-rate weighting sweep + WR-agreement metric.

## Shared caveats

Single seed; sampled data (60–80k picks/set); structured features unstandardized; text dims
uncontrolled across encoders; one holdout set. Results are directional. Tightening (multiple seeds,
rotating holdout, feature standardization) is the immediate next step before Phase 3.
