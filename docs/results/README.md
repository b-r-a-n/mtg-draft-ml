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
| 3 | + **adjusted-WR aux head** (λ=1.0) | BLB+OTJ+WOE+MKM | DSK | **0.574** | best set-model; generalization ↑ |
| 5 | sequence model (signal-reading) | BLB+OTJ+WOE+MKM | DSK | 0.574 | ties — does NOT break the ~0.58 ceiling |

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
`event_match_wins` is a weak lever.

The **adjusted-WR auxiliary head** (predict each card's WR from its embedding, multi-task) is the
bigger win: held-out 0.563 → **0.574** (best), novel-only 0.546 → 0.558 — but it boosts
*generalization*, not WR-agreement (the shared encoder gets better at judging unseen cards, rather
than re-steering the pick policy toward the top-WR card).

The **pick-time quality blend** (`logit += α·predicted_quality`) is the lever that *does* move
WR-agreement — a tunable dial that matches humans at α≈4 and exceeds them at α≈8 (0.312 > 0.299),
trading human top-1 for win-rate-seeking. It dominates win-weighting and works on unseen cards
(uses the head's prediction). Use α≈0 for best generalization, raise α to be "good, not just human".

## Phase 4 — robustness (seeds × rotating holdout)

Best config across **3 seeds × 5 rotating holdouts** (15 runs): **held-out top-1 = 0.5405 ± 0.0229.**
Seeds are stable (±0.001–0.003); the spread is *which set* is held out (range 0.510–0.575). The
flagship 0.574 was DSK-specific (an easy holdout) — the honest rotated number is **~0.54**.

**Feature standardization A/B: it *hurts*** (0.5405 → 0.5264, every holdout down ~1–2 pt) — z-scoring
the already-unit-norm text block amplifies noise dims, and the encoder's LayerNorm already handles
raw feature scale. **Not adopted** (kept as an off-by-default `--standardize` flag).

**Data-scaling curve (holdout DSK): saturates at ~3–4 sets.** 1→2 sets +9.6 pt, but 4→7 sets
(nearly 2× data) only +0.15 pt — within noise. The small model is **data-saturated, not
data-limited**, so the **full-corpus GPU run is NOT justified for accuracy** — train the deployable
model on the laptop on ~4–7 diverse sets.

**Capacity sweep (tuned, on GPU): bigger models do NOT help either.** With per-size LR + warmup +
grad-clip (fixing an earlier fixed-LR collapse), S/M/L (2M/8M/15M) give held-out 0.579/0.577/0.577 —
flat. **Neither more data nor more capacity breaks ~0.58** → we're at a task/data ceiling for this
framing. The remaining lever is **inputs** (Phase-5 sequence model / signal-reading), not parameters.
Recipe locked: content encoder → Set Transformer → in-pack CE + aux-WR (raw features).

## Probes

- **Human-disagreement probe** ([human-disagreement-probe.md](human-disagreement-probe.md)) —
  tests whether ~0.58 is irreducible human noise. Pairwise human agreement collapses from 0.85
  (early) to 0.68 (mid-pack, 31% near-coinflip); the model's errors concentrate in exactly that
  mid-pack region; pure popularity scores only 0.41 (so the model is genuinely contextual).
  **Conclusion: ~0.58 is largely a human-noise ceiling; the lever with headroom is the win-rate
  objective, not human-pick accuracy.**
- **Objective experiment** (`scripts/pod_objective_sweep.py`) — win-rate blend × IWD-vs-GIH target.
  IWD (less-confounded) gives a much steeper win-rate frontier: at α=1, WR-agreement 0.25→0.29 for
  ~free top-1; pushes to 0.40 at α=8. **Recommended for "good, not just human": IWD aux target +
  modest blend (α≈1–2).**

## Detailed docs

- [phase1-generalization.md](phase1-generalization.md) — single-set vs multi-set LOSO; the
  encoder ablation (features / hashing / MiniLM) and why text needs set diversity.
- [phase2-set-transformer-infonce.md](phase2-set-transformer-infonce.md) — full Phase 2 analysis.
- [phase3-winrate.md](phase3-winrate.md) — win-rate weighting + aux-WR head + pick-time blend.
- [phase4-robustness.md](phase4-robustness.md) — seeds × rotating holdout; standardization A/B.

## Shared caveats

Single seed; sampled data (60–80k picks/set); structured features unstandardized; text dims
uncontrolled across encoders; one holdout set. Results are directional. Tightening (multiple seeds,
rotating holdout, feature standardization) is the immediate next step before Phase 3.
