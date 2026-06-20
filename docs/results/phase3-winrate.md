# Phase 3 — Win-rate weighting + WR-agreement ("good, not just human")

**Question:** can we nudge picks toward *winning* cards, not just *human-like* ones? We weight each
training example by its draft's `event_match_wins` (advantage-weighted BC) and measure, on the
held-out set, **WR-agreement**: how often the model takes the highest-GIH-WR card in the pack
(17lands `ever_drawn_win_rate`), plus the average GIH-WR of the chosen card. A human reference comes
from the same held-out picks.

**Setup:** best Phase-2 model (`set_transformer` + `ce`, MiniLM), LOSO BLB+OTJ+WOE+MKM → DSK,
WR metrics over 22,371 DSK packs with ≥2 rated cards. Single seed.

Reproduce: add `--win-weight {linear,exp} --win-beta B --holdout-ratings <17lands.json>`.

## Results

| win-weight | in-set top-1 | held-out top-1 | WR-agreement (model) | avg pick GIH-WR (model) |
|---|---|---|---|---|
| none *(baseline)* | 0.6134 | **0.5626** | 0.2553 | 0.5465 |
| **exp β=0.4** | 0.6087 | 0.5623 | **0.2632** | **0.5471** |
| linear (wins+1) | 0.6119 | 0.5597 | 0.2607 | 0.5469 |
| exp β=0.8 (aggressive) | 0.5770 | 0.5393 | 0.2520 | 0.5461 |
| *human reference* | — | — | 0.2990 | 0.5509 |

## Findings

1. **Gentle win-weighting is a free, small win.** `exp β=0.4` raises WR-agreement 0.2553 → 0.2632
   (+0.8 pt) and avg pick-WR slightly, at **no cost** to held-out human top-1 (0.5626 → 0.5623).
   This is the recommended setting.
2. **Aggressive weighting backfires.** `exp β=0.8` concentrates training on a few high-win drafts,
   losing data diversity — held-out top-1 drops to 0.539 **and** WR-agreement drops *below* baseline
   (0.252). More signal is not better; there's a clear sweet spot. `linear` sits in between.
3. **The imitation model trails humans on WR-agreement** (0.26 vs 0.30) and picks slightly
   lower-WR cards on average (0.5465 vs 0.5509). It mimics the *average* drafter, and even average
   humans approximate the GIH-WR signal a bit better. Win-weighting closes part of the gap but does
   not surpass humans.
4. **`event_match_wins` is a weak lever.** It's a deck-level, confounded, noisy signal (one value
   per 42-pick draft), so it can only nudge picks. Surpassing humans on pick quality needs a
   stronger, card-level signal.

## Interpretation / caveats

- GIH WR is itself a **confounded proxy** (favors controlling decks; entangles deck/archetype/player
  skill). That humans hit only ~30% "best-GIH-WR" confirms the highest-GIH-WR card is often *not*
  the right pick (synergy, curve, signal). So WR-agreement is **directional**, not ground truth.
- Single seed, sampled data, one holdout — treat deltas of <1 pt as suggestive.

## Adjusted-WR auxiliary head (multi-task card-quality signal)

A second head predicts each card's (per-set-standardized) GIH WR from its embedding; the loss is
`pick_loss + λ·MSE(predicted_WR, target_WR)`. The aux head shares the card encoder, so it pushes the
representation to encode *card quality*, not just pick-imitation. Targets built from 17lands ratings
for the **training** sets; held-out unchanged.

| config (set_transformer + ce) | in-set | held-out top-1 | novel-only | MTPD | WR-agreement | avg pick WR |
|---|---|---|---|---|---|---|
| baseline (no aux) | 0.6134 | 0.5626 | 0.5461 | 0.927 | 0.2553 | 0.5465 |
| + aux-WR λ=0.5 | 0.6189 | 0.5737 | 0.5571 | 0.879 | 0.2565 | 0.5467 |
| + aux-WR λ=1.0 | 0.6214 | **0.5744** | **0.5578** | 0.892 | 0.2547 | 0.5467 |
| *human reference* | — | — | — | — | 0.2990 | 0.5509 |

**Findings (a useful surprise):**
1. **The aux head is the biggest single generalization gain since the Set Transformer** — held-out
   0.5626 → **0.5744** (+1.2 pt), novel-only 0.546 → 0.558, MTPD 0.927 → 0.879. New best overall.
   Robust to λ (0.5 ≈ 1.0).
2. **But WR-agreement barely moved** (~0.255). So the aux head's value is *not* the targeted
   "good-not-just-human" steering — it's that forcing the encoder to **predict card quality from
   content** is a strong, transferable auxiliary task, yielding richer card embeddings that judge
   *unseen* cards better. The benefit is generalization, not picking the single top-WR card more.
3. Why WR-agreement stays flat: the pick is still driven by the imitation pointer head; the aux
   signal improves the representation diffusely rather than re-pointing the policy at the top-WR
   card — and GIH-WR-best is a noisy, confounded target anyway.

## Do the two levers stack? (no)

| config (set_transformer + ce) | held-out top-1 | WR-agreement | avg pick WR |
|---|---|---|---|
| baseline | 0.5626 | 0.2553 | 0.5465 |
| win-weight only (exp β=0.4) | 0.5623 | **0.2632** | 0.5471 |
| aux-WR only (λ=1.0) | **0.5744** | 0.2547 | 0.5467 |
| aux-WR + win-weight | 0.5677 | 0.2593 | 0.5472 |

Combining is **Pareto-interior**: it beats baseline on both axes but is *worse than each lever on
that lever's own strength* (top-1 0.5677 < aux's 0.5744; WR-agreement 0.2593 < win-weight's 0.2632).
They partly work against each other — win-weighting concentrates on winning drafts, shrinking the
data diversity that drives aux's generalization; aux broadens the representation, diluting
win-weighting's targeted WR push. **Don't combine; pick the lever that matches the goal.**

## Pick-time quality blend (the lever that finally moves WR-agreement)

At draft time, blend the aux head's **predicted** quality into the pick:
`logit = pointer_logit + α · predicted_quality(card)`. α is a post-training knob (no retraining),
and it uses the *prediction*, so it works on unseen cards (no ratings needed at draft time).
Sweep on held-out DSK (aux-WR model, λ=1.0):

| α | held-out top-1 | novel-only | WR-agreement | avg pick WR |
|---|---|---|---|---|
| 0.0 | **0.5740** | 0.5576 | 0.2552 | 0.5466 |
| 0.5 | 0.5716 | 0.5544 | 0.2636 | 0.5476 |
| 1.0 | 0.5635 | 0.5457 | 0.2704 | 0.5482 |
| 2.0 | 0.5421 | 0.5232 | 0.2814 | 0.5493 |
| 4.0 | 0.4987 | 0.4777 | 0.2961 | 0.5506 |
| 8.0 | 0.4426 | 0.4190 | **0.3117** | 0.5516 |
| *human reference* | — | — | 0.2990 | 0.5509 |

**Findings:**
1. **This is the first lever that substantially and monotonically raises WR-agreement** (0.255 →
   0.312) — and it can **match humans at α≈4** (0.296 vs 0.299) and **exceed them at α≈8** (0.312 >
   0.299; avg pick WR 0.5516 > human 0.5509). The aux head and win-weighting alone never moved it.
2. **No free lunch — it's a frontier, not a strict win.** Every step toward winning cards costs
   human top-1 (0.574 → 0.443), because "highest-GIH-WR card" and "human pick" are *different
   objectives* (humans themselves only agree ~30%). The blend lets you *choose where on that
   frontier to sit*.
3. **The blend dominates win-weighting.** At α=0.5 it gives WR-agreement 0.2636 at top-1 0.5716 —
   better on **both** axes than win-weighting (0.2632 at 0.5623), because it doesn't shrink data
   diversity. Prefer the blend over win-weighting for the "good, not just human" axis.
4. **α is a deployable dial** ("draft aggressiveness"): low α = human-like; high α = win-rate-greedy.

## Recommendation & next steps

- **Adopt the aux-WR head (λ≈0.5–1.0)** — it's the current best model (held-out 0.574) and stacks
  with the Set Transformer. Optionally combine with gentle win-weighting (`exp β≈0.4`).
- For the "good, not just human" axis, use the **pick-time quality blend** (it dominates
  win-weighting and is a tunable dial that can match/exceed humans). Gentle α≈0.5 is a near-free
  WR boost; raise α to trade human-likeness for win-rate-seeking.
- Use a less-confounded target (IWD `drawn_improvement_win_rate`, or a deck-adjusted WR) — should
  make the blended picks "better" in a less control-biased sense.
- Tighten with multiple seeds + rotating holdout; **soft-label distillation** from a win-rate-aware
  teacher (DD-004) remains the heavier alternative.
