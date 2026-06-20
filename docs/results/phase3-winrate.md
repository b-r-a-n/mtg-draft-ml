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

## Recommendation & next steps

- **Adopt the aux-WR head (λ≈0.5–1.0)** — it's the current best model (held-out 0.574) and stacks
  with the Set Transformer. Optionally combine with gentle win-weighting (`exp β≈0.4`).
- To actually move *WR-agreement* (steer the policy, not just the representation): try a
  pick-time blend of the aux quality score with the pointer logits, or **soft-label distillation**
  from a win-rate-aware teacher (DD-004).
- Use a less-confounded target (IWD `drawn_improvement_win_rate`, or a deck-adjusted WR).
- Tighten with multiple seeds + rotating holdout.
