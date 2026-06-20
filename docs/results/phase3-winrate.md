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

## Recommendation & next steps

Adopt **gentle win-weighting (`exp β≈0.4`)** as a free improvement. For a real "good, not just
human" gain (DD-004), go beyond deck-level wins:
- **Auxiliary adjusted-WR head** — regress a confounder-adjusted card win rate as a multi-task
  target, injecting card-quality signal directly into the representation.
- **Soft-label distillation** from a win-rate-aware / ensemble teacher (DD-004).
- Tighten with multiple seeds; consider per-game outcome signal from 17lands `game_data`.
