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

**Data-scaling curve (holdout DSK): saturates at ~3–4 sets *on top-1*.** 1→2 sets +9.6 pt, but 4→7
sets (nearly 2× data) only +0.15 pt — within noise. The small model is **data-saturated** *for top-1*,
so the full-corpus run is NOT justified *for accuracy*. **⚠ But this is top-1-only:** on
**WR-agreement** the corpus is *not* saturated — extending to 15 sets lifts WR-agree:GIH +0.018 (3.5σ)
and breaks the 0.29–0.31 ceiling ([wr-scaling.md](wr-scaling.md)). For the win-rate objective, train on
**as many diverse sets as available**, not 4–7.

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
  ~free top-1; pushes to 0.40 at α=8.
- **IWD advantage-weighted objective** (`scripts/pod_adv_objective.py`) — weight imitation by how good
  the human's pick was (IWD vs pack). Flips the model from *lagging* humans (0.257 < human 0.278) to
  **beating** them (τ=0.03 → 0.295 > 0.278) at ~2pt top-1 cost (0.571→0.548), baked into training (no
  runtime dial). **Recommended "good, not just human" recipe: IWD advantage-weighting (τ≈0.03),
  optionally + the IWD blend as a runtime aggressiveness dial.**

## Distillation (DD-004)

Soft-label KD arms (LOSO BLB+OTJ+WOE+MKM → DSK, GPU, 8 epochs). Full tables:
[distillation.md](distillation.md). Read on **WR-agreement**, not top-1 (all arms sit at the noise
ceiling). Human reference 0.2981 matches Phase 3 (eval consistency anchor).

| arm | WR-agree (baseline → method) | vs | verdict |
|---|---|---|---|
| ensemble-of-seeds (denoise) | 0.2597 → 0.2538 | — | **no help** (control: same labels, no new info) |
| **WR-softmax (dense WR target)** | 0.2597 → **0.2885** | scalar advantage 0.2725 | **win: +0.016 over the scalar method**, ~75% of the human gap |
| leaky → release-day | 0.2597 → 0.2590 | teacher **0.3303** (>human) | strong teacher, **transfer failed** (λ=0.5/temp2) |

**Takeaway:** the dense WR-softmax *target* is the keeper; denoising adds nothing; privileged-feature
teachers don't distill as-is. The lever with headroom is a better win-rate *signal*, not more
imitation data.

## "More data" axes + good players (DD-004 follow-ups)

Full detail: [good-players.md](good-players.md). Holdout DSK, composite-WR target, single seed.

| lever | result | verdict |
|---|---|---|
| more **sets** (4→7 diversity) | baseline top1 0.5710 → **0.5778** | helps (free) |
| richer **fields** (GIH+IWD+ALSA, conf-shrunk) | WR-agree **0.2902** at 4 sets (+0.0053 vs single) | helps, washes out at 7 sets |
| more **picks/set** (depth 60k→240k) | flat-to-down everywhere | **no — saturates** (confirmed under the new objective) |
| **good players** (multi-seed; vs volume-matched random) | good−rand WR **+0.0018→+0.0049→+0.0158** (4-set → 7-set → big net); top1-on-good **−0.0076→+0.0091** | **real, and grows with scale**; small-seed runs misled |
| **capacity** (big vs small net, 7 sets, multi-seed) | good/comp WR 0.2955 → 0.2988 (within noise) | **doesn't raise the ceiling** (signal-bound); but *amplifies* the good-player lever |

Best confirmed "good, not just human" config: **big net + good players + composite-WR + 7 sets →
WR-agreement 0.299 ± 0.005 ≈ average human (0.298)**. All WR-agreement results plateau at 0.29–0.31,
capped by the confounded GIH-WR proxy — the remaining lever with headroom is a less-confounded value
signal (game_data). (Note: single-seed runs over-stated the good-player win and "refuted" the
cleaner-label effect; multi-seed at scale reversed both — see [good-players.md](good-players.md).)

## Detailed docs

- [play-prob.md](play-prob.md) — **P(played | pool)**: the pool-dependent dynamic (color
  commitment / curve / castability) that `won` couldn't teach **IS strongly learnable from the build
  decision** (deck vs sideboard): pool context lifts AUC 0.81→0.93 and recovers castability (a white
  2-drop: 0.17 play-prob in a white-splash deck → 0.90 when committed). The signal lives in the *build*.
- [outcome-eval.md](outcome-eval.md) — **outcome eval** (the decisive, non-circular test): replay
  held-out drafts, score each policy's drafted pool by the game_data deck-value model. **The model's
  decks beat the humans' by +0.063 est. deck-WR in 95% of drafts** (GIH-trained, deck_value-scored).
  Caveat: the metric is a card-power sum (greedy rating policies beat it), not a curve/mana simulator.
- [play-prob-pick-time.md](play-prob-pick-time.md) — **buildability at PICK time** (resolved: flat).
  Training `P(played | pool)` on *partial* pools fixes the OOD problem — at λ=0.5 drafted pools get more
  color-coherent (75%→78%, matching humans), unlike the full-pool run where it fell — but **deck-WR stays
  flat** (+0.0027, within noise). The drafter is already coherent enough, and `gih_greedy` wins as much
  with only 58% coherence, so color-coherence isn't the WR bottleneck. **Buildability is a build-time
  lever, not a pick-time one**; the faithful pick-sequence version is shelved (gate not met).
- [wr-scaling.md](wr-scaling.md) — **WR-agreement data-scaling curve** (corpus 8→23 sets): on the
  *win-rate* axis the corpus is **NOT saturated** (Phase-4 saturation was top-1-specific). 7→15 sets
  lifts WR-agree:GIH **+0.014–0.025 across 4/4 rotated holdouts** (DSK/OTJ/MOM/FDN); it then **peaks
  ≈0.326 at ~19 relevant sets and DECLINES at 22** when padded with remaster/Masters sets — corpus
  *relevance* matters, not just count. Cleanest break of the 0.29–0.31 ceiling found, from data alone.
  **Curation pass (explicit corpora, 3 seeds) confirms relevance, not count:** a 19-set corpus that
  *keeps* the reprint sets SIR/PIO scores 0.315 vs 0.325 for the one that drops them (equal count); the
  culprits are SIR (Innistrad remaster) + PIO (Pioneer Masters), STX is neutral. Curation recovers but
  does not exceed the ~0.326 peak → that's the corpus-breadth ceiling. Recipe: exclude remaster/Masters.
- [game-data-value-model.md](game-data-value-model.md) — **game_data value model (Steps 0–2)**:
  per-game outcome regression β_c → `deck_value`. Step 0 = **PROCEED**; Step 1 = built for all 8 sets
  (face-plausible, split-half ρ 0.64, on HF); **Step 2 (4-seed)** = `deck_value` is a **better target**
  (top-1 +0.009, value-agree +0.023 4/4) but the single-seed **GIH-ceiling-break did NOT hold up**
  (+0.007, 3/4, within noise). Decisive test still owed: outcome eval (estimated deck-WR).
- [distillation.md](distillation.md) — soft-label KD: ensemble / WR-softmax / leaky-feature (DD-004).
- [good-players.md](good-players.md) — skill-filtered training + the volume-matched control.
- [phase1-generalization.md](phase1-generalization.md) — single-set vs multi-set LOSO; the
  encoder ablation (features / hashing / MiniLM) and why text needs set diversity.
- [phase2-set-transformer-infonce.md](phase2-set-transformer-infonce.md) — full Phase 2 analysis.
- [phase3-winrate.md](phase3-winrate.md) — win-rate weighting + aux-WR head + pick-time blend.
- [phase4-robustness.md](phase4-robustness.md) — seeds × rotating holdout; standardization A/B.

## Shared caveats

Single seed; sampled data (60–80k picks/set); structured features unstandardized; text dims
uncontrolled across encoders; one holdout set. Results are directional. Tightening (multiple seeds,
rotating holdout, feature standardization) is the immediate next step before Phase 3.
