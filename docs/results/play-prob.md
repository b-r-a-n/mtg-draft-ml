# P(played | pool) — the pool-dependent dynamic IS learnable, from the build decision

**Question.** The outcome study ([outcome-eval.md](outcome-eval.md)) found that curve / castability /
synergy are **not** recoverable from `won`-over-deck-composition (noise + range restriction). But those
effects are *build-time* decisions — so the right signal should be the **build itself**, not the game
outcome. 17lands `game_data` carries `deck_<card>` (played) **and** `sideboard_<card>` (in the pool,
cut), so for every card in a drafter's pool we know whether they **played or cut** it. That cut is the
human's build judgment, and it's *context-dependent* (a white card gets cut from a green deck), not
range-restricted, not single-game-noisy. Can a model learn it?

**Method** (`scripts/prototype_play_prob.py`, DSK, 12k built decks, 434k (deck, card-in-pool) rows,
played rate 0.64; split by draft). Predict `played` two ways, held-out:
- **baseline** — card features only (CMC, β, colors, type): the card's *intrinsic* playability.
- **+context** — card features **+ pool composition** (per-color counts, CMC histogram, size).

## Result — pool context is the dominant signal (opposite of `won`)

| P(played) model | log-loss | AUC |
|---|---|---|
| baseline (card only) | 0.5029 | 0.8066 |
| **+context (card + pool)** | **0.3277** | **0.9295** |

Adding the pool **halves the log-loss and lifts AUC 0.81 → 0.93** — whereas the same kind of
interaction added *nothing* to predicting `won`. The build decision is **strongly pool-dependent and
strongly learnable.** And with **no feature engineering** (raw pool color counts in; the model learns
the card-color × pool-color interaction), it recovers exactly the castability dynamic:

```
P(play a median white 2-drop)  vs  the pool's WHITE commitment (deck ≈ 23 spells):
   2 white cards (splash) -> 0.165     # cut: unreliable to cast a white card in a green/2-white deck
   5 white                -> 0.379
   9 white                -> 0.727
  13 white                -> 0.871
  17 white (committed)    -> 0.899     # played
```

That's the "two W two-drops in a GW-splash-W deck are bad" intuition, learned end-to-end from the build
data — not from `won` (where it's censored), not hand-coded.

## Why this matters / how it could feed drafting

The whole pool-dependent dynamic (color commitment, curve fit, buildability) **does** exist and **is**
learnable — it just lives in the **build**, and the supervised signal is "what good players play vs
cut," not game outcomes. A natural use at *pick* time: discount a card's pick value by
**P(played | my current pool)** — a learned, contextual *buildability* signal that down-weights cards
unlikely to make the deck (off-color, redundant high-drops), without a simulator or hand-engineered
curve/color features. This is the concrete realization of the roadmap's "deckbuild signal" unlock.

## Folding it into the pick policy — doesn't help (it belongs at build time)

Tested the obvious use (`scripts/test_play_policy.py`): at each pick, reweight the model's logit by the
buildability signal — `effective(c) = model_logit(c) + λ·logit P(played | c, pool)` — and score the
drafted pool by the outcome eval's constrained 2-color+curve deck-WR (600 DSK drafts).

| policy | deck-WR | Δ vs model | top-2-color % |
|---|---|---|---|
| human | 0.568 | −0.030 | 78% |
| **model** (λ=0) | 0.598 | — | 75% |
| model+build@0.5 | 0.599 | +0.0015 | 75% |
| model+build@1.0 | 0.581 | −0.016 | 66% |
| model+build@2.0 | 0.555 | −0.042 | 62% |

**It doesn't improve drafting** — flat at λ=0.5 (+0.0015, noise) and *worse* at higher λ. Two reasons,
both informative: (1) **OOD** — `P(played|pool)` was trained on *full* ~40-card pools but applied to
*partial* mid-draft pools, so the signal is noisy exactly where it's used (color concentration goes
*down* with λ, not up — it's adding noise, not coherence); (2) the plain model already drafts coherently
(75% on-color ≈ humans' 78%) and the eval's deck-*builder* already handles construction, so there's
little pick-time headroom. **Conclusion: the buildability signal is real and learnable, but belongs at
the deck-*build* step (where it's strong and in-distribution), not folded into picks** — which is
exactly the "pick for value, build for buildability" division the rest of the analysis converged on.
(A partial-pool-trained `P(played)`, or using it only late in the draft, might fare better — untested.)

## Honest caveats

- **This imitates human build judgment, not winning.** It answers "will good players play this card from
  this pool," not "does playing it win more." It's the right *buildability* signal, but whether folding
  it into the pick policy improves *outcomes* is the next experiment (and `won` is a noisy judge of it).
- **AUC 0.93 is partly easy cases** (a clearly off-color card is always cut) — but the inspection shows
  real *gradation*, not just a hard color gate, which is the contextual part that matters.
- Single set / single build per draft (dedup by `draft_id`); sideboarding variants (`build_index`)
  collapsed. Directional prototype, not a tuned model.
