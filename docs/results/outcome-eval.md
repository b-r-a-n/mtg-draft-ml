# Outcome eval — do the model's picks build decks that actually win?

**Question.** Every other metric is *agreement with a card rating* (WR-agreement vs GIH or deck_value).
The decisive, long-deferred test (docs/game-data-plan.md "estimated deck-WR"): replay held-out drafts,
have each policy draft a seat, and score the resulting **pool** by the **game_data deck-value model** —
σ(Σ of the best-23 spell β) ≈ the deck's estimated win probability. Then compare the model's decks to
the **humans who actually drafted those seats**.

**Why it's non-circular.** The deployed model was trained on the **GIH-composite** target, *not* on
`deck_value`. Scoring its decks by the `deck_value` outcome model therefore tests whether
imitation-style drafting yields *outcome-good* decks — it can't win by having been trained toward the
scorer. (`scripts/run_outcome_eval.py`, 800 held-out DSK drafts, the deployed 20-set webapp model.)

## Result — the model drafts outcome-better decks than humans

Each pool is scored by building its **best legal 2-color, curve-respecting deck** (`build_best_deck`:
best of the 10 color pairs, filling a CMC-bucket curve, using the `ci`/`cmc`/`t` fields) — so a 5-color
bomb pile collapses to one pair's depth and a pool with no early drops can't fill the low buckets. The
naive global top-23 (no color/curve) is shown for contrast.

| policy | **deck-WR (2-color+curve)** | Δ vs human | beats human | (naive top-23) |
|---|---|---|---|---|
| `deckvalue_greedy` (oracle: highest-β each pack) | 0.622 | +0.072 | 97% | 0.773 |
| `gih_greedy` (highest GIH-WR each pack) | 0.587 | +0.037 | 81% | 0.738 |
| **model** (deployed, GIH-trained) | **0.579** | **+0.029** | **74%** | 0.703 |
| *human* (recorded picks) | 0.550 | — | — | 0.639 |
| `random` | 0.479 | −0.071 | 8% | 0.531 |

**The model's decks beat the humans' by +0.029 estimated deck-WR, in 74% of drafts — non-circularly**
(GIH-trained, deck_value-scored). The edge is real but **modest**, and it **survives the realistic
metric**: under the 2-color+curve build, the color-undisciplined greedy policies lose most of their
apparent advantage — `gih_greedy`'s lead over the model collapses from +0.035 (naive) to **+0.009**,
because its rainbow bomb-pile can't form a legal deck. So forcing a playable deck punishes
"hoard the bombs" hardest, exactly as it should.

## Honest caveats (what this does and does NOT show)

- **It's a deck-strength estimate, not a game simulator.** The score is Σβ over a constrained legal
  deck — now color/curve-aware, but still no **synergy/sequencing/mana-base** modeling, and the per-card
  β is itself an observational outcome estimate. So this is "the model builds higher-β legal decks than
  humans," not a played-out win rate.
- **On-rails packs.** Policies draft the *recorded* pack sequence — their own picks don't change what
  wheels (a standard counterfactual approximation, not a full pod simulation).
- **In-distribution.** The deployed model trained on DSK (among 20 sets); this is a deck-quality
  comparison (model vs human on the same packs), not a generalization test.

## How much headroom is there beyond per-card value? (≈ none)

The pool-dependent considerations a "smarter" drafter would weigh — mana **curve**, contextual
**castability** (a WW card is worse in a W-light deck), **synergy** (two-card combos) — are all
*interactions*: a card's value depends on the rest of the deck. The `deck_value` model is **linear**
(`σ(Σ β_c·count_c)`), so it can't represent them. Do they actually predict **winning**? (`scripts/probe_deck_outcome.py`,
DSK game_data, held-out.)

| model | log-loss | AUC |
|---|---|---|
| linear (deck_value family) | **0.6682** | **0.6084** |
| gradient boosting (any card interaction) | 0.6709 | 0.6005 |
| linear + top-60 **synergy** interactions | 0.6687 (Δ +0.0006) | 0.6068 |

**No interaction is recoverable as outcome-predictive structure.** Gradient boosting — free to use any
curve×color/castability/synergy interaction — does *not* beat the linear per-card model; and a
*targeted* synergy test (rank card pairs by where the linear model under-predicts wins, add the top
60) doesn't improve held-out either, and its "top synergy pairs" are unrecognizable commons (train
noise that doesn't generalize, not real combos). **Deck win rate is ≈ linear in which cards are in the
deck.**

Why — three compounding reasons:
1. **Single-game outcomes are very noisy.** AUC ≈ 0.61 from the *whole deck* — opponent, draws,
   mulligans, and play skill dominate who wins a given game. Interaction effects are a second-order
   slice of an already-small signal, below the noise floor even at 150k games.
2. **Range restriction.** The game_data only contains decks humans actually *built* — castable and
   curve-sane, because they fixed it at deckbuild. The bad case (uncastable splash, no curve) is
   censored, so castability/curve can't show up. (Synergy isn't censored — and still shows nothing.)
3. **Limited synergy is mostly soft** (good cards in a color), already absorbed into per-card β; hard
   combos are rare and below the support threshold.

**Implication (the bound):** a pool-conditioned objective (marginal deck-WR, curve/synergy-aware) has
**essentially no extra outcome signal to learn from** beyond per-card value. This isn't "those effects
don't exist" — it's that they're enforced downstream, absorbed into averages, or below the
outcome-noise floor, so they're **not learnable from win/loss here.** Per-card-value drafting is close
to the achievable ceiling for the outcome we can measure. (Caveat: rare bomb-combos below the support
cut could be missed; this bounds what's *learnable from these outcomes*, not metaphysics.)

## Verdict

The WR-agreement gains **do** translate to outcome-better *decks* than humans — and the result holds up
under a realistic 2-color + curve deck-build (+0.029, 74% of seats; non-circular), which also strips the
greedy "bomb-hoard" policies of the edge the naive metric handed them. The remaining rungs are
**synergy/mana-base** modeling and a played-out win simulator, if this becomes load-bearing.

## Mechanistic probe — is the model itself curve-aware? (no — it's curve-blind)

Does the draft model do *contextual* curve-completion (up-weight cheap cards when its pool is
top-heavy)? A causal intervention on the deployed ONNX (`scripts/probe_curve.py`) tests it: hold a pack
fixed (a cheap + an expensive card of the same color, **β-matched** so value isn't the tiebreaker),
swap only the **pool** between top-heavy and low-curve (both drawn from **mid-β filler** so the pools
are power-matched and differ only in CMC), and measure the shift toward the cheap card.

| | confounded (top-heavy = bombs) | **β/power-matched** |
|---|---|---|
| shift toward cheap when pool top-heavy | −0.169 logit | **−0.024 logit** (−5σ, n=535) |
| % favoring cheap more when top-heavy | 15% | **39%** (50% = curve-blind) |

Controlling for power shrinks the effect ~7×: most of the apparent "anti-curve" behavior was just
"top-heavy pool = committed high-power deck." What remains is **near-zero, slightly negative** — i.e.
the model is **curve-blind**: changing the pool's curve barely moves its pick, and if anything nudges
*away* from the cheap card. Its pool-conditioning is **color / power / synergy, not mana curve.**

**And the human pick data is curve-blind too — which is the real explanation.** Same question on the
17lands picks (`probe_curve.py --human`): how often does a drafter take a card *cheaper* than the
highest-GIH card in the pack, split by whether their pool is top-heavy?

| | balanced pool | top-heavy pool | Δ |
|---|---|---|---|
| all players | 0.270 | 0.270 | +0.000 |
| **good players** | 0.266 | 0.270 | **+0.004** |

Even **good players don't pick curve-fixers contextually** — the cheaper-pick rate is flat in their
pool's curve. The reason is the **pick-vs-build split**: in Premier draft you draft a 45-card *pool*
then *build* a 40-card deck, so **curve is fixed at build time, not pick time** — a good drafter picks
the powerful card and cuts it later if the curve is bad. So the model is curve-blind because it
faithfully imitates curve-blind-*at-pick* humans; there's no pick-time curve signal to learn, and a
curve-specific training objective would model a non-pick-time decision. The right division of labor is
the one this eval already uses: **pick for value** (the model, which beats humans) + **build for curve**
(the deck builder above). (Synergy, unlike curve, *is* pick-time — so that, not curve, is where a
pool-conditioned "marginal deck-WR" objective could still pay off.)
