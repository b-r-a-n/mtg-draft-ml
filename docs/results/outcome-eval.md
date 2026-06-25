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

| policy | est. deck-WR | Δ vs human | beats human |
|---|---|---|---|
| `deckvalue_greedy` (oracle: take highest-β each pack) | 0.773 | +0.134 | 100% |
| `gih_greedy` (take highest GIH-WR each pack) | 0.738 | +0.099 | 99% |
| **model** (deployed, GIH-trained) | **0.703** | **+0.063** | **95%** |
| *human* (recorded picks) | 0.639 | — | — |
| `random` | 0.531 | −0.108 | 3% |

**The model's drafted decks beat the human's by +0.063 estimated deck-WR, in 95% of drafts — and
non-circularly** (GIH-trained, deck_value-scored). This is the signal the WR-agreement work couldn't
give: the pick-quality gains translate into decks the *outcome* model rates higher than the people who
actually played them. The model captures ~47% of the human→oracle gap (0.639 → 0.703 → 0.773).

## Honest caveats (what this does and does NOT show)

- **The metric is a card-power sum, not a deck simulator.** "est. deck-WR" = σ(Σ top-23 β) rewards raw
  card value and ignores curve, mana, synergy, and playability — which is why the **greedy rating
  policies beat the model** (`gih_greedy` 0.738 > model 0.703). A real win-rate eval would need
  mana/curve/sequencing. So this validates *"the model picks winning-er cards than humans"* (a
  card-power claim), not *"the model builds optimal decks."* The greedy policies "win" the metric by
  maximizing raw value at the cost of a real deck's balance.
- **On-rails packs.** Policies draft the *recorded* pack sequence, so their own picks don't change what
  wheels — a standard counterfactual approximation, not a full pod simulation.
- **In-distribution.** The deployed model trained on DSK (among 20 sets); this is a deck-quality
  comparison (model vs human on the same packs), not a generalization test.

## Verdict

The WR-agreement gains **do** translate to outcome-better *picks* than humans (decisive, non-circular:
+0.063, 95% of drafts), closing the loop the project kept deferring. The remaining honest gap is the
**deck-strength metric itself** — a card-power proxy, not a curve/mana/win simulator — so the headline
is "the model picks cards that build higher-outcome-value decks than humans," with a real deck-WR
simulator as the next rung if this becomes load-bearing.
