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

## Verdict

The WR-agreement gains **do** translate to outcome-better *decks* than humans — and the result holds up
under a realistic 2-color + curve deck-build (+0.029, 74% of seats; non-circular), which also strips the
greedy "bomb-hoard" policies of the edge the naive metric handed them. The remaining rungs are
**synergy/mana-base** modeling and a played-out win simulator, if this becomes load-bearing.
