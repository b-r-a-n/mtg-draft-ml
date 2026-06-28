# Does the pick model read archetype synergy? — yes (Detective probe)

**The deployed pick model genuinely drafts pool-conditioned synergy** — controlling for color, a Detective
payoff goes from a 4th-choice afterthought to the **#1 pick** once the pool is Detective-heavy, while an
off-synergy bomb collapses. This corrects an earlier overstatement that the model is "synergy-blind," and
it refines (does not contradict) the outcome-linearity bound. (2026-06-28, MKM deployed model.)

## The question

A user observed their drafted deck "didn't seem aware of the Detective synergy" (MKM). Separately we had a
bound — **deck win rate is ≈ linear in card composition** ([game-data-value-model.md]) — that was loosely
read as "the model ignores synergy." Those are different claims: one is about whether synergy predicts
*winning*; the other is about whether the *pick model* drafts toward it. This probe tests the second.

## Method (controlling for color)

A Detective-heavy pool is also a *color*-heavy pool, and the model has strong color commitment — so a naive
test conflates the two. We score a Detective **payoff** in a fixed 7-card pack under three pools and isolate
the Detective effect from color:

- **empty** pool (baseline),
- **+13 Detectives** (white Detective creatures = enablers),
- **+13 same-color non-Detectives** (white non-Detective creatures = the color control).

Payoff: **Case of the Pilfered Proof** — the set's Detective anthem, and a clean test because its
context-free `deck_value` is **−0.039** (the model rates it *low* on its own). Pick-% = masked softmax over
the pack from the deployed `MKM.onnx`, run in-browser (the real deployment path).

## Result — Detective-specific, not color

Pick-% of each pack card under each pool:

| pack card | Detective? | empty | **+Detectives** | +color control |
|---|---|---|---|---|
| **Case of the Pilfered Proof** (payoff) | ✓ | 0.080 (4th) | **0.336 (1st)** | 0.115 (4th) |
| Agency Outfitter (a Detective) | ✓ | 0.052 | **0.130** | 0.064 |
| Absolving Lammasu (white, non-Det) | ✗ | 0.038 | 0.099 | 0.135 |
| Anzrag, the Quake-Mole (RG bomb) | ✗ | **0.481** | 0.157 | 0.308 |
| Archdruid's Charm (G) | ✗ | 0.211 | 0.147 | 0.205 |
| Assassin's Trophy (BG) | ✗ | 0.111 | 0.086 | 0.111 |

- **Both Detective cards jump** with Detectives (the payoff 4×, to #1; logit −0.21 → +0.85); the
  **off-synergy bomb Anzrag collapses** 0.48 → 0.16 as the synergy cards eat its share.
- **The color control does *not* reproduce it**: with same-color non-Detectives the payoff barely moves
  (0.08 → 0.12, still 4th), and the non-Detective white card (Lammasu) rises *more* than the Detectives do.
  So the boost is **Detective-type-specific**, not "white cards got better."

## Interpretation — pool-conditioned synergy IS learned (and it ties up a loose end)

The Set Transformer pool encoder learned Detective synergy by **imitating good drafters**, who pick payoffs
once they have the enablers. This **refines, not contradicts**, the prior findings:

- **Outcome linearity still holds.** "Synergy doesn't predict *winning* beyond per-card value"
  ([game-data-value-model.md], `probe_deck_outcome.py`) is a statement about win-rate, and stands. The pick
  model can still *draft* synergy (imitation) even if synergy isn't a big independent win driver.
- **This explains why pick-time buildability was FLAT** ([play-prob-pick-time.md]). The model *already*
  pool-conditions (synergy, coherence) from imitation, so bolting on an explicit `P(played)` pool signal at
  pick time was redundant — exactly the flat result we saw. "The model already drafts coherently" was more
  literally true than credited.

So the model is a **value drafter that does pick up in-context synergy**, not a synergy-blind one.

## Why a draft can *feel* synergy-unaware

**Cold-start.** A payoff like Case of the Pilfered Proof is *correctly* rated low until you actually have
Detectives (8% / 4th at an empty pool — it's weak on its own). The synergy only switches on once you commit,
so early picks won't look Detective-y and the overlay swings toward payoffs **after** you take a few
enablers. Take 2-3 Detectives early and the payoffs' pick-% climbs.

## Caveats
- One synergy, one set, one payoff (+ one corroborating Detective card) — strong evidence the model reads
  Detective synergy, not a comprehensive multi-archetype eval. Other synergies may be learned less well.
- This is about the **pick** model (in-draft); deck *building* uses the separate `P(played)`/curve tools.
- Pick-% is a within-pack softmax, so it measures *relative* preference shifts (which is the right unit for
  "did the pool change what it wants").

Reproduce: drive the deployed `webapp/model/MKM.onnx` with the three pools (Scryfall `type_line`/`oracle_text`
identify Detectives + payoffs); pick-% = softmax over `session.run({pool, pool_mask, pack, pack_mask}).logits`.
