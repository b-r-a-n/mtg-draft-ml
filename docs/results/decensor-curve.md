# De-censoring the curve question — is mana curve / castability real, or just censored?

**Verdict: curve/castability has a SMALL but REAL and CONSISTENT marginal effect on winning** — positive
and statistically significant in **8/8 sets** after a leak-free card-power control — but the effect is
modest (no held-out AUC lift). The mechanistic castability model is **validated against real win/loss**;
curve is a genuine, secondary, primarily build-time lever — not a large hidden win driver. (2026-06-28)

## The question and why it was open

"deck win rate is ≈linear in card composition / curve doesn't matter" ([game-data-value-model.md]) was
never really *tested*: game_data only contains decks humans **actually built** (curve-sane, castable), so
the bad-curve region is **censored**, and the outcome metric (`deck_value`, a card-power SUM) is
castability-**blind**. We attacked it from two ends (the [castability.py](../../src/mtg_draft_ml/eval/castability.py)
mechanistic model + this empirical test):

**Does a deck's castability predict its `won`, *after* controlling for card power + player skill?** —
on skill-diverse decks (17lands gives `user_game_win_rate_bucket`), `scripts/decensor_curve.py`.

**The one fix that made it valid (the earlier version was wrong):** the power control must be
**cross-fit out-of-fold**. `power = Σ deck_value`, and `deck_value` is itself fit on the same `won`, so an
in-sample power launders the outcome into the control and biases toward "curve flat." `crossfit_power()`
refits won~composition K-fold so each deck's power comes from a model that never saw it.

## Result — 8 sets, cross-fit power, ~30k decks/set

Marginal effect of each deck's **castability** on `won`, after cross-fit power + skill + on-play + mulligans
(LR χ²>3.84 ≈ p<0.05; band = power-**residualized** high-minus-low-castability win-rate gap):

| set | cast coef/SD | LR χ² | held-out AUC lift | band gap | pip-conc χ² |
|---|---|---|---|---|---|
| DSK | +0.071 | 28.2 | −0.000 | +0.044 | 32.3 |
| MKM | +0.048 | 13.5 | +0.002 | +0.029 | 30.9 |
| MH3 | +0.048 | 13.2 | −0.000 | +0.027 | 21.2 |
| LCI | +0.047 | 12.3 | −0.001 | +0.024 | 7.4 |
| BLB | +0.046 | 12.5 | −0.001 | +0.020 | 11.8 |
| OTJ | +0.033 | 6.2 | −0.001 | +0.019 | 8.9 |
| MOM | +0.029 | 4.8 | −0.001 | +0.012 | 19.5 |
| WOE | +0.029 | 4.7 | −0.000 | +0.007 | 17.2 |
| **mean** | **+0.044** | **11.9** | **~0** | **+0.023** | **18.7** |
| consistency | 8/8 + | **8/8 sig** | — | **8/8 +** | 8/8 sig |

- **Statistically: (a) curve matters.** Castability's effect is **positive and significant in all 8 sets**,
  sign-consistent, with an 8/8-positive residualized win-rate gap (+0.023 mean). **Color concentration
  (`pip_conc`) is even stronger** (mean χ² 18.7, 8/8). So once de-censored *and* power-de-leaked,
  buildability genuinely predicts winning beyond card power — **the censoring conclusion does not hold.**
  This **validates `castability.py` against real outcomes** (the mechanistic mana math already matched
  Karsten to ±2; now it tracks win/loss too).
- **Practically: small.** The held-out **AUC lift is ~0** (≤0.002, mostly slightly negative) and the
  win-rate band gap is ~2 points. So castability is a *real but secondary* contributor — consistent with
  "curve is a deck-**build** decision," not a large lever the pick model is missing.
- **Raw curve height (`avg_cmc`) and land count alone are NOT predictive** (2/8 and 1/8 sig, not
  sign-consistent) — it's *castability / color-coherence* that matters, not "low curve" per se.

## What it does and doesn't reopen

- **Reopens:** the "curve is censored / untestable" caveat is **gone** — curve/castability is measurable
  and real at the deck/build level. `castability.py` is now an outcome-validated descriptor, not just a
  plausible heuristic — it earns its place in the deck-doctor and as the webapp deckbuilder's function score.
- **Does NOT (yet) reopen "pick-time curve is valueless":** this is a **build-level** result (a built deck's
  castability predicts its win rate). It does not show that *picking* for curve helps — and the small
  magnitude suggests a curve-aware picker would add little. The definitive test is the **picker arm**
  (`run_curve_experiment.py`: hold a curve-aware builder fixed, vary only the picker, under the
  function-aware metric). Worth running only if we want to formally close pick-time curve; the effect size
  says don't expect much.

## Caveats
- Built decks are **still self-censored** even for bad players (their avg-cmc spread is barely wider — bad
  players play weak *cards*, not broken *curves*), so this measures curve within the human-built support,
  not the truly-broken region. The `--counterfactual` arm bounds the broken region (model-vs-model only).
- Single-game outcomes are noise-dominated (base AUC ~0.605 from the whole deck); a ~2-point effect is
  about the ceiling of what's detectable here. The **consistency across 8 sets** is what makes it credible,
  not any single set's χ².

Reproduce: `scripts/pod_decensor.sh` (8 sets, cross-fit power, CPU pod) → `scripts/decensor_aggregate.py`.
Per-set + aggregate JSONs in [decensor/](decensor/).
