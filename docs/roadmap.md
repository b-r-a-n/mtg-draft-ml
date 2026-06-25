# Roadmap

Local-first: Phases 0–2 run on the M1 dev machine; the multi-set generalization pretrain
(Phase 3+) is the step intended for a rented cloud GPU. Each phase is shippable and comparable
to the last via the eval harness stood up in Phase 0.

## Status (2026-06-25) — what's settled, and how it reshapes the rest

Phases 0–3 shipped and the **game_data value model (Phase 4)** is now fully characterized
(`docs/results/`). The headline updates change what's worth doing next:

- **Corpus breadth is the real lever — and it broke the ceiling.** WR-agreement plateaued at 0.29–0.31
  not because of the model but because the corpus was capped at ~7 sets. The best recipe on **~19
  *relevant* draft sets** lifts WR-agreement to **0.326** (vs ~0.30 at 7), **confirmed across 4 rotated
  holdouts**, beating average-human (0.298). Relevance matters — padding with remaster/Masters sets
  *hurts* (peaks ~19, declines at 22). → **train the deployable on as many relevant sets as available,
  not 4–7.** ([wr-scaling.md](results/wr-scaling.md))
- **`deck_value` (the de-confounded game_data signal) is real but not the lever.** A genuinely
  different, less-confounded card value (Sp(β,IWD)≈0.6, face-plausible), and a slightly better *target*
  at small corpus — but at 19 sets it's **subsumed by corpus breadth** (no compounding). Keep it as the
  webapp aggressiveness dial + a de-confounded eval, **not** as a training target.
  ([game-data-value-model.md](results/game-data-value-model.md))
- **The model drafts outcome-better decks than humans** (non-circular), and the edge survives a
  realistic 2-color + curve deck-build (+0.029, 74% of seats). ([outcome-eval.md](results/outcome-eval.md))
- **Hard ceiling on "beyond imitation": deck win rate is ≈ linear in card composition.** No
  curve / castability / synergy interaction is recoverable as outcome-predictive structure (gradient
  boosting ≈ linear; a targeted synergy test adds nothing held-out) — single-game outcomes are
  noise-dominated (whole-deck AUC ≈ 0.61) and built-deck data is range-restricted. **Per-card value is
  near the achievable ceiling.** Curve is a deck-*build* decision, not a pick decision (the model AND
  good-player picks are both curve-blind), so **"pick for value, build for curve" is correct by
  design**, and a smarter pool-conditioned pick objective has ~no extra signal to learn.
- **Shipped:** a static in-browser **draft-pod webapp** (WASM) running the deployed model on GitHub
  Pages ([webapp/](../webapp/README.md)).

⇒ The productive remaining levers are **corpus breadth / data curation** and the **product surface
(webapp + aggressiveness dial)** — *not* a smarter per-pick objective or a value/lookahead RL agent
(Phase 4's stretch goals are now bounded out — see below).

## Phase 0 — Baseline + data pipeline
**Goal:** reproduce a known result and stand up the data join + eval harness as a yardstick.
- 17lands `draft_data` ingestion (streaming) → compact integer-index Parquet.
- Scryfall `oracle_cards` join, deduped by `oracle_id`.
- Baseline: masked-softmax MLP over one-hot collection+pack vectors
  (Statistical-Drafting / Draftsim style).
- Eval harness: top-1 accuracy, mean pick distance (MTPD), per-pick-position accuracy.
- **Targets:** reproduce NNetBot ~48.7% on Draftsim M19; ~70% top-player agreement on a recent
  17lands set.

## Phase 1 — Content-feature card encoder (generalization core)
**Goal:** replace one-hot IDs with the ID-free content encoder so the model can score unseen cards.
- Card encoder = structured features + frozen sentence-transformer oracle-text embedding → MLP.
- Plug into the masked-softmax head.
- Stand up the **leave-one-set-out / temporal held-out-new-set** benchmark; report unseen-card
  accuracy vs the ~22% random floor and the ~55% published bar, including a **release-day**
  variant that zeroes meta/usage features.

## Phase 2 — Set/context encoder + contextual contrastive head
**Goal:** capture synergy and take the largest accuracy jump.
- Deep Sets pool encoder → upgrade to 1–2 layer Set Transformer.
- Cross-attention / pointer scoring head; train with masked in-pack InfoNCE (fallback CE).
- Inject `pick_number` / `pack_number` + environment-summary vector.
- Ablate: pool self-attention vs mean pooling, text-encoder family, numeric-magnitude augmentation.

## Phase 3 — Good-not-just-human signal *(cloud for the multi-set pretrain)*
**Goal:** debias from conformity; bias picks toward winning decks.
- Win-rate-weighted/filtered training; auxiliary heads (adjusted card WR, ALSA).
- **Distillation hooks** (see [design-decisions.md](design-decisions.md) DD-004):
  - *Soft-label distillation* — a stronger/ensemble/win-rate-aware teacher provides a full
    ranking over the pack; student matches it (KL) on top of the human label.
  - *Teacher-with-leaky-features → release-day student* — train the teacher with meta/win-rate
    features, distill into a student that only sees release-day-available inputs.
- Metrics: WR-agreement, estimated deck-WR; verify the bot deviates from crowd consensus toward
  higher-WR cards without losing sanity/legality.

## Phase 4 — Beyond imitation *(DONE / characterized → bounded out)*
**Goal:** picks that exceed human demonstrators, if a reliable value signal exists.
**Outcome:** built and fully characterized via [game-data-plan.md](game-data-plan.md) — the regression
card-value model over 17lands `game_data` (`deck_value`), the WR-agreement scaling study, the outcome
eval, and the nonlinearity/synergy bound. Conclusions (see Status above + `docs/results/`):
- ✅ **`deck_value` works as a less-confounded signal** but is **subsumed by corpus breadth** as a
  training target (Steps 0–2). The ceiling was broken by **data breadth, not de-confounding** — so the
  old framing ("plateau capped by the confounded GIH-WR proxy") was wrong; the cap was corpus size.
- ✅ **The model already exceeds human demonstrators** on outcome-scored decks (non-circular,
  curve-aware build) — so "beyond imitation" is *achieved* on the metric we can measure.
- ⛔ **The value-RL / lookahead ambitions are bounded out.** Deck win rate is **≈ linear in card
  composition** (no curve/castability/synergy interaction is outcome-predictive), so a deck-strength
  *value* model over pools, advantage-weighted offline RL, and JueWuDraft-style policy/value + lookahead
  have **~no extra outcome signal to learn** beyond per-card value. Not worth building **on this data**.

**What would actually unlock the pool-dependent dynamic** (curve/castability/synergy) — i.e. the signal
the current `won`-over-composition data lacks:
- **Deckbuild signal (cheapest, already in `game_data`) — ✅ PROTOTYPED, it works:** `deck_<card>` vs
  `sideboard_<card>` encodes what good players *cut* — the build-time curve/castability judgment,
  directly. `P(played | pool)` is **strongly pool-dependent and learnable** (AUC 0.81→0.93 when adding
  pool context, vs *zero* gain for `won`), and recovers castability with no feature engineering (a white
  2-drop's play-prob: 0.17 in a white-splash deck → 0.90 when white-committed).
  ([play-prob.md](results/play-prob.md)) **Folding it into the *pick* policy doesn't help** (flat/worse
  — OOD on partial pools + the model already drafts coherently); the signal belongs at the deck-*build*
  step, reinforcing "pick for value, build for buildability." A build-time deckbuilder using
  `P(played|pool)` (replacing the eval's hand-coded 2-color+curve heuristic) is the natural next use.
- **A game simulator / self-play (principled, expensive — the long-standing gate):** removes the
  range restriction (built decks are all castable/sane) by playing out *arbitrary* decks, incl. bad
  curves, so the interaction becomes observable. This is the only path to a true value/lookahead agent.
- **Lower-noise in-game proxies:** predict tempo/castability outcomes (mulligans, `num_turns`,
  drawn-but-stranded from the per-card `drawn_/opening_hand_` columns) instead of end-of-game `won` —
  higher signal-per-game for the specific dynamic.

## Phase 5 (plan) — Sequence modeling + capacity spectrum
**Goal:** model the draft as a sequence (history of packs seen / cards passed) to capture
signal-reading the memoryless model can't — the architectural lever that actually *uses* more data.
Full plan + the parameter-scale tradeoff table: [phase5-sequence-modeling.md](phase5-sequence-modeling.md).
Key finding: architectural richness (mean→set→sequence+value) is cheap (~1M→~23M, mostly laptop);
the expensive, orthogonal axis is unfreezing/upgrading the text encoder (+23–110M, GPU).

## Distillation track — soft labels for sample efficiency + cold start *(DD-004)*
Two tracks in `src/mtg_draft_ml/distill/`; full design in [coldstart-distillation.md](coldstart-distillation.md).

**Ensemble-of-seeds soft-label distillation (DD-004 #1)** — the higher-leverage one: the flat
data/capacity scaling says the *objective* (a lossy one-hot label), not the data, is the bottleneck,
and `pick_advantage_weights` only reweights the hard label without ever reshaping the target. KD term
`pack_distillation_kl` + `topk_renormalize` in `training/losses.py`; opt-in `teacher`/`distill_*`
hook in `train_loop`; `run_ensemble_distill` (`scripts/pod_distill.py`) trains K seed teachers and
compares CE baseline vs CE+KD vs the ensemble. Judge on WR-agreement + ranking quality (top-3/5,
MTPD) and `--train-frac` (KD on a fraction vs CE on all) — top-1 is likely noise-ceilinged.

**WR-softmax soft target (DD-004 #1, good-not-just-human as a *dense* target)** — `pick_advantage_weights`
only *reweights* the hard-label example by a scalar; `WRSoftmaxTeacher` reshapes the *target* into the
win-rate ranking over the pack. `run_wr_distill` (`scripts/pod_wr_distill.py`) compares CE vs CE+WR-KD
(dense) vs CE+advantage (scalar) on WR-agreement / avg-pick-WR — does reshaping the target beat
reweighting the example? Reuses the same KD term + `train_loop` hook (teacher-agnostic plumbing).

**Leaky-feature → release-day teacher (DD-004 #3)** — a teacher with per-card win rate as an input
feature (`augment_with_winrate` appends WR + a rated flag) learns a win-rate-informed *contextual*
policy, distilled into a student that sees only release-day inputs. `run_leaky_distill`
(`scripts/pod_leaky_distill.py`) compares baseline / distilled / the leaky teacher (ceiling) on
WR-agreement. `CompositeTeacher` averages any teachers' `mean_probs` into one target (denoise + good-
not-just-human + leaky), reusing the same KD hook. Next: distill a search/lookahead policy
(DD-004 #5, gated on a simulator — Phase 4).

### Cold-start track — LLM-teacher distillation for brand-new sets *(DD-004 #4)*
**Goal:** provide pick guidance on a set's release day, when there is **zero** 17lands data — the one
niche where an LLM that has read the cards has signal nothing else does. Scaffold built; see
[coldstart-distillation.md](coldstart-distillation.md).
- The teacher emits a **17lands-shaped ratings file** (`[{"name","llm_quality"}]`) so it is a drop-in
  for the data that doesn't exist yet — flows through `align_winrates` / `WRMeter` / the blend dial
  unchanged (`src/mtg_draft_ml/distill/`).
- Experiment (`scripts/pod_coldstart.py`): train on N sets, hold out a 5th as "the new set", compare
  no-data baseline vs LLM cold-start blend vs real-17lands oracle; report the % of the oracle gap the
  LLM closes + teacher↔real rank correlation.
- Next: per-pack soft-label KL distillation (DD-004 #1) if the per-card signal proves out.

## Parallel track — Card-design / editor tool
**Goal:** interactive in-browser tool that embeds **novel typed oracle text** and estimates a
card's power/value. (See [design-decisions.md](design-decisions.md) DD-005.)
- Reuse the Phase-1 card encoder; add a value head trained against (adjusted) win-rate.
- Client-side: start with an off-the-shelf small text encoder via `transformers.js` / ONNX-web;
  distill into a smaller text encoder only if download/latency demands it.
- **Must** feed the value head explicit cost/stat numbers — the text embedding alone is not
  power-aware. Treat outputs as relative/uncertain (no ground truth for never-played cards).
