# Roadmap

Local-first: Phases 0–2 run on the M1 dev machine; the multi-set generalization pretrain
(Phase 3+) is the step intended for a rented cloud GPU. Each phase is shippable and comparable
to the last via the eval harness stood up in Phase 0.

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

## Phase 4 (stretch) — Beyond imitation
**Goal:** picks that exceed human demonstrators, if a reliable value signal exists.
- Deck-strength value model over finished pools.
- Advantage-weighted offline RL first; only then policy/value + lookahead (JueWuDraft-style),
  kept as a separate model from the human-pick predictor.
- *Search distillation:* distill the slow lookahead policy into the fast reactive network.
- Gated on a usable simulator or rich enough outcome data; treat as experimental.

## Phase 5 (plan) — Sequence modeling + capacity spectrum
**Goal:** model the draft as a sequence (history of packs seen / cards passed) to capture
signal-reading the memoryless model can't — the architectural lever that actually *uses* more data.
Full plan + the parameter-scale tradeoff table: [phase5-sequence-modeling.md](phase5-sequence-modeling.md).
Key finding: architectural richness (mean→set→sequence+value) is cheap (~1M→~23M, mostly laptop);
the expensive, orthogonal axis is unfreezing/upgrading the text encoder (+23–110M, GPU).

## Parallel track — Card-design / editor tool
**Goal:** interactive in-browser tool that embeds **novel typed oracle text** and estimates a
card's power/value. (See [design-decisions.md](design-decisions.md) DD-005.)
- Reuse the Phase-1 card encoder; add a value head trained against (adjusted) win-rate.
- Client-side: start with an off-the-shelf small text encoder via `transformers.js` / ONNX-web;
  distill into a smaller text encoder only if download/latency demands it.
- **Must** feed the value head explicit cost/stat numbers — the text embedding alone is not
  power-aware. Treat outputs as relative/uncertain (no ground truth for never-played cards).
