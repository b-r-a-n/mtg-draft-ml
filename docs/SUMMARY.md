# Project summary — MTG draft-pick ML (final)

A from-scratch research + engineering effort to build a Magic: The Gathering draft-pick model that
**generalizes to cards it has never seen** (new sets) and makes **good** picks — winning ones, not
merely human-like ones — then to ship it. This is the consolidated, final story; see `docs/research/`
for the literature synthesis, `docs/results/` for every experiment, `docs/roadmap.md` for the phase
plan, and `docs/design-decisions.md` for the rationale (DD-001…007).

## The problem

Given the cards already drafted (the **pool**) and the cards on offer (the **pack**), pick one. Two
hard constraints: (1) generalize to a brand-new set's cards on release day, and (2) optimize for
*winning*, not imitating the average drafter. Data: 17lands public draft + game logs + Scryfall.

## The model (the deployed recipe)

```
each card ─► [73 structured features ; frozen MiniLM oracle-text embedding (384d)] ─► shared MLP ─► card vec
pool (set) ─► Set Transformer (self-attention + pooling) ─► context vec
pack        ─► [card vecs]
context · pack vecs ─► masked softmax over the pack ─► pick     (+ deck_value aggressiveness dial)
```
- **Content-based card encoder (DD-001)** — never a per-card ID, so unseen cards are scored from their
  text+stats. This is the entire generalization mechanism.
- **Set Transformer pool encoder (DD-002)** — models cross-pool synergy; beats mean-pooling.
- **Masked-softmax / pointer head** — scores a variable candidate set; handles any pack.
- **Training:** in-pack cross-entropy (≡ in-pack InfoNCE) + a **win-rate objective** — either IWD
  advantage-weighting (τ≈0.03) or, in the deployed model, **good-player filtering + a composite-WR
  softmax teacher** (GIH+IWD+ALSA) distilled into the encoder — on **~19 relevant sets**. Big-net deploy
  config emb512/h1024/L4 (~8.7M params); trains on a laptop or a ~$0.30 rented GPU.
- **Inference dial:** blend a per-card `deck_value` (a de-confounded game_data card value) into the
  logits — 0 = human-like, higher = greedier on card quality.

## Headline results

| capability | result |
|---|---|
| **Generalize to unseen cards** (top-1, leave-one-set-out, ~98% novel cards) | **~0.57** (flagship DSK 0.574) vs 0.233 random / ~0.55 published bar; honest rotated **~0.54** |
| Human pick in the model's top-k | **top-3 0.92, top-5 0.98** |
| **Win-rate agreement** (takes the highest-WR card; the axis with headroom) | **0.326** on ~19 relevant sets, **> average human 0.298**, across 4/4 rotated holdouts |
| **Do its decks actually win more?** (outcome eval, non-circular) | model's drafted decks beat humans' by **+0.063 est. deck-WR in 95% of seats** (+0.049 with the learned deckbuilder) |
| Curve/castability really matters? (de-censored) | **small but real & consistent (8/8 sets)** — a build-time lever, not a pick-time miss |

## What we learned (the research arc)

1. **Representation is the whole generalization lever.** Architecture/scale barely move *in-set*
   accuracy; the content encoder + multi-set training is what reaches the unseen-card bar. Text
   embeddings help only *with set diversity* (single-set: text hurt; multi-set: text best).
2. **~0.58 top-1 is a human-noise ceiling, not a model limit** — shown four ways (more data, more
   capacity, sequence/signal-reading models, better pool-conditioning all plateau there) and confirmed
   by a human-disagreement probe (pairwise human agreement collapses mid-pack, exactly where the model
   errs). So predicting humans *better* is the wrong goal.
3. **The axis with headroom is win-rate — and corpus breadth broke its ceiling.** WR-agreement plateaued
   at 0.29–0.31 because the corpus was capped at ~7 sets, not because of the model. The best recipe on
   **~19 *relevant* draft sets** reaches **0.326** (> human 0.298), 4/4 holdouts. A curation pass proved
   it's **relevance, not count**: dropping the off-distribution remaster/Masters sets (SIR/PIO) is what
   matters; padding past ~19 *hurts*. ([wr-scaling.md], [game-data-value-model.md])
4. **A de-confounded value signal exists but isn't the lever.** `deck_value` — a per-card value from a
   `won ~ deck-composition` regression on 17lands `game_data` — is genuinely different from GIH-WR
   (Sp≈0.6) and a slightly better *target* at small corpus, but at 19 sets it's **subsumed by corpus
   breadth**. Kept as the webapp aggressiveness dial + a de-confounded eval, not a training target.
5. **The model drafts outcome-better decks than humans** (non-circular: GIH-trained, game_data-scored)
   — and the edge survives a realistic deck build. ([outcome-eval.md])
6. **"Beyond imitation" is bounded — and the curve question is now resolved.** Deck win rate is ≈linear
   in card composition, so per-card value is near the achievable ceiling and a smarter pool-conditioned
   *pick* objective has ~no extra signal (pick-time buildability weighting is flat once in-distribution).
   The open caveat — game_data is range-restricted to curve-sane built decks, so curve was *censored*,
   not tested — is **closed**: with skill-diverse decks + a cross-fit leak-free power control, castability
   has a **small but real, consistent positive** effect on winning (**significant in 8/8 sets**, mean coef
   +0.044). So "curve doesn't matter" was a censoring artifact and the mechanistic castability model
   (`castability.py`, matches Karsten ±2) is **outcome-validated** — but the effect is **small** (~0
   held-out AUC lift): **curve is a real *build*-time lever, not a pick-time miss.** "Pick for value,
   build for curve" now stands empirically. ([decensor-curve.md], [play-prob-pick-time.md])
7. **Buildability lives in the build decision, and it shipped.** `P(played | pool)` — learned from
   deck-vs-sideboard — is strongly pool-dependent (AUC 0.81→0.93 with pool context) and recovers
   castability with no feature engineering. It **helps at build time** (+0.049 deck-WR as the deckbuilder)
   but is **flat at pick time**. It's now the **webapp's deck builder**, collapsing rainbow value-piles
   into ~2-color decks. ([play-prob.md])
8. **Negative results that saved effort:** global-negative InfoNCE hurt (the pack *is* the right negative
   set); feature standardization hurt; naive capacity scale-up "collapsed" until LR warmup fixed it
   (optimization, not capacity); the value-RL / lookahead ambitions are bounded out on this data.

## Engineering

- **Pipeline:** 17lands CSV → compact integer-index Parquet (streaming) → Scryfall join → frozen
  text-embedding cache → train/eval; per-game `game_data` → `won`-regression `deck_value`. Hugging Face
  Datasets as the public hub.
- **Cloud:** fully CLI-driven RunPod (create/bootstrap/train/teardown via `runpodctl` + SSH), `uv`-based.
  Root-caused silent crashes from `uv run` auto-sync pulling a too-new CUDA torch → pinned cu124 for
  Linux. Added a **CPU-only bootstrap** for sklearn/pandas jobs after the CUDA wheel hung a session.
  Total cloud spend across the whole project: a **few dollars**.
- **Tests:** 70+ covering preprocessing, encoders, heads, losses, eval metrics, sequence causality, the
  deployable drafter, the hypergeometric castability model, and the in-browser deckbuilder parity.

## Deployment

- **`mtg_draft_ml.deploy.Drafter`** — loads a bundle, `pick(pool, pack, aggressiveness)` → ranked picks.
- **Static draft-pod webapp** (`webapp/`, GitHub Pages) — an 8-seat pod where you draft one seat while
  the model auto-drafts 7, scoring entirely **in-browser via onnxruntime-web (WASM)** — no server. All 8
  shipped sets run the **big-net model trained on the curated ~19-set corpus**, int8-quantized. The
  end-of-draft deck is built by the learned **`P(played|pool)` buildability deckbuilder** (sklearn HGB
  exported as ~12-line-walkable JSON trees), yielding human-like ~2-color decks, with a per-seat
  aggressiveness dial on `deck_value`.

## Bottom line

The research arc is **complete**: the pick model is at its characterized ceilings (top-1 ≈0.58 human
noise; WR-agreement ≈0.326 corpus-breadth ceiling, **beating average human**), it drafts **outcome-better
decks than humans**, the curve question is **resolved** (real-but-small, build-time), and the best recipe
is **deployed end-to-end** in a live in-browser webapp. Every modeling lever with headroom has been pulled
or explicitly bounded out; what remains is **product**, not new modeling. The one open *research* probe —
a curve-aware picker arm — is judged not worth running, since the de-censored effect is too small to
expect a pick-time gain.

**Product track underway — tools over the learned embedding space.** The content encoder's per-card
latent (trained on draft value) is a reusable asset: cosine-near ≈ similar draft role/archetype.
`eval/card_tools.py` + `scripts/export_embeddings.py` expose **card similarity ("plays like X")**,
**cross-set analogues** ("the BLB version of a DSK card"), and an **unsupervised archetype map** — all
verified coherent (e.g. a DSK black bomb's neighbors are other black bombs; its BLB analogue is a BLB
black bomb). Next on this track: a **deck doctor** ("rate my deck" = power + buildability + the now
outcome-validated castability), and optionally surfacing similarity inside the webapp.
