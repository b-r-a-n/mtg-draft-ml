# Project summary — MTG draft pick ML

A from-scratch research + engineering effort to build a Magic: The Gathering draft-pick model that
**generalizes to cards it has never seen** (new sets) and makes **good** picks, not merely
human-like ones. This is the consolidated story; see `docs/research/` for the literature synthesis,
`docs/results/` for every experiment, and `docs/design-decisions.md` for the rationale (DD-001…007).

## The problem

Given the cards already drafted (the **pool**) and the cards on offer (the **pack**), pick one.
Hard constraints: (1) generalize to brand-new sets' cards at release, and (2) optimize for *winning*,
not just imitating the average drafter. Data: 17lands public draft logs + Scryfall card attributes.

## The model (the recommended recipe)

```
each card ─► [structured features ; frozen MiniLM oracle-text embedding] ─► shared MLP ─► card vec
pool (set) ─► Set Transformer (self-attention + pooling) ─► context vec
pack        ─► [card vecs]
context · pack vecs ─► masked softmax over the pack ─► pick
```
- **Content-based card encoder (DD-001)** — never a per-card ID, so unseen cards are scored from
  their text+stats. This is the entire generalization mechanism.
- **Set Transformer pool encoder (DD-002, Phase 2)** — models synergy across the pool; beats
  mean-pooling.
- **Pointer / masked-softmax head** — scores a variable candidate set; handles any pack.
- **Training:** in-pack cross-entropy (≡ in-pack InfoNCE) + **IWD advantage-weighting (τ≈0.03)** to
  bias toward winning picks; optional **aggressiveness dial** at inference (blend predicted card
  quality into the logits). ~2–15M params; trains on a laptop or a ~$0.20 rented GPU.

## Headline results (leave-one-set-out, train on 4 sets → hold out a 5th, ~98% novel cards)

| capability | result |
|---|---|
| Generalization to unseen cards (top-1) | **~0.57** vs 0.233 random floor, ~0.55 published bar |
| Is the human pick in the model's top-k? | **top-3 0.92, top-5 0.98** |
| Does it use pool context like humans? | model follows pool-lean **0.92** ≥ human 0.90 |
| "Good, not just human" (takes highest-IWD card) | baseline **0.257 < human 0.278**; advantage-weighted **0.295 > human** |

## What we learned (the research arc)

1. **Representation is the whole generalization lever.** Architecture/scale barely move *in-set*
   accuracy; the content encoder + multi-set training is what reaches the unseen-card bar. Semantic
   text embeddings only help *with set diversity* (single-set: text hurt; multi-set: text best).
2. **~0.58 top-1 is a human-noise ceiling, not a model limit** — proven four ways: more data, more
   capacity, sequence modeling (signal-reading), and better pool-conditioning all plateau there; and
   the model already rates the human pick top-3 92% of the time and follows pool context as well as
   humans. The residual is genuine human disagreement.
3. **So the goal isn't predicting humans better — it's picking *better*.** IWD advantage-weighting
   flips the model from below-human to above-human on winning-card selection. That's the real win.
4. **Negative results that saved effort:** global-negative InfoNCE hurt (the pack *is* the right
   negative set); feature standardization hurt; naive capacity scale-up "collapsed" until LR warmup
   fixed it (an optimization, not capacity, issue).

## Engineering

- **Pipeline:** 17lands CSV → compact integer-index Parquet (streaming) → Scryfall join → frozen
  text-embedding cache → train/eval. Hugging Face Datasets as the public data hub.
- **Cloud:** fully CLI-driven RunPod (create/bootstrap/train/teardown via `runpodctl` + SSH);
  `uv`-based bootstrap. Root-caused a class of silent crashes: `uv run` auto-sync pulled a torch CUDA
  build (cu130) too new for host drivers → fixed by pinning torch to cu124 for Linux in
  `pyproject.toml`. Total cloud spend across the whole project: **< $1**.
- **Tests:** 46+ covering preprocessing, encoders, heads, losses, eval metrics, sequence causality,
  and the deployable drafter.

## Deployment

`mtg_draft_ml.deploy.Drafter` loads a bundle and exposes
`pick(pool, pack, aggressiveness)` → ranked picks. The **aggressiveness dial** trades human-likeness
for win-rate-seeking (0 = human-like, higher = greedier on card quality). Retarget to a new set by
rebuilding its content matrix — the content-based weights transfer to unseen cards.

## What's next (optional)

- Combine advantage-weighting **+** the inference blend for a tunable "good-not-just-human" frontier.
- LLM-teacher distillation, especially **cold-start for brand-new sets** (no 17lands data on day one)
  — the one niche where an LLM ranking has no competitor.
- Product surface: an in-browser card-value editor reusing the encoder (DD-005), or a draft assistant.
