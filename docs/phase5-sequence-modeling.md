# Phase 5 (plan) — sequence modeling + the capacity spectrum

Status: **plan, not built.** Motivated by the Phase-4 finding that the current model is
data-saturated at ~3–4 sets (`docs/results/phase4-robustness.md`).

## Why: the signal the current model can't see

The current model is **memoryless / per-pick**: it sees the current pool (a set) + the current pack,
and scores. It never sees **what you saw and passed earlier** — so it structurally cannot learn
**signal reading** (which colors/archetypes are *open*, inferred from what flows to you).

A draft is a **partially-observed sequential decision process**: 45 ordered picks where the key
latent (what everyone else is drafting) is only observable through **the sequence of packs you were
passed and what was in them**. Two signals live purely in that history:
- **Openness** — repeatedly being passed good red late ⇒ red is open ⇒ move in.
- **The wheel** — a pack returns ~8 picks later minus 8 cards; what wheels back is direct evidence
  of what's open.

**Key implication:** our ~0.57 saturation may be partly an **input limit, not a capacity limit** —
you can't learn signal-reading from more data if the model is never shown the packs that carry it.
The sequence model adds an *information channel*, not just parameters, which could raise the
task-noise floor that pure capacity scaling cannot.

## The architecture (an evolution, not a rewrite)

Keep the card encoder and pointer head; **replace the pool set-encoder with a causal sequence
encoder over pick steps:**

```
per step t:  set-pool the pack you saw + your pick + pack_no/pick_no  -> step token z_t
[z_1 … z_t]  ->  causal Transformer  ->  h_t   ("read" of table + deck so far)
h_t (query)  ×  current pack card embeddings  ->  pointer head  ->  pick
```

`h_t` summarizes the whole history: it **subsumes** the pool encoder (your picks are in the history)
and **adds** the table-read (the packs are in the history). The pick head is unchanged.

**Data already supports it:** 17lands draft rows carry pack contents + pick, grouped by `draft_id`,
ordered by `pick_number` — the full per-draft sequence (incl. passed cards) is reconstructable from
what we already store; only the DataLoader needs to emit sequences.

## Training options

- **Imitation (as now, sequence-structured):** masked cross-entropy on the human pick at every step.
- **+ Value head:** predict deck win rate from the end-of-draft state (the pool value model, now
  with full history) → shared policy+value trunk.
- **Decision-Transformer framing (clean fit):** condition the sequence on the realized outcome
  (`event_match_wins`) as a return token; at inference condition on a high target → drafts toward
  winning decks. Unifies imitation + "good, not just human" in one architecture, using outcome data
  we already have — cleaner than the bolted-on pick-time blend.

## The spectrum of solutions and their size tradeoff

Trainable parameters (measured for existing configs at input dim D=457 = 73 structured + 384 MiniLM;
estimated for the not-yet-built sequence/text variants). The frozen text encoder contributes **0
trainable params** (precomputed + cached), so it's free until you unfreeze it.

| Option | Trainable params | Train where | What it adds |
|---|---|---|---|
| S, mean-pool (Phase 1) | **~1.0M** | laptop | baseline content model |
| **S, Set Transformer (current best)** | **~2.0M** | laptop | synergy modeling |
| + openness aggregate features (flavor #2) | ~2.0M (+<0.1M) | laptop | cheap "what's open" signal |
| Capacity M | ~7.8M | laptop | more capacity |
| Capacity L | ~15.2M | laptop | more capacity |
| Capacity XL | ~38.9M | laptop (slow) / GPU | more capacity |
| Sequence model — small (d256, 4 layers) | **~4–5M** | laptop | full history / signal-reading |
| Sequence model — large (d512, 6 layers) | **~20–23M** | GPU | full history at scale |
| + value head / decision-transformer return | +~0.1M | — | beyond-imitation, negligible add |
| **Unfreeze the MiniLM text encoder** | **+~23M** trainable | GPU | representation grows with data |
| Upgrade to mpnet text encoder | +~110M | GPU | bigger text representation |
| LLM-as-policy (cards-as-tools, separate paradigm) | **1B–8B+** | GPU | ~1000×; different bet |

### The key insight from these numbers

There are **two clusters**, and the expensive axis is *not* the one people assume:
1. **Architectural richness is cheap.** Going mean-pool → Set Transformer → openness → full sequence
   model + value head spans only **~1M → ~23M** — all still tiny, mostly laptop-trainable. Richer
   *structure* costs surprisingly little.
2. **Representation is the expensive axis, and it's orthogonal.** The jump is **unfreezing /
   upgrading the text encoder** (+23M → +110M *trainable*, and now you're training a transformer →
   GPU) or going LLM-as-policy (1000×). The frozen encoder is free; making it *learn* is the
   regime change.

So the size tradeoff isn't "sequence model vs. not" (that's a few M either way) — it's **"do you
keep the text encoder frozen?"** That single choice dominates the parameter (and cost) budget.

## Recommended ordering

1. **Openness aggregate features** on the current model — cheapest test of "does history signal help
   at all?" (~2M, laptop). If yes, the saturation was partly an input limit.
2. **Full-history sequence Transformer (small)** with per-step imitation (~4–5M, laptop) — the real
   structural upgrade; co-scale with more data.
3. **Value head / decision-transformer return-conditioning** (+~0.1M) — principled
   "good, not just human" + lookahead.
4. **Only if 1–3 still leave accuracy on the table:** unfreeze/upgrade the text encoder (+23–110M,
   GPU) — the genuine reason to scale to the full corpus on a rented GPU.

This keeps everything cheap and on the laptop until step 4, which is the one deliberate jump into
GPU/representation-learning territory.
