# Architecture

The model we're building, component by component. This is the distilled design; see
[research/synthesis.md](research/synthesis.md) for the evidence behind each choice and
[design-decisions.md](design-decisions.md) for the alternatives we rejected.

## One sentence

Encode each card from its content (never an ID), encode the drafted pool with a
permutation-invariant set encoder, and score each card in the current pack with a
context-conditioned pointer head, trained with a masked in-pack contrastive loss and biased
toward winning decks via win-rate weighting.

## Data flow

```
                       ┌─────────────── card encoder (shared) ───────────────┐
 Scryfall attrs ──►  structured features ─┐
                                          ├─► MLP ─► card embedding (d≈512)
 oracle text  ──► frozen text encoder ────┘            │
                                                       │  (same encoder applied to every card)
   pool cards ──► [card emb, …] ─► set encoder ─► pool/context vector ─┐
                                                                       ├─► pick head ─► P(pick | pool, pack)
   pack cards ──► [card emb, …] ───────────────────────────────────────┘
```

## Components

### 1. Card encoder (the generalization core)
`[structured_features ; text_embedding] → shared MLP → card embedding`

- **Structured features:** CMC, colors/identity (multi-hot), type/subtypes, P/T/loyalty
  (with sentinels for `*`/`X`), rarity (ordinal), keywords (multi-hot), produced mana.
- **Text embedding:** a **frozen** sentence-transformer over normalized oracle text. This is
  the component that buys zero-shot understanding of new abilities. Precompute once per unique
  card (keyed by `oracle_id`) and cache; the heavy model never runs during training.
- **Gotchas:** encode multi-faced cards per face and pool; pull numeric magnitudes
  ("deal 3 damage") out into explicit features since text encoders under-weight numbers.
- **Hard rule:** no learnable per-card ID embedding in the candidate path — that reintroduces
  cold-start and forces per-set retraining.

### 2. Context encoder (the pool)
Permutation-invariant over the already-picked pool.
- Baseline: **Deep Sets** mean-pooling of card embeddings.
- Upgrade: **1–2 layer Set Transformer** (self-attention + attention pooling) to model
  synergy/anti-synergy.
- Concatenate non-set scalars to the pooled vector without breaking invariance: `pick_number`,
  `pack_number`, and a fixed "format/environment summary" vector.

### 3. Pick head (variable candidate set)
One logit per card **present in the pack**, via pointer/cross-attention scoring:
`logit_i = f(pool_context, card_i)` (dot product, optionally refined by a small MLP), then a
**masked softmax over exactly the pack**. Natively handles any pack size and unseen contents.
This is the same "describe options in context, attend, point at one" pattern as LLM
tool-calling — just specialized and tiny (see [design-decisions.md](design-decisions.md) DD-003).

### 4. Training objective
- **Base:** masked, row-wise **InfoNCE** over in-pack cards (picked = positive, rest = negatives).
  Fallback: masked-softmax cross-entropy.
- **Beyond imitation:** weight/filter examples by drafter skill (`event_match_wins`,
  `user_game_win_rate_bucket`); auxiliary heads regressing confounder-adjusted card win-rate
  and ALSA. Keep meta/usage features zeroable and exclude them from release-day eval (they leak).

## The encoder is a reusable service

The card encoder produces a general card representation that multiple heads consume:
- **draft pointer head** → pick prediction (this repo's main goal)
- **value head** → power/win-rate estimate for an interactive **card-design / editor tool**
  (in-browser, embeds novel typed oracle text on the fly — see
  [design-decisions.md](design-decisions.md) DD-005)
- later: archetype head, deck-fit head, …

Train the representation once; hang task-specific heads off it.

## Model size & hardware

~8–12M parameters total (~30–50 MB). Training footprint < 1 GB. The constraint is **data
volume**, not model size, and it's handled by streaming minibatches from disk — so dataset
size is bounded by disk, not the 8 GB RAM. See [design-decisions.md](design-decisions.md) DD-006.
