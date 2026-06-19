# Design Decisions

Lightweight ADR-style log of the architectural choices, with the reasoning we worked through.
Each entry: the decision, why, and what we rejected.

---

## DD-001 — Content-based card encoder, not per-card ID embeddings
**Decision:** represent every card from its *content* — structured attributes (CMC, colors,
type, P/T, keywords, …) plus a frozen embedding of its oracle text — and never feed a learnable
per-card ID into the candidate-scoring path.

**Why:** the #1 goal is generalizing to cards that didn't exist at training time. A learned ID
embedding is structurally incapable of scoring an unseen card (cold-start) and forces per-set
retraining. Content features + text comprehension are the *entire lever* for unseen-card
performance — the published result is ~55% pick accuracy on a brand-new set vs ~22% chance
(Bertram, Fürnkranz & Müller 2024), whereas representation choice barely moves *in-set* accuracy.

**Rejected:** one-hot / ID-embedding models (Statistical-Drafting, Draftsim NNetBot). Kept only
as the Phase-0 baseline and as a per-set fallback where cross-set transfer isn't needed.

---

## DD-002 — Pointer / masked-softmax pick head for variable candidate sets
**Decision:** score each card *present in the pack* with a shared function
`logit_i = f(pool_context, card_i)`, then masked-softmax over exactly the pack.

**Why:** the candidate set varies in size (1–15) and contents (including unseen cards). Because
the same scoring function is applied per-card and the softmax runs over however many scores we
produce, nothing in the model has a dimension tied to the candidate count. In batches we pad to
a max size and set padded logits to `-inf` so they get zero probability. This is a pointer
network: attention scores over the *input set* are the pick distribution.

**Rejected:** a fixed-vocabulary softmax over all cards in the set — reintroduces cold-start and
locks the head to a fixed card list.

---

## DD-003 — Small specialized model over a full-LLM policy; LLM used only as the text encoder
**Decision:** the picker is a small (~10M param) specialized model. An LLM enters only as the
*frozen oracle-text encoder* (DD-001), not as the policy that makes the pick.

**Why:** generality is a way to buy capability with money/compute instead of data and
engineering. It wins when data is scarce, scope is broad, or volume is low. Draft-picking is the
opposite: narrow task, abundant labeled data (millions of 17lands picks), high call volume,
deployment constraints. A small model trained on that data matches/beats an LLM at this one task
at ~1/1000th the cost. We still borrow the LLM's generality at the *representation* layer, where
data can't substitute for language understanding of novel card text.

**The tool-calling analogy:** fine-tuning an LLM to pick (UrzaGPT-style) is best understood as
tool-calling where cards are the tools — describe options in context, train to select one,
generalize to unseen options. It's a *proven* generalization paradigm, but it's the expensive
implementation of the same pattern our pointer head implements cheaply. The special tokens in
tool-calling are just formatting plumbing — there is no token per tool/card (that would be
cold-start again); the choice is generated as text referencing an in-context description.

**Rejected (for the core picker):** generative LLM-as-policy. Note its cross-set generalization
is asserted but untested, and "unseen" sets may be in the base model's pretraining
(contamination). **Kept open as:** a quick zero-shot baseline / upper-bound probe, an ensemble
member, a source of soft labels (DD-004), and the right choice *if* the product needs a
conversational/explainable drafting assistant.

---

## DD-004 — Knowledge distillation as an organizing principle
**Decision:** use distillation in several places; the whole project is essentially "distill a
general LLM's card understanding + an expensive teacher's judgment into a small, deployable,
generalizing picker."

**Patterns, by value:**
1. **Soft-label (ranking) distillation over the pack** — a stronger/ensemble/win-rate-aware
   teacher gives a full distribution over the pack; student matches it (KL, temperature) on top
   of the human label. Captures the *ranking* the one-hot label throws away. Highest value.
2. **Representation distillation** — the frozen text encoder already is this; can go further and
   compress it into a smaller text encoder for deployment (DD-005).
3. **Teacher-with-leaky-features → release-day student** — train the teacher *with* meta/win-rate
   features (powerful but leak / absent on release day), distill into a student that only sees
   release-day inputs. Smuggles the knowledge in without needing the leaky features at inference.
4. **LLM cold-start pseudo-labeling** — for a *brand-new set with zero human data*, an LLM teacher
   reading oracle text produces provisional pick guidance to bootstrap the student until real
   17lands data accumulates. (Solves new-*format* cold-start, distinct from new-*card*.)
5. **Search distillation** — distill a slow lookahead policy (Phase 4) into the fast network.

**Caveat:** distillation transfers judgment but can't manufacture generalization the student is
structurally incapable of — it complements, never replaces, the content encoder (DD-001).
It also only helps when the teacher's signal is richer than the bare label.

---

## DD-005 — Distilled in-browser text encoder for an interactive card-design tool
**Decision:** for a card editor where a user types *novel* oracle text and wants an interactive
power/value estimate, run a small text encoder client-side; distill the big encoder into a
smaller one only if download/latency demands it.

**Why:** this is the cleanest case for on-the-fly embedding — text is novel (no cache) and
changes every keystroke (the draft case can precompute + cache, so it doesn't need this).

**Important corrections:**
- It must be a small **text encoder**, *not* an MLP on fixed features — an MLP can't read
  free-form typed text, and reading the novel text is the whole point.
- You may not need to distill at all: a MiniLM-class encoder (~22M, ~30 MB quantized) already
  runs in-browser via `transformers.js` / ONNX-web at tens of ms. Distill only to shrink it.
- The value head must see explicit **cost/stat numbers** — a text embedding is semantic, not
  power-aware ("deal 3 for ①" ≈ "deal 3 for ⑥" in text space). Pull numeric magnitudes out as
  features.
- Outputs are **relative/uncertain**: no ground truth exists for never-played cards, so it
  predicts "cards that read like this tend to score here," good for design intuition and
  balance-checking, not validated power.

---

## DD-006 — Train locally for Phases 0–2; cloud only for the multi-set pretrain
**Decision:** develop and train on the M1 / 8 GB machine through Phase 2; rent a single
mid-range cloud GPU for the large multi-set generalization pretrain.

**Why:** the model is tiny (~10M params, < 1 GB training footprint) — size is never the
constraint. "Big data" splits into two things: (a) *fitting in RAM* — not a real constraint,
because minibatch training streams from disk, so dataset size is bounded by disk not RAM; and
(b) *wall-clock per epoch* — the only thing "train slower" trades against. Phases 0–2 on one set
are minutes-to-hours locally. The ~100M-decision multi-set pretrain is feasible locally over a
weekend but painful (fanless throttling, hogged laptop, slow iteration), so cloud is a
convenience trade — turn days into hours for a few dollars — not a hard requirement.

**Practical:** preprocess once to compact integer-index Parquet + cached frozen embeddings; use
a streaming DataLoader (flat RAM regardless of dataset size); checkpoint frequently; `caffeinate`
long runs.

> **OPEN — remote infra for the data piece is the next discussion.** Where to store/stage the
> 17lands CSVs and compact Parquet, how to run preprocessing, and what to rent for the Phase-3
> pretrain. To be filled in.
