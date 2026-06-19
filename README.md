# mtg-draft-ml

Machine-learning models for **Magic: The Gathering draft pick prediction**, with a
primary design goal of **generalizing to cards the model has never seen** (e.g. a brand-new
set) and a secondary goal of making *good* picks (informed by deck win-rate signal), not
merely human-like ones.

The core idea: represent every card from its **content** (structured attributes + an
oracle-text embedding) rather than a per-card ID, so the model can score cards that did not
exist at training time. A small, specialized model then encodes the drafted pool and scores
each card in the current pack.

## Status

🟡 **Research complete, implementation not started.** This repo currently holds the research
synthesis and a scaffold for the upcoming build.

## Where to start reading

| Doc | What it is |
|---|---|
| [docs/research/synthesis.md](docs/research/synthesis.md) | The headline recommendation: stack, data, eval, risks, roadmap |
| [docs/architecture.md](docs/architecture.md) | The model design we're building, component by component |
| [docs/roadmap.md](docs/roadmap.md) | Phased implementation plan (local-first, cloud for the big pretrain) |
| [docs/design-decisions.md](docs/design-decisions.md) | Why this approach over the alternatives (LLM-policy, ID embeddings, …) |
| [docs/research/findings/](docs/research/findings/) | Per-dimension research detail |
| [docs/research/sources.md](docs/research/sources.md) | Every cited source + adversarial fact-check verdict |

## Repo layout

```
docs/                 research synthesis + design docs
src/mtg_draft_ml/
  data/               17lands download, CSV -> compact parquet, streaming dataset
  cards/              Scryfall feature extraction + frozen text embeddings
  models/             card encoder, pool encoder, pick head, full model
  training/           losses (masked CE / InfoNCE / distillation), train loop
  eval/               top-1, MTPD, per-pick, new-set generalization metrics
configs/              experiment configs (phase0.example.yaml)
data/                 datasets (gitignored)
notebooks/            exploration
tests/
```

## Quickstart (placeholder)

Not runnable yet — see [docs/roadmap.md](docs/roadmap.md) Phase 0 for the first build step
(17lands → Scryfall data pipeline + baseline model).

## Hardware note

Target dev machine is an Apple M1 / 8 GB. Phases 0–2 run locally (streaming data loader keeps
RAM flat regardless of dataset size); the large multi-set pretrain is the one step intended for
a rented cloud GPU. See [docs/design-decisions.md](docs/design-decisions.md) DD-006.
