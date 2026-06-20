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

🟢 **Phase 0 data pipeline implemented** (download → compact Parquet → streaming loader),
validated on real 17lands data. Baseline model + training loop are next. Research synthesis and
the full design live under `docs/`.

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

## Setup

Uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"          # add ".[dev,embeddings]" for the Phase-1 text encoder
uv run pytest                        # run tests
```

## Quickstart — Phase 0 data pipeline

```bash
# Fast local iteration: small sample download of a real set (a few MB, not GBs)
uv run python -m mtg_draft_ml.data.pipeline --set FDN --sample-rows 50000

# Full set + Scryfall oracle_id join
uv run python -m mtg_draft_ml.data.pipeline --set FDN --scryfall

# Reprocess a local CSV without downloading
uv run python -m mtg_draft_ml.data.pipeline --set FDN --csv data/raw/FDN.PremierDraft.csv.gz
```

Outputs compact integer-index Parquet under `data/processed/draft/` and a card-vocab manifest
under `data/processed/manifests/`. Load it for training:

```python
from mtg_draft_ml.data.dataset import DraftPickDataset, collate_picks
from torch.utils.data import DataLoader

ds = DraftPickDataset("data/processed/draft/FDN.PremierDraft.parquet")
dl = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=collate_picks)
```

See [docs/roadmap.md](docs/roadmap.md) Phase 0 for the next step (baseline model + eval harness).

## Hardware note

Target dev machine is an Apple M1 / 8 GB. Phases 0–2 run locally (streaming data loader keeps
RAM flat regardless of dataset size); the large multi-set pretrain is the one step intended for
a rented cloud GPU. See [docs/design-decisions.md](docs/design-decisions.md) DD-006.
