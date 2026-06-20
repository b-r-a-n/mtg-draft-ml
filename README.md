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

## Train the Phase-0 baseline

One-hot MLP over the collection, scoring the pack with a masked softmax (Statistical-Drafting /
Draftsim style). Train/val split is by draft (no same-draft leakage); runs on MPS automatically.

```bash
uv run python -m mtg_draft_ml.training.train \
    --parquet data/processed/draft/FDN.PremierDraft.parquet \
    --manifest data/processed/manifests/FDN.PremierDraft.json \
    --epochs 8 --batch-size 512
```

Reports per epoch: `val_top1`, `val_mtpd` (mean pick distance), and `mid-pack_top1` (picks 3–9,
the hard synergy region). Checkpoints to `data/checkpoints/{last,best}.pt`. This is the
fixed-vocabulary baseline that **cannot** generalize to unseen cards — Phase 1 replaces it with
the content card encoder.

## Sync with Hugging Face (the data hub — DD-007)

HF Datasets is the hub between local/CPU preprocessing and the marketplace GPU. The dataset
format is unchanged — `DraftPickDataset` always reads a local path; HF just moves the files.

```bash
uv pip install -e ".[hub]"     # huggingface-cli login (or HF_TOKEN) needed only to push

# Producer: preprocess a set and push its shard + manifest
uv run python -m mtg_draft_ml.data.pipeline --set FDN --scryfall \
    --push --hf-repo <user>/mtg-draft

# Consumer (e.g. on a GPU pod): pull a processed shard, skip all preprocessing
uv run python -m mtg_draft_ml.data.pipeline --set FDN --pull --hf-repo <user>/mtg-draft

# Whole-corpus push/pull (all shards):
uv run python -m mtg_draft_ml.data.hf push --repo <user>/mtg-draft
uv run python -m mtg_draft_ml.data.hf pull --repo <user>/mtg-draft --revision <sha-or-tag>
```

Pin `--hf-revision` / `--revision` for reproducible train/val and leave-one-set-out splits.

## Train the Phase-1 content model

Content card encoder (Scryfall structured features + frozen oracle-text embedding) → masked
mean-pool over the pool → pointer head over the pack. ID-free, so it can score **unseen** cards.

```bash
# build the per-set content matrix (hashing embedder = no heavy deps; for real use pass a model)
uv run python -m mtg_draft_ml.cards.content_table \
    --manifest data/processed/manifests/FDN.PremierDraft.json \
    --scryfall data/scryfall/oracle-cards.json \
    --out data/processed/cards/FDN.content.npy --embedder hash

# train the content model
uv run python -m mtg_draft_ml.training.train_content \
    --parquet data/processed/draft/FDN.PremierDraft.parquet \
    --manifest data/processed/manifests/FDN.PremierDraft.json \
    --content data/processed/cards/FDN.content.npy --epochs 8
```

Use `--embedder all-MiniLM-L6-v2` (needs `".[embeddings]"`) for the real frozen text encoder —
the hashing embedder is for fast iteration only and is **not** semantically meaningful, so don't
judge generalization with it. On a single in-set sample the content model only matches the one-hot
baseline; its advantage is cross-set generalization (the leave-one-set-out test — next).

See [docs/roadmap.md](docs/roadmap.md) for the remaining Phase 1 work (new-set generalization
benchmark) and [docs/data-infra.md](docs/data-infra.md) for the full storage plan.

## Hardware note

Target dev machine is an Apple M1 / 8 GB. Phases 0–2 run locally (streaming data loader keeps
RAM flat regardless of dataset size); the large multi-set pretrain is the one step intended for
a rented cloud GPU. See [docs/design-decisions.md](docs/design-decisions.md) DD-006.
