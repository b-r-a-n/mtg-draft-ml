# Data Infrastructure

Concrete plan for storing, preprocessing, and serving the dataset.

**Constraints chosen** (2026-06-19): full historical 17lands corpus; GPU pretrain on a cheap
marketplace (RunPod / Vast.ai / Lambda); dataset may be **public**.

## Decisions

| Concern | Decision |
|---|---|
| Storage hub | **Hugging Face Datasets** (free, public, versioned, streamable) — single canonical store for both local dev and rented GPUs. No egress bills. |
| Preprocessing | **Set-by-set, idempotent.** Initial full-corpus crunch on a cheap throwaway CPU box; incremental new sets locally. |
| Embeddings | Precompute frozen oracle-text embeddings for **all** unique cards once (~27k), store the table in the hub keyed by `oracle_id`. GPU training never loads the text model. |
| GPU access | At pod start, `snapshot_download` Parquet + embeddings to **local NVMe**, train off local disk. |
| Durability | Pods are **ephemeral / reclaimable** → checkpoint frequently to a durable store (HF model repo), make training resumable. |

## What lives in the hub

```
hf://datasets/<user>/mtg-draft/
  manifest.json                 # set -> shards, card vocab, oracle_id -> row idx
  cards/
    features.parquet            # structured features per oracle_id
    text_embeddings.parquet     # frozen sentence-transformer vectors per oracle_id
  draft/
    <SET>.<EVENT>.parquet       # compact integer-index picks, sharded per set
```

Compact Parquet for the whole corpus is only ~5–20 GB (raw CSVs are ~50–150 GB but are deleted
after compaction).

## Preprocessing flow (idempotent, set-by-set)

```
for set in all_sets:
    if set already in hub: skip
    download <SET>.<EVENT>.csv.gz       # ~1–4 GB
    stream -> compact integer-index Parquet (drop 259-wide one-hot)
    join names -> oracle_id via Scryfall
    delete raw                          # peak disk = one set
    upload shard to hub
```

Peak disk stays at one set, so it runs anywhere — locally (slow, network-bound) or on a cheap
CPU VM (fast, parallel across cores; a few dollars for the full corpus).

## GPU pretrain flow (Phase 3)

1. Rent a single mid-range GPU (4090 / A10 / L4) on RunPod / Vast / Lambda.
2. `snapshot_download` Parquet + embedding table to local NVMe (~minutes).
3. Train off local disk; **checkpoint every N steps to an HF model repo** (resumable).
4. Tear down.

## Notes / gotchas

- **CPU data-loading is the bottleneck, not GPU FLOPs** — the model is ~10M params. Pick an
  instance with enough vCPUs for DataLoader workers; don't overpay for GPU.
- HF dataset versioning makes the leave-one-set-out / temporal splits reproducible.
- Keep the embedding table separate from the picks Parquet so heads/encoders can be re-trained
  without re-touching the large picks data.
- Reused for the card-design tool (DD-005): the same features + embeddings artifacts feed the
  value head.
