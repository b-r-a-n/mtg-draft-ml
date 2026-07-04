# WS2.1 — Modern embedder swap

**Why.** The deployed text encoder is a frozen 2020-era MiniLM (384d)
(`src/mtg_draft_ml/cards/text_embed.py`). Our own research flagged this as the unremediated fix:
*"fine-tune the encoder on MTG text or on interaction data (see beeFormer)"*
([research/findings/card-representation.md]). Day-0 generalization (new-set release) is when the
webapp matters most and the representation is weakest — a card's text is all we have. The simplest
question to answer first: does swapping to a state-of-the-art general-purpose encoder help at all?
Two and a half years of embedding progress and 2.7× the dimensionality are free to try.

## Method

**One RunPod RTX A5000 session (~$1.60, ~6 min/run).**

Three arms run through the modernized rotated-LOSO harness (commit 4061441):
- `all-MiniLM-L6-v2` 384d — control, replicates Phase-4 baseline
- `BAAI/bge-large-en-v1.5` 1024d — top MTEB general-purpose encoder
- `intfloat/e5-large-v2` 1024d — substituted for `gte-large-en-v1.5`, which requires
  `trust_remote_code` and was unavailable on the pod

All three arms: 3 seeds × 5 rotating holdouts (BLB/OTJ/WOE/MKM/DSK), 15 runs per arm, 10 epochs,
`sample60000` corpus. The encoder input width was adjusted for the 1024d arms (the capacity sweep
confirmed params don't matter — [results/README.md] Phase 4). Embeddings are cached per model name
so MiniLM and the modern arms write separate cache files.

**Critical plumbing fix (commit 4061441).** The pre-existing `rotate_seeds.py` passed a hardcoded
embedder string directly into `run_loso`, silently bypassing the `MTG_EMBED_MODEL` environment
default introduced in commit 022a95b. Without this fix all three arms would have secretly run
MiniLM and produced a fake null. The fix was verified with a `--embedder` hash smoke test before
launching the full sweep (described in the commit message). Any future arm comparison run through
this harness stands on confirmed, per-arm plumbing.

Raw artifacts: `docs/results/ws21/rotate_minilm.json`, `rotate_bge_large.json`,
`rotate_e5_large.json`.

## Results

### Rotated top-1 (15 runs each; 3 seeds × 5 holdouts)

| arm | embedder | dim | top-1 mean ± sd | novel-only top-1 | WR-agree (model) | Δ top-1 | Δ WR-agree |
|---|---|---|---|---|---|---|---|
| minilm (control) | all-MiniLM-L6-v2 | 384 | **0.5409 ± 0.0234** | 0.5349 | 0.2528 | — | — |
| bge_large | BAAI/bge-large-en-v1.5 | 1024 | 0.5410 ± 0.0180 | 0.5346 | 0.2561 | **+0.0000** | +0.0033 |
| e5_large | intfloat/e5-large-v2 | 1024 | 0.5420 ± 0.0213 | 0.5359 | 0.2577 | **+0.0011** | +0.0049 |

Human WR-agree reference (this config): **0.2893**. Gate: either Δ rotated top-1 > +0.01 **or**
Δ WR-agree > +0.01 over 3 seeds.

### Per-holdout top-1 (means over 3 seeds)

| holdout | minilm | bge_large | e5_large |
|---|---|---|---|
| BLB | 0.5594 | 0.5585 | 0.5610 |
| OTJ | 0.5315 | 0.5268 | 0.5318 |
| WOE | 0.5118 | 0.5275 | 0.5168 |
| MKM | 0.5262 | 0.5255 | 0.5280 |
| DSK | 0.5759 | 0.5665 | 0.5725 |

The per-holdout ordering matches Phase 4 exactly: DSK easiest (~0.57), WOE hardest (~0.51). This
is the expected pattern from [results/phase4-robustness.md] and confirms the harness and data
pipeline are unchanged.

## Key findings

### 1. Harness validated exactly

The MiniLM control arm reproduces the Phase-4 baseline 0.5405 ± 0.0229
([results/phase4-robustness.md]) at **0.5409 ± 0.0234** through the new configurable-embedder code
path. The per-holdout pattern also matches (DSK easiest 0.576, WOE hardest 0.512). Any future arm
comparison run through this harness stands on a confirmed reproduction.

### 2. The env-bypass catch

The pre-existing `rotate_seeds.py` passed a hardcoded embedder string into `run_loso`, silently
bypassing the `MTG_EMBED_MODEL` environment default from commit 022a95b. Without the commit 4061441
fix, all three arms would have run MiniLM and produced a fake null — the most dangerous kind of
negative result. The fix and the `--embedder` hash smoke test that proved per-arm plumbing are
recorded in the commit message. The lesson is reusable: any configurable-by-env parameter must be
explicitly threaded through all call sites; env-only defaults are invisible to callers that construct
the call themselves.

### 3. Interpretation: generic semantics is not the binding constraint

2.5 years of embedding progress and 2.7× the dimensionality move rotated top-1 by ≤ 0.001. The
gate is Δ > +0.01; neither modern arm clears 0.002. The faint WR-agree ordering (minilm < bge <
e5, +0.003..+0.005) is directionally consistent but well below noise and below the gate. Generic
semantic-embedding quality is not the binding constraint on day-0 generalization.

This refines the research doc's cold-start thesis. The untested levers are:

- **(a) Explicit function tags (WS2.2):** structured features that encode what a card *does*
  (removal, evasion, ramp, payoff/enabler role) — information not recoverable from surface text by
  any embedding model trained on general English.
- **(b) Interaction-grounded tuning (WS2.3, beeFormer-style):** fine-tuning the encoder so
  card-embedding dot products predict co-pick/co-play on nested19. This is the fix our own research
  flagged and has never been run.

The null on generic semantic quality is informative: the bottleneck is not *better understanding of
the words* but *grounding in draft-specific co-occurrence*.

### 4. Consequences for the plan

WS2.3's own gate says "only if 2.1/2.2 show signal." With 2.1 null, WS2.3 now hinges entirely on
WS2.2. More broadly, the plan's stop-condition reads: *WS1.3 null + WS1.4 confirmed-linear +
WS2.1 null means WS2.2 is the last task standing between the plan and its stop-condition* ("walls
confirmed with a clean ruler → close the modeling track for good"). WS2.4 (cold-start number) is
independent and still owed either way. WS3 remains open as the only route to a richer outcome
signal.

## Verdict

**NULL.** Neither modern arm clears the gate on either metric. Rotated top-1 moves ≤ +0.001 (gate
+0.01); WR-agree moves ≤ +0.005 (gate +0.01). Keep the MiniLM-384 encoder. The representation
problem is not in generic semantic quality — it is in the absence of draft-function grounding.

## Repro

**Pod training (3 arms):**

```
# Control (MiniLM)
PYTHONPATH=src uv run python scripts/rotate_seeds.py \
  --data-dir data/hf --out /tmp/ws21/rotate_minilm.json

# bge-large-en-v1.5
MTG_EMBED_MODEL=BAAI/bge-large-en-v1.5 \
PYTHONPATH=src uv run python scripts/rotate_seeds.py \
  --data-dir data/hf --out /tmp/ws21/rotate_bge_large.json \
  --embedder BAAI/bge-large-en-v1.5

# e5-large-v2 (substituted for gte-large-en-v1.5; no trust_remote_code required)
MTG_EMBED_MODEL=intfloat/e5-large-v2 \
PYTHONPATH=src uv run python scripts/rotate_seeds.py \
  --data-dir data/hf --out /tmp/ws21/rotate_e5_large.json \
  --embedder intfloat/e5-large-v2
```

All runs: seeds 0,1,2; holdouts BLB/OTJ/WOE/MKM/DSK; sample60000; 10 epochs; A5000 pod.
The `--embedder` flag is required on the modern arms — see the env-bypass catch above.
