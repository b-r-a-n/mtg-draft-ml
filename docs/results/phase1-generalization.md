# Phase 1 — New-set generalization benchmark (first run)

**Experiment:** train the content model on one set, evaluate zero-shot on a *different* set by
swapping in its card table (`ContentDraftModel.set_content`). The test set's cards are ~98% novel
(absent from training), so cross-set top-1 ≈ unseen-card accuracy.

- Train: **BLB** (Bloomburrow) PremierDraft, 80k-pick sample (~276 cards)
- Test: **DSK** (Duskmourn) PremierDraft, 25k-pick sample (~286 cards, 98% novel vs BLB)
- 12 epochs, batch 512, MPS, single seed. Card encoder: 3×512 MLP → 256-d. Pool: mean-pool.
  Head: pointer/masked-softmax. Loss: masked CE.

Reproduce: `python -m mtg_draft_ml.eval.generalization --train-* … --test-* … --embedder …`

## Results

| Representation | input dim | in-set top-1 (BLB val) | cross-set top-1 (DSK) | novel-only top-1 | random floor |
|---|---|---|---|---|---|
| Structured features only | 73 | 0.616 | 0.461 | 0.441 | 0.233 |
| Features + hashing text (256) | 329 | 0.620 | **0.478** | **0.459** | 0.233 |
| Features + MiniLM text (384) | 457 | 0.619 | 0.437 | 0.416 | 0.233 |

(Published bars, Bertram et al. 2024: ~0.22 chance; ~0.43 trained on one set; ~0.55 pretrained on
a ~100M-decision corpus.)

## Findings

1. **The content encoder genuinely generalizes to unseen cards.** All variants score ~0.44–0.48
   cross-set on a ~98%-novel set — roughly **2× the random floor (0.233)** and in line with the
   published ~0.43 one-set bar. The Phase-0 one-hot baseline is *inapplicable* here (it has no
   parameters for DSK's cards), which is exactly the architectural win DD-001 predicted.
2. **Representation barely affects in-set accuracy** (0.616–0.620 across all three) — matches the
   research. The differences show up only cross-set.
3. **Structured features carry the generalization here; the semantic text encoder did not help
   (and slightly hurt).** Features-only already generalizes (0.461); hashing text adds a little
   (0.478); MiniLM text *lowers* it (0.437). Likely cause: with a single 80k-pick set, the 384-d
   semantic block lets the encoder overfit BLB-specific text/mechanics that don't transfer. This
   is consistent with the research caveat that text/semantic gains toward ~0.55 require **large
   multi-set pretraining**, not one small set — and with "budget ablation time to find the driver
   for YOUR data."

## Caveats (do not over-read)

Single seed; sampled data (80k/25k picks, not full sets); uncontrolled text dims (256 vs 384);
structured features are **not standardized** (raw CMC/pips vs unit-norm text — may bias the encoder
toward features); one set pair. These are enough to demonstrate generalization but not to rank
encoders conclusively.

## Multi-set leave-one-set-out (the real test)

**Experiment:** train on the union of **BLB + OTJ + WOE + MKM** (4 sets, 1,294 unique cards,
~260k picks), hold out **DSK** entirely, evaluate zero-shot. Same model/hyperparameters; 10 epochs.

Reproduce: `python -m mtg_draft_ml.eval.generalization --train … --train … --holdout … --embedder …`

| Encoder | single-set (BLB→DSK) held-out top-1 | **multi-set LOSO (4→DSK) held-out top-1** | LOSO novel-only |
|---|---|---|---|
| Structured features only | 0.461 | 0.523 | 0.508 |
| + hashing text | 0.478 | 0.531 | 0.512 |
| + MiniLM text | 0.437 | **0.552** | **0.534** |

(random floor 0.233; published ~0.55 pretrained bar.)

### The headline findings

1. **Multi-set training lifts generalization across the board** — held-out top-1 rose for every
   encoder (features 0.461→0.523; MiniLM 0.437→0.552). Diversity in training sets ⇒ better transfer.
2. **The encoder ranking FLIPPED, exactly as the research predicted.** Single-set: MiniLM was
   *worst* (0.437 < features 0.461) — it overfit one set's flavor text. Multi-set: MiniLM is *best*
   (0.552 > hashing 0.531 > features 0.523). **The semantic text encoder's value only materializes
   with set diversity.** This is the single most important result of Phase 1.
3. **Multi-set MiniLM (0.552) matches the published ~0.55 bar** — reached here with ~260k picks
   rather than the ~100M-decision corpus cited (DSK likely shares structure/mechanics with the
   training sets), but the qualitative story is exactly right.
4. In-set val is slightly *lower* for the multi-set model (~0.588 vs single-set BLB 0.62) — one
   model fitting 4 heterogeneous sets — yet it generalizes far better. The right trade.

## Caveats (still apply)

Single seed; sampled data; structured features unstandardized; text dims uncontrolled (256 vs 384);
one holdout set. Conclusions are directional, not final rankings.

## Next steps

- **Multiple seeds + more / full sets + rotate the holdout** to tighten the ±.
- **Standardize structured features** and match text dims for a fully fair ablation.
- Add the **WR-agreement** metric (good-not-just-human) and win-rate weighting (DD-004).
- **Phase 2:** Set Transformer pool encoder + contextual InfoNCE (the next accuracy lever).
