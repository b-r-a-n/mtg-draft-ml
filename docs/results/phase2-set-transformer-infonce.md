# Phase 2 — Set Transformer + InfoNCE ablation

**Setup:** same LOSO benchmark as Phase 1 — train on BLB+OTJ+WOE+MKM (~260k picks, 1,294 cards),
hold out **DSK** (98% novel), MiniLM text encoder, 10 epochs, single seed. We vary the pool encoder
(`mean` vs `set_transformer`) and the loss (`ce` = in-pack masked softmax vs `infonce` = in-pack +
512 sampled global negatives).

Reproduce: `python -m mtg_draft_ml.eval.generalization --train … --holdout … --pool … --loss …`

## Results (held-out DSK top-1)

| pool | loss | in-set top-1 | **held-out top-1** | novel-only | MTPD |
|---|---|---|---|---|---|
| mean | ce  *(Phase 1 baseline)* | 0.588 | 0.552 | 0.534 | 1.005 |
| **set_transformer** | **ce** | 0.614 | **0.563** | 0.546 | **0.919** |
| mean | infonce (512 neg) | 0.436 | 0.349 | 0.324 | 2.341 |
| set_transformer | infonce (512 neg) | 0.441 | 0.350 | 0.324 | 2.177 |

(random floor 0.233; published ~0.55 bar.)

## Findings

1. **The Set Transformer is a real, consistent win.** Held-out 0.552 → **0.563**, in-set 0.588 →
   0.614, MTPD 1.005 → 0.919. Modeling synergy/anti-synergy across the pool (SAB self-attention +
   PMA pooling) beats order-blind mean pooling — modest but consistent across in-set, held-out, and
   ranking quality. **Best config: `set_transformer` + `ce`.**

2. **Global-negative InfoNCE *hurt* badly** (0.55 → 0.35), regardless of pool encoder. This is an
   important negative result. The proven "contextual InfoNCE" for drafting uses **the pack as the
   negative set** — which is *exactly* what our `ce` (masked softmax over the pack) already is
   (synthesis: "masked-softmax CE and in-pack InfoNCE correspond exactly to one card preferred over
   the rest of the pack"). Adding 512 random global negatives to the same softmax dilutes the signal:
   ~97% of the partition function becomes "picked card vs. a random card" — an easy, coarse
   card-quality discrimination — drowning out the hard, relevant within-pack ranking. In-set top-1
   collapsing to 0.44 confirms the objective shifted away from the actual task.

   **Lesson:** the pack *is* the correct negative set; do not dilute it with global negatives. Keep
   `loss=ce` (≡ in-pack InfoNCE). The `infonce` global-negative mode is retained only as a
   documented dead-end / ablation knob.

## Recommendation

Adopt **`set_transformer` + `ce`** as the Phase-2 model (held-out 0.563, the new best). Leave the
global-negative InfoNCE off.

## Caveats

Single seed; sampled data; `n_negatives=512` not swept (fewer negatives would dilute less but is not
expected to beat in-pack-only); one holdout set. Directional, not final.

## Next

- Feature standardization + multiple seeds + rotate the holdout (tighten ±, fair ablation).
- **Phase 3 signal:** WR-agreement metric + win-rate-weighted training (good-not-just-human, DD-004).
