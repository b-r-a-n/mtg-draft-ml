# Pick-time buildability (P(played | current picks)) — RESOLVED: flat (build-time tool)

**Status: DONE.** Training the buildability model on *partial* pools fixes the out-of-distribution
problem at pick time (pools get measurably more color-coherent), but deck-WR stays flat — the drafter
is already coherent enough. Buildability remains a **build-time** lever, not a pick-time one. The
faithful pick-sequence version is **not worth building** (gate not met). Last updated 2026-06-25.

## The question

Does weighting *picks* by **P(card eventually played | my current picks)** improve drafting? This is
the pick-time version of the buildability signal — down-weight a pick that won't make your deck given
your developing pool.

## Verdict

Replaying 400 held-out DSK drafts; each policy's pool scored by the outcome eval (constrained
2-color+curve deck-WR). Effective pick logit = `model_logit(c) + λ·logit P(c eventually played | current picks)`,
with the **partial-pool** (in-distribution) buildability model. Noise band ≈ ±0.005.

| policy | deck-WR | Δ vs model | Δ vs human | top2-color% |
|---|---|---|---|---|
| human | 0.5648 | — | — | 78% |
| model (λ=0) | 0.5968 | +0.0000 | +0.0320 | 75% |
| **model+build@0.5** | **0.5995** | **+0.0027** | +0.0347 | **78%** |
| model+build@1.0 | 0.5966 | −0.0002 | +0.0318 | 74% |
| model+build@2.0 | 0.5827 | −0.0141 | +0.0179 | 66% |
| gih_greedy | 0.5996 | +0.0028 | +0.0347 | 58% |

Three reads, in order of importance:

1. **The OOD fix (reason a) is real.** At the meaningful operating point λ=0.5, pool color-coherence
   *rises* 75% → 78% (matching humans), exactly as predicted — versus the failed full-pool run where
   it *fell*. Training on partial pools genuinely made the signal in-distribution at pick time. It also
   removes the catastrophic over-weighting: at λ=2, deck-WR Δ is −0.014 here vs **−0.042** full-pool
   (beats-human 65% vs 38%).

2. **But it adds no upside (reason b dominates).** Despite the extra coherence, deck-WR peaks at
   **+0.0027 (λ=0.5), inside the ±0.005 noise band.** The drafter at λ=0 is already 75%-coherent; the
   marginal 3 pts of color concentration buy no win-rate. Push harder (λ≥1) and both WR and coherence
   fall.

3. **Color-coherence is not the WR bottleneck.** The tell: `gih_greedy` wins just as much (0.5996)
   with only **58%** color-coherence — the constrained deckbuilder extracts a good 2-color deck from a
   rainbow pool as long as the cards are individually strong. So pushing pools *toward* coherence
   (what buildability does) is optimizing a dimension that isn't binding.

**Conclusion:** buildability is a **build-time tool** (where it sharpens the model's edge: +0.049,
84% in the outcome eval, [play-prob.md](play-prob.md)). At pick time it is flat once in-distribution —
the policy is already coherent enough. The faithful pick-sequence model below is **not worth building**.

## Why this experiment existed (the story)

1. `P(played | pool)` is a strong, learned buildability signal from `deck_<card>` vs `sideboard_<card>`
   ([play-prob.md](play-prob.md)): adding pool context lifts AUC 0.81→0.93, recovers castability.
2. **At BUILD time it works** — using it as the deck-builder in the outcome eval (`--build playprob`)
   *sharpens* the model's edge over humans (+0.049, 84%). The pool is full → in-distribution.
3. **Folding the FULL-pool model into PICKS failed** ([play-prob.md](play-prob.md), `test_play_policy.py`):
   flat at λ=0.5 (+0.0015), worse at higher λ. Two suspected reasons: **(a) OOD** — full-pool model
   applied to *partial* mid-draft pools; **(b)** the model already drafts coherently (~75% on-color),
   so little pick-time headroom.
4. **This experiment tested the fix for (a):** train `P(played | PARTIAL pool)` so it's in-distribution
   at pick time, then redo the pick-policy test. Result above: (a) was real but the cure is flat → **(b)
   dominates the upside.**

## What's implemented (committed)

- `src/mtg_draft_ml/eval/play_prob.py::train_play_model_partial` — **subsample approximation**: for each
  built deck, draw `samples` random partial pools (k-subsets of the pool); CANDIDATES = pool cards NOT
  in the subset, labelled by whether they made the deck. Trains `P(eventually played | partial pool,
  candidate not yet in pool)`. Returns a `PlayModel` (same `.probs(pool, pack)` interface).
- `scripts/test_play_policy.py --partial` — uses the partial model; λ sweep; scores drafted pools by the
  outcome eval; reports `top2-color%` and now **persists it to the JSON** under `top2_color`.

## Reproduce

```bash
# subsample partial-pool model, pick-time weighting, λ sweep (CPU; ~slow: GBM per pick, ~15-20 min)
uv run python scripts/test_play_policy.py --set DSK --n-drafts 400 --partial \
    --lambdas 0,0.5,1,2 --out docs/results/play-policy-partial.json
```

Results: [`play-policy-partial.json`](play-policy-partial.json) (this run) vs
[`play-policy.json`](play-policy.json) (the full-pool baseline; same command **without** `--partial`).
The full-pool run took ~30 min at 600 drafts; use `--n-drafts 300-400` for a faster read.

## The faithful version (NOT pursued — gate not met)

The subsample fakes "current picks" with a random subset. The faithful version would join the **draft
pick-sequence** (`data/hf/draft/<SET>...parquet`, picks in order) with the **final deck** (`game_data`
deck vs sideboard) by `draft_id`, training `P(played | actual picks-so-far)` per real pick step. More
correct, more work. The doc's gate was "only worth it if the subsample version shows a signal." **It
doesn't** (peak +0.0027, within noise), so this is shelved. Revisit only if a different framing of
pick-time buildability emerges.

## Key files / pointers

- `eval/play_prob.py` (the model), `scripts/test_play_policy.py` (the policy test), `eval/outcome.py`
  (the scorer + learned-deckbuilder `deck_fn`), `docs/results/play-prob.md` + `outcome-eval.md` (context).
- The deployed webapp DSK model (`webapp/model/DSK.onnx`, GIH-composite, 20 sets) is the draft policy
  being reweighted; `gih_greedy` is the strong rating-only reference.
