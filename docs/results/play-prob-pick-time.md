# Pick-time buildability (P(played | current picks)) — IN PROGRESS (resume doc)

**Status: experiment implemented, running/to-run. This doc is a self-contained handoff so a fresh
session can resume.** Last updated 2026-06-25.

## The question

Does weighting *picks* by **P(card eventually played | my current picks)** improve drafting? This is
the pick-time version of the buildability signal — down-weight a pick that won't make your deck given
your developing pool.

## Why this experiment exists (the story so far)

1. `P(played | pool)` is a strong, learned buildability signal from `deck_<card>` vs `sideboard_<card>`
   ([play-prob.md](play-prob.md)): adding pool context lifts AUC 0.81→0.93, recovers castability.
2. **At BUILD time it works** — using it as the deck-builder in the outcome eval (`--build playprob`)
   *sharpens* the model's edge over humans (+0.049, 84%). The pool is full → in-distribution.
3. **Folding the FULL-pool model into PICKS failed** ([play-prob.md](play-prob.md), `test_play_policy.py`):
   flat at λ=0.5 (+0.0015), worse at higher λ. Two suspected reasons: **(a) OOD** — full-pool model
   applied to *partial* mid-draft pools; **(b)** the model already drafts coherently (75% on-color ≈
   humans), so little pick-time headroom.
4. **This experiment tests the fix for (a):** train `P(played | PARTIAL pool)` so it's in-distribution
   at pick time, then redo the pick-policy test. If it now helps → buildability-at-pick adds value once
   in-distribution. If still flat → reason (b) dominates (the drafter is already coherent enough).

## What's implemented (committed)

- `src/mtg_draft_ml/eval/play_prob.py::train_play_model_partial` — **subsample approximation**: for each
  built deck, draw `samples` random partial pools (k-subsets of the pool); CANDIDATES = pool cards NOT
  in the subset (cards you might still take), labelled by whether they made the deck. Trains
  `P(eventually played | partial pool, candidate not yet in pool)`. Returns a `PlayModel` (same
  `.probs(pool, pack)` interface), so everything downstream is unchanged.
- `scripts/test_play_policy.py --partial` — uses the partial model; effective pick logit =
  `model_logit(c) + λ·logit P(c eventually played | current picks)`; scores drafted pools by the
  outcome eval (constrained 2-color+curve deck-WR), λ sweep, with a color-concentration diagnostic.

## How to run (resume here)

```bash
# subsample partial-pool model, pick-time weighting, λ sweep (CPU; ~slow: GBM per pick)
uv run python scripts/test_play_policy.py --set DSK --n-drafts 400 --partial \
    --lambdas 0,0.5,1,2 --out docs/results/play-policy-partial.json
```

Compare to the full-pool run (the failed one): same command **without** `--partial` (results in
[play-prob.md](play-prob.md): flat/worse). NOTE: the full-pool pick run took ~30 min at 600 drafts;
use `--n-drafts 300-400` for a faster first read.

## How to interpret

- **`model+build@λ` deck-WR vs `model` (λ=0):** a meaningful positive Δ (clear of ~±0.005 noise) at some
  λ ⇒ in-distribution buildability-at-pick **helps** — reason (a) was the blocker, worth the faithful
  version next. Flat/negative ⇒ reason (b) dominates: the drafter is already coherent, so pick-time
  buildability adds nothing on top, and the signal stays a build-time tool. (Either answer is publishable.)
- Watch `top-2-color%`: it should now *rise* with λ (more coherent pools), unlike the full-pool run
  where it fell (noise). If it still falls, the partial model didn't fix the OOD.

## If it works → the faithful version (next step after this)

The subsample fakes "current picks" with a random subset. The faithful version joins the **draft
pick-sequence** (`data/hf/draft/<SET>...parquet` — picks in order) with the **final deck**
(`game_data` deck vs sideboard) by `draft_id`, training `P(played | actual picks-so-far)` per real pick
step. More correct, more work. Only worth it if the subsample version shows a signal.

## Key files / pointers

- `eval/play_prob.py` (the model), `scripts/test_play_policy.py` (the policy test), `eval/outcome.py`
  (the scorer + learned-deckbuilder `deck_fn`), `docs/results/play-prob.md` + `outcome-eval.md` (context).
- The committed full-pool failure is the baseline to beat. The deployed webapp DSK model
  (`webapp/model/DSK.onnx`, trained on the GIH-composite, 20 sets) is the draft policy being reweighted.
