# WS3.1 — Cast-conditioned adjudicator swap (Stage 1)

**Decision: RULER-ROBUST** (2026-07-07). The model beats human drafters in **70–82% of 800 DSK
drafts** under every cast-β ruler, including the confound-mitigated early-cast variant. The
deck_value adjudicator had flattered the model (92–98% → 70–82%); the cast-adjudicated margin is
the most credible the project has. Stage 2 (cast-β as teacher) declined — WS1.3 subsumption logic
+ budget. Ran entirely locally, $0.

---

## Why

WS3.0 established that per-card cast-conditioned value (β_cast) is reliable (split-half ρ 0.976),
genuinely distinct from GIH (Sp 0.637), and face-plausible — but with a documented
cast-selection confound: cheap combat tricks (Turn Inside Out) rank high because they're cast from
winning positions; desperation-diggers (Glimmerburst, Grab the Prize) rank low because they're cast
when behind. Before using cast-β as an adjudicator or teacher, the confound had to be addressed in
design.

WS3.1 Stage 1 therefore has two parts:

1. **Confound-mitigation study** — two new rulers designed to dampen the selection bias, verified
   on probe cards before running the outcome eval.
2. **Adjudicator swap** — the WS1.2 outcome eval (same 800 DSK drafts, current deployed net,
   `--l2 30`) re-run with three cast-β rulers in place of deck_value, producing the
   cross-ruler-comparable verdict.

---

## Confound-mitigation study

Two variants of β_cast were fitted on the same 1,011,949-game DSK replay corpus (no new downloads):

- **early-cast** (`--max-turn 6`): cast counts restricted to turns 1–6 only; late-game trick casts
  that inflate winning-position metrics are excluded.
- **normalized-cast** (`--normalize-casts`): each game's cast-count vector divided by that game's
  total cast count; removes the overall-game-activity confound.

Reliability and independence are preserved for both:

| variant | split-half ρ | Sp vs WS3.0 full cast | Sp vs GIH |
|---|---|---|---|
| WS3.0 full cast (reference) | 0.976 | — | 0.637 |
| early-cast (turn ≤ 6) | 0.977 | 0.871 | 0.570 |
| normalized-cast | 0.976 | 0.966 | 0.621 |

The probe-card table shows which variant actually de-confounds:

| probe card | WS3.0 rank | early-cast rank | normalized rank | interpretation |
|---|---|---|---|---|
| Turn Inside Out | #7 / 180 | #13 / 131 | **#1** / 180 | combat trick; should be mid-pack |
| Glimmerburst | #173 / 180 | #56 / 131 | #149 / 180 | desperation digger; should rank low |
| Grab the Prize | bottom-3 | bottom-2 | bottom-4 | desperation digger; correctly demoted |

**early-cast verdict:** Turn Inside Out drops from #7 to #13 (meaningful reshuffle; Sp vs WS3.0 =
0.871) and Glimmerburst rises from #173 to #56 (the desperation-digger penalty is substantially
removed). Grab the Prize stays near the bottom. Mitigation works partially — the confound is
reduced but not eliminated.

**normalized-cast verdict:** Turn Inside Out jumps to **#1 of 180**; diggers stay at the bottom
(Glimmerburst #149, Grab the Prize #177). Normalization *amplifies* the cast-selection confound —
dividing by total cast count up-weights games where few cards were cast (fast, decisive wins), which
are exactly the games where trick casts dominate. **Rejected as a mitigation; kept as an orthogonal
ruler arm** to test whether the outcome eval results are consistent across perspectives.

Both variants were carried forward to the adjudicator swap as additional arms.

Raw diagnostics: `docs/results/ws31/variant-diagnostics.{early,norm}.json`
(new flags in `scripts/replay_value_gonogo.py`: `--max-turn N`, `--normalize-casts`).

---

## Adjudicator-swap results

### Methods caveat (read before the tables)

The sigma-scale margins (`delta_vs_human`) are **NOT comparable across rulers**. The reason:
β_cast magnitudes saturate the logistic sigmoid — human-mean scores run 0.92–1.00 under cast rulers
vs 0.63 under deck_value, so the scale of "one sigma" differs dramatically. However, the
**per-draft beats-human fraction is invariant under any monotone score transform** — it is the exact
cross-ruler comparable. All headline conclusions below are based on that fraction; the delta tables
are included for completeness with this caveat attached.

### Summary table

| adjudicator | builder | human mean | model mean | model beats human | Δ vs human |
|---|---|---|---|---|---|
| deck_value (WS1.2 ref) | playprob | 0.6325 | 0.6892 | **92%** | +0.0567 |
| deck_value (WS1.2 ref) | naive | 0.6728 | 0.7446 | **98%** | +0.0717 |
| cast (full) | playprob | 0.9203 | 0.9514 | **74%** | +0.0311 |
| cast (full) | naive | 0.9763 | 0.9887 | **81%** | +0.0124 |
| cast-early (turn ≤ 6) | playprob | 0.9956 | 0.9966 | **70%** | +0.0010 |
| cast-early (turn ≤ 6) | naive | 0.9972 | 0.9976 | **81%** | +0.0004 |
| cast-normalized | playprob | 0.5904 | 0.8458 | **77%** | +0.2554 |
| cast-normalized | naive | 0.9868 | 0.9969 | **82%** | +0.0102 |

*The delta column is shown for reference; cross-ruler delta comparison is invalid (saturation caveat
above). Beats-human % is the cross-ruler fact.*

### Per-ruler delta tables (playprob)

**cast (full):**

| policy | est score | Δ vs human | beats human |
|---|---|---|---|
| deckvalue_greedy | 0.9962 | +0.0759 | 100% |
| gih_greedy | 0.9690 | +0.0487 | 87% |
| **model** | **0.9514** | **+0.0311** | **74%** |
| human | 0.9203 | — | — |
| random | 0.7867 | −0.1336 | 14% |

**cast-early (turn ≤ 6):**

| policy | est score | Δ vs human | beats human |
|---|---|---|---|
| deckvalue_greedy | 0.9982 | +0.0026 | 100% |
| gih_greedy | 0.9972 | +0.0016 | 88% |
| **model** | **0.9966** | **+0.0010** | **70%** |
| human | 0.9956 | — | — |
| random | 0.9900 | −0.0056 | 14% |

**cast-normalized:**

| policy | est score | Δ vs human | beats human |
|---|---|---|---|
| deckvalue_greedy | 0.9997 | +0.4093 | 100% |
| gih_greedy | 0.9594 | +0.3691 | 90% |
| **model** | **0.8458** | **+0.2554** | **77%** |
| human | 0.5904 | — | — |
| random | 0.1104 | −0.4800 | 12% |

### Per-ruler delta tables (naive top-23)

**cast (full):**

| policy | est score | Δ vs human | beats human |
|---|---|---|---|
| deckvalue_greedy | 0.9981 | +0.0218 | 100% |
| gih_greedy | 0.9937 | +0.0174 | 96% |
| **model** | **0.9887** | **+0.0124** | **81%** |
| human | 0.9763 | — | — |
| random | 0.9359 | −0.0404 | 18% |

**cast-early (turn ≤ 6):**

| policy | est score | Δ vs human | beats human |
|---|---|---|---|
| deckvalue_greedy | 0.9982 | +0.0010 | 100% |
| gih_greedy | 0.9977 | +0.0005 | 95% |
| **model** | **0.9976** | **+0.0004** | **81%** |
| human | 0.9972 | — | — |
| random | 0.9961 | −0.0011 | 12% |

**cast-normalized:**

| policy | est score | Δ vs human | beats human |
|---|---|---|---|
| deckvalue_greedy | 0.9997 | +0.0129 | 100% |
| gih_greedy | 0.9977 | +0.0109 | 97% |
| **model** | **0.9969** | **+0.0102** | **82%** |
| human | 0.9868 | — | — |
| random | 0.9098 | −0.0770 | 28% |

---

## Three conclusions

### 1. The headline verdict is RULER-ROBUST

The model beats humans in **70–82% of drafts** under every cast variant, including the
confound-mitigated early-cast ruler (70% playprob, 81% naive). "Model drafts outcome-better decks"
is not a deck_value artifact — it holds under a genuinely independent adjudicator with structurally
different confounds.

### 2. deck_value flattered the model

92–98% → 70–82% is genuine adjudicator disagreement, not noise. The mechanism is interpretable:
the deployed model is GIH-composite-trained; deck_value correlates with GIH reasonably well
(Sp 0.72 well-sampled, from WS3.0), while cast-β is more independent (Sp 0.637–0.702). A more
independent judge is a more conservative one. The cast-adjudicated 70–82% is the most credible
margin the project has produced.

### 3. gih_greedy > model under ALL rulers

The oracle ordering (deckvalue_greedy > gih_greedy > model > human > random) holds under every
ruler. The card-power-sum caveat — that greedy rating policies beat the model policy — is
confound-universal. Nothing in Stage 1 changes it. This remains the binding ceiling for the
outcome-eval metric as designed.

---

## Stage 2 (cast-β as teacher) — declined

Stage 2 would repeat WS1.3 with cast-β replacing deck_value as the training teacher: fit cast-β on
the 19-set nested corpus, add it as a target, and re-run the 6-seed GIH-agree + outcome-prong sweep.

**Declined. Reasoning:**

1. **WS1.3 subsumption.** WS1.3 established that deck_value adds nothing as a teacher at 19 sets
   with a reliable β (ρ 0.87): corpus breadth subsumes the value sense. Cast-β is a different
   signal (Sp 0.637 vs GIH) with a different confound — but there is no evidence it would escape
   the same subsumption at 19 diverse sets. The prior is against it.
2. **Budget.** Running Stage 2 requires ~19 set replay downloads (~500 MB gz each, ~9.5 GB total)
   plus re-parsing, a β fit per set, and a ~$2.50 GPU pod run against a $4.60 balance. Against a
   prior that favors no effect, this spend is not justified.
3. **This is a scoping judgment, not a gate result.** Stage 2 was never run; its efficacy is
   unknown. The decline reflects the balance of evidence (WS1.3 analogy) and resource constraints,
   not a measured null.

**What would reopen it:** a future need for a de-confounded teacher on a *small-corpus* set (where
breadth subsumption cannot operate) or a corpus where cast-β and GIH diverge substantially in
practice. The 19-set nested corpus is the wrong test bed for this question.

---

## Repro commands

### Confound-mitigation study (new flags; uncommitted)

```bash
# Early-cast variant (turn ≤ 6 only) — outputs /tmp/ws31_early.json
PYTHONPATH=src .venv/bin/python scripts/replay_value_gonogo.py \
  --set DSK --max-turn 6 \
  --out /tmp/ws31_early.json

# Normalized-cast variant — outputs /tmp/ws31_norm.json
PYTHONPATH=src .venv/bin/python scripts/replay_value_gonogo.py \
  --set DSK --normalize-casts \
  --out /tmp/ws31_norm.json
```

### Outcome eval — three cast-β rulers × two builders (6 runs)

```bash
# Full cast β (WS3.0 NPZ, already built)
PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build playprob \
  --game-npz data/game/replay_cast.DSK.PremierDraft.full.npz \
  --out docs/results/ws31/outcome-eval.playprob.cast.json

PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build naive \
  --game-npz data/game/replay_cast.DSK.PremierDraft.full.npz \
  --out docs/results/ws31/outcome-eval.naive.cast.json

# Early-cast β (maxturn6 NPZ, written by replay_value_gonogo --max-turn 6)
PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build playprob \
  --game-npz data/game/replay_cast.DSK.PremierDraft.full.maxturn6.npz \
  --out docs/results/ws31/outcome-eval.playprob.castearly.json

PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build naive \
  --game-npz data/game/replay_cast.DSK.PremierDraft.full.maxturn6.npz \
  --out docs/results/ws31/outcome-eval.naive.castearly.json

# Normalized-cast β (normalized NPZ, written by replay_value_gonogo --normalize-casts)
PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build playprob \
  --game-npz data/game/replay_cast.DSK.PremierDraft.full.normalized.npz \
  --out docs/results/ws31/outcome-eval.playprob.castnorm.json

PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build naive \
  --game-npz data/game/replay_cast.DSK.PremierDraft.full.normalized.npz \
  --out docs/results/ws31/outcome-eval.naive.castnorm.json
```

Raw artifacts: `docs/results/ws31/outcome-eval.{playprob,naive}.{cast,castearly,castnorm}.json`,
`docs/results/ws31/variant-diagnostics.{early,norm}.json`
