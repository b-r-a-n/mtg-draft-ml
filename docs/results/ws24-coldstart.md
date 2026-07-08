# WS2.4 — Cold-start LLM teacher: first day-0 number

**Decision: FIRST DAY-0 NUMBER OBTAINED** (2026-07-07). A content-only LLM teacher (Claude Sonnet,
286 DSK cards scored 0–10 by a 10-agent workflow) closes **19/21/19% of the oracle−baseline
WR-agreement gap** across 3 seeds (mean ~20%) at blend alpha=4. The heuristic floor closes −4% at
best (alpha=0.5), confirming the LLM signal is doing real work. Ran entirely locally, $0 pod spend
(~240k agent tokens for teacher scoring). This is an **upper bound** due to training-data leak; see
Caveats. Raw artifacts: [ws24/](ws24/).

---

## Why

On a new set's release day, 17lands has zero draft data. The content model generalizes to new
cards via its text encoder, but cannot know which cards are *strong in this specific format* — only
human drafts accumulate that signal. The cold-start scaffold (DD-004 #4) asks: can an LLM reading
oracle text provide enough pick guidance to be worth blending in before human data exists?

This is the one niche with an LLM advantage: the existing model already generalizes to unseen
*cards* (~0.57 top-1), and the ~0.58 human-noise ceiling means "predict humans better" is a dead
end. Day-0 format strength is the residual slice where an LLM has information nothing else has.

The WS2.4 task was explicitly optional and survives the 2026-07-05 stop condition. The goal was
"report even a null result." This run exceeded that: 20% gap closure on the first run.

---

## Method

### Training and holdout

Small content model (`emb_dim=256`, `enc_hidden=512`, `enc_layers=3`) trained for 10 epochs on 4
sets (BLB, OTJ, WOE, MKM) — the Phase-4-era config, chosen for comparability with the scaffold's
design and past results. DSK is treated as "the new set on release day": zero DSK picks in
training. Evaluation is on the DSK holdout (286 cards, 97.6% novel).

Three pick policies are compared on the **same trained model per seed**:

| policy | blend signal | meaning |
|---|---|---|
| baseline | none (α=0) | pure content-encoder zero-shot |
| **agent-claude** | LLM `llm_quality` scores | what we could ship on day one |
| oracle | real 17lands GIH-WR | upper bound — data that doesn't exist on day 0 |

Both quality vectors are z-scored before blending, so `alpha` means the same thing across sources
(scale-invariant, matching the deploy.Drafter dial). Alpha grid: [0.5, 1, 2, 4].

**Primary metric:** WR-agreement vs real 17lands GIH-WR on the holdout. **Secondary:** avg-pick-WR
and top-1. **Headline:** % of the (oracle − baseline) WR-agreement gap closed.

### Teacher scoring: 10-agent workflow (cached)

The teacher is the `CachedTeacher` variant (`distill/teacher.py::get_teacher('cached:agent-claude')`),
which reads pre-scored ratings from disk with no API key needed at eval time. The scores were
produced in a separate workflow:

- **286/286 DSK cards** scored 0–10 (content-only power) by a 10-agent workflow: 10 parallel Claude
  Sonnet scorers using a day-0 rubric (oracle text + stats, explicitly no win-rate or meta signals),
  with an **adversarial verification pass** (Claude cross-examines a random 25-card sample).
- Verify result: ACCEPT — 1/25 violations flagged; **4 fixes applied**, including a within-cycle
  Verge-land inconsistency (cycle members must score consistently given identical stats and text).
- Scores pre-populated the scaffold's disk cache at
  `data/teacher_cache/DSK.cached_agent-claude.ratings.json` (gitignored; reproducibility copy at
  `docs/results/ws24/agent-claude.DSK.ratings.json`).
- **~240k agent tokens total** (no API spend beyond tokens; no pod needed).

### HeuristicTeacher floor

`HeuristicTeacher` provides a deterministic stats-only baseline (no LLM, no API key) using rarity,
mana cost, and P/T heuristics. It runs on the same harness and the same alpha grid. Spearman vs
real 17lands GIH: 0.081 (n=265) — near-zero correlation.

### Harness changes (uncommitted, part of this deliverable)

- `distill/teacher.py` — `CachedTeacher` class; `get_teacher('cached:<id>')` factory.
- `distill/coldstart.py` — `run_coldstart` accepts multiple teacher ratings files evaluated on one
  trained model (avoids re-training per teacher).
- `scripts/pod_coldstart.py` — `--teacher` comma-list, `--seed`, `--out` flags.

---

## Results

### Baseline values (per seed)

| seed | baseline WR-agree | human WR-agree | in-set top-1 |
|---|---|---|---|
| 0 | 0.2628 | 0.2981 | 0.6138 |
| 1 | 0.2523 | 0.2981 | 0.6192 |
| 2 | 0.2570 | 0.2981 | 0.6195 |
| **mean** | **0.2573** | **0.2981** | **0.6175** |

### Agent-claude teacher: alpha sweep (WR-agreement)

| alpha | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| baseline (0) | 0.2628 | 0.2523 | 0.2570 | 0.2573 |
| 0.5 | 0.2806 | 0.2694 | 0.2778 | 0.2759 |
| 1.0 | 0.2946 | 0.2845 | 0.2910 | 0.2900 |
| 2.0 | 0.3117 | 0.3034 | 0.3088 | 0.3080 |
| **4.0** | **0.3316** | **0.3253** | **0.3287** | **0.3285** |

### Agent-claude teacher: avg-pick-WR sweep

| alpha | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| baseline (0) | 0.5475 | 0.5467 | 0.5468 | 0.5470 |
| 0.5 | 0.5487 | 0.5478 | 0.5483 | 0.5483 |
| 1.0 | 0.5496 | 0.5487 | 0.5491 | 0.5491 |
| 2.0 | 0.5505 | 0.5496 | 0.5500 | 0.5500 |
| **4.0** | **0.5513** | **0.5505** | **0.5508** | **0.5509** |

### Oracle (real 17lands GIH-WR): alpha sweep (WR-agreement)

| alpha | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| 0.5 | 0.3250 | 0.3095 | 0.3223 | 0.3189 |
| 1.0 | 0.3850 | 0.3657 | 0.3849 | 0.3785 |
| 2.0 | 0.4842 | 0.4644 | 0.4856 | 0.4780 |
| **4.0** | **0.6230** | **0.6029** | **0.6262** | **0.6173** |

### Gap closure: agent-claude at alpha=4

| seed | baseline | oracle (α=4) | agent-claude (α=4) | gap | gap closed |
|---|---|---|---|---|---|
| 0 | 0.2628 | 0.6230 | 0.3316 | 0.3601 | **19.1%** |
| 1 | 0.2523 | 0.6029 | 0.3253 | 0.3506 | **20.8%** |
| 2 | 0.2570 | 0.6262 | 0.3287 | 0.3692 | **19.4%** |
| **mean** | **0.2573** | **0.6173** | **0.3285** | **0.3600** | **19.8%** |

The headline result is **remarkably stable across seeds** (SD = 0.9 pp), confirming this is a real
signal rather than seed noise.

### Heuristic floor: alpha sweep (WR-agreement)

| alpha | seed 0 | seed 1 | seed 2 | mean | gap closed (mean) |
|---|---|---|---|---|---|
| **0.5** | **0.2482** | **0.2388** | **0.2428** | **0.2433** | **−3.9%** |
| 1.0 | 0.2341 | 0.2252 | 0.2275 | 0.2289 | −8.0% |
| 2.0 | 0.2076 | 0.2039 | 0.2021 | 0.2045 | −14.6% |
| 4.0 | 0.1784 | 0.1762 | 0.1736 | 0.1760 | −22.6% |

The heuristic's best alpha is 0.5 (smallest blend). Even there it closes −4% — blending a
near-zero-correlation prior actively harms picks. The harness does not reward arbitrary priors; the
LLM signal is doing real work.

### Teacher quality

| teacher | n rated | Spearman vs real GIH | Pearson |
|---|---|---|---|
| agent-claude | 272 | **0.463** | 0.455 |
| heuristic | 265 | 0.081 | 0.149 |

A modest-correlation LLM teacher (Sp=0.463) still buys ~20% of the gap. The heuristic's Sp=0.081
buys nothing (and costs −4% at best alpha). The 272-card overlap for agent-claude vs 286 total
reflects 14 cards with no 17lands rating data (unplayable basics/tokens excluded from GIH metrics).

---

## Caveats

### 1. TRAINING-DATA LEAK (most important — upper bound)

**DSK (Duskmourn, September 2024) predates the scoring agents' knowledge cutoff.** Despite the
content-only rubric (oracle text + stats, no win-rate or meta signals), post-release format
knowledge can leak implicitly: the LLM has read DSK set reviews, tier lists, and draft content
published after release, and that knowledge may inform its 0–10 scores even when the scorer
believes it is judging structure only.

**The ~20% figure is an upper bound on true day-0 teacher value.** Only a set that is genuinely
post-cutoff (spoilers released after the model's training cutoff) would give a clean measurement.
The correct interpretation: "an agent with modest Spearman correlation to real GIH-WR closes ~20%
of the gap; a real day-0 agent would close no more than this, and possibly less."

### 2. Single holdout, small network, 4-set corpus

DSK is the only holdout tested. The 4-set corpus and Phase-4-era small network were chosen for
comparability with the scaffold's design, not the deployed recipe (which uses nested19 ~19 sets).
Gap-closure percentages may differ at production scale; 19-set training would likely raise both
baseline and oracle.

### 3. Oracle is ratings-greedy at high alpha

At alpha=4 the oracle policy is dominated by card power rankings; the gap the LLM closes is
measured against a strong but attainable target. The oracle at alpha=4 exceeds human WR-agreement
(0.617 vs 0.298 human ref) by a large margin — the LLM's 20% closure is measured against this
ceiling, not against human performance.

---

## Product implication

On release day the webapp could ship a **"day-0 mode"**: LLM-score the new set's spoiler list
(~$1 of API or one agent workflow), blend at alpha≈4, and recover **~a fifth of the
data-informed pick quality** before any 17lands data exists. This is the first day-0 number the
project has ever produced.

The mechanism scales to any new set: the CachedTeacher reads a pre-built ratings JSON, so no API
key is needed at inference time; the scoring workflow runs once per set on spoiler release. The
teacher file is ~50 KB per set.

A richer follow-up: per-pack soft-label KL distillation (DD-004 #1 pattern applied to the LLM
teacher) would capture synergy the per-card signal misses, at the cost of one LLM call per pick —
worth exploring if the ~20% headline holds on a post-cutoff holdout.

---

## Repro commands

```bash
# 1. Build teacher ratings (requires ANTHROPIC_API_KEY; ~240k tokens; result already cached)
PYTHONPATH=src .venv/bin/python -m mtg_draft_ml.distill.teacher \
    --manifest data/hf/manifests/DSK.PremierDraft.sample60000.json \
    --scryfall data/hf/scryfall/dsk.json \
    --teacher anthropic --set-code DSK \
    --out data/teacher_cache/DSK.cached_agent-claude.ratings.json

# 2. Run cold-start scaffold, 3 seeds (uses cached teacher — no API key needed):
for SEED in 0 1 2; do
  PYTHONPATH=src .venv/bin/python scripts/pod_coldstart.py \
    --train-sets BLB,OTJ,WOE,MKM --holdout DSK \
    --teacher heuristic,cached:agent-claude \
    --seed $SEED \
    --out /tmp/ws24_seed${SEED}.json
done
```

Raw JSON outputs: `docs/results/ws24/ws24_seed{0,1,2}.json`
Teacher scores (reproducibility artifact): `docs/results/ws24/agent-claude.DSK.ratings.json`
