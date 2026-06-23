# Distillation in this project (DD-004)

Two distillation tracks live in `src/mtg_draft_ml/distill/`, attacking different problems:

| track | pattern | problem | teacher | status |
|---|---|---|---|---|
| **ensemble-of-seeds** | DD-004 #1 | sample efficiency on the data we already have | K seed models, same data | `ensemble.py`, `scripts/pod_distill.py` |
| **WR-softmax** | DD-004 #1 | good-not-just-human, as a *dense* target | win-rate ranking over the pack | `wr.py`, `scripts/pod_wr_distill.py` |
| **leaky → release-day** | DD-004 #3 | smuggle privileged knowledge into a release-day student | a win-rate-augmented model | `leaky.py`, `scripts/pod_leaky_distill.py` |
| **cold-start** | DD-004 #4 | zero data on a new set's release day | LLM reading oracle text | `teacher.py`, `coldstart.py`, `scripts/pod_coldstart.py` |

All four share `pack_distillation_kl` + the opt-in `train_loop` hook; the teachers all expose the same
`mean_probs(pool, pool_mask, pack, pack_mask, temp)` interface, so `CompositeTeacher` can average any
of them into one target (e.g. denoise *and* bias toward winning).

The ensemble track is the higher-leverage one for the project's central finding (the ~0.58 ceiling
and the flat data/capacity scaling are *objective*-bound, not data-bound). The cold-start track is a
different, narrower regime. Both are below.

---

# Ensemble-of-seeds soft-label distillation (DD-004 #1)

**Status:** scaffold (`distill/ensemble.py`, runner `scripts/pod_distill.py`, KD loss in
`training/losses.py`, opt-in hook in `training.train_content.train_loop`).
**Question it answers:** the one-hot human pick collapses an 8-15 card pack to a single bit and is
itself noisy (individual-drafter disagreement *is* the ~0.58 top-1 ceiling). Does a denser, denoised
target — the averaged pack ranking of K seed models — extract more from the **same drafts**?

## Why this is the higher-leverage KD use case

The settled negative results are sample-efficiency statements: 60k→500k picks/set is flat, and S/M/L
capacity all plateau at ~0.58. That points at the *objective* (a lossy hard label), not the data or
the model, as the bottleneck. And it's a gap in what's already built: `pick_advantage_weights` only
*reweights* the hard-label CE example (its docstring: "WITHOUT ever changing the target") — the IWD
signal is a scalar multiplier. Distilling a ranking *into the target* is the dense form of that lever.

## The loss

```
L = (1−λ)·CE(logits, human_pick) + λ·T²·KL(teacher ‖ softmax(logits/T))   # over the pack
```

`training/losses.py::pack_distillation_kl` is the KD term (masked to the pack, no -inf arithmetic);
`topk_renormalize` optionally restricts the soft target to the teacher's **top-k** picks so the loss
isn't dominated by the unplayable tail. `train_loop` gains opt-in params (`teacher`,
`distill_lambda`, `distill_temp`, `distill_topk`) — default-off, byte-identical when unused, and
composable with the existing win/adv/aux hooks. It requires the CE path (in-pack logits; InfoNCE's
global negatives are refused).

## The teacher: an ensemble of seeds, run on the fly

`EnsembleTeacher` holds K independently-seeded models (distinct init + split + shuffle for diversity)
and returns `mean_probs` = the mean of their pack softmaxes — a denoised crowd-of-models ranking. It
runs **inside the student's training step** rather than caching per-pick distributions to disk: the
models are small (~2-15M params) and frozen, so K extra forward passes/batch is cheap, and it
sidesteps the example-id join a cached path would need.

Ensemble-of-seeds is deliberately first because it's the purest test of "denser target on the same
data" — no leaky features, no LLM, no new data, so a gain is unambiguously the objective's doing. The
WR-softmax teacher (good-not-just-human as a dense per-pack target) and the leaky-feature→release-day
teacher (DD-004 #3) are the natural follow-ups that add *information*, separable from this denoising.

## The experiment

`run_ensemble_distill` trains K teachers + a CE baseline + a CE+KD student (baseline and student
share seed 0 and the same data), then compares all three on a held-out set:

| policy | meaning |
|---|---|
| baseline (CE) | single seed, hard label only |
| **distilled (CE+KD)** | single seed + ensemble soft target — the candidate |
| ensemble (target) | the K-model average — the upper reference KD chases |

**Read the right metrics.** If the residual is genuine label noise, **top-1 stays ceilinged even
when the picks get better** — the same reason the project concluded 0.58 is a ceiling, and it applies
to the eval label too. Judge on **WR-agreement / avg-pick-WR** and **ranking quality** (top-3/5,
MTPD), and on the sample-efficiency test: `--train-frac 0.25` asks whether KD on a quarter of the
data matches plain CE on all of it. The report prints the per-metric delta and the fraction of the
ensemble gap the student closes.

## Running it

```bash
uv run python scripts/pod_distill.py                  # full-data CE+KD vs CE vs ensemble
uv run python scripts/pod_distill.py --train-frac 0.25 --distill-topk 5
```

## Caveats / next steps

- Self-distillation gains are real but bounded — the ensemble can only denoise variance the seeds
  disagree on; it adds no information the hard labels lack. The information-adding teachers (WR-softmax,
  leaky-feature→release-day) are the follow-ups, and KD's plumbing here (`pack_distillation_kl`,
  `train_loop` hook) is teacher-agnostic, so they drop in by swapping `EnsembleTeacher.mean_probs`.
- If top-1, top-k, *and* WR-agreement are all flat vs the CE baseline, that is itself the result: it
  corroborates the noise-ceiling conclusion from the objective side rather than the data/capacity side.

---

# WR-softmax soft-label distillation (DD-004 #1) — good-not-just-human as a *dense* target

**Status:** scaffold (`distill/wr.py`, runner `scripts/pod_wr_distill.py`; reuses the same KD term and
`train_loop` hook as the ensemble track).
**Question it answers:** `pick_advantage_weights` already biases toward winning picks — but only by
*reweighting* the hard-label CE example by a scalar (`exp(advantage/tau)`), never reshaping the
target. Does turning the win-rate signal into a **dense per-pack target** (the student matches the
full WR ranking over the pack) beat that scalar reweighting for taking winning picks?

This is the second distillation track on purpose: ensemble-of-seeds denoises the *same* labels and
adds no information; WR-softmax *adds information* (outcome win rate the hard label lacks). Running
them separately keeps "denoising" and "good-not-just-human" cleanly attributable.

## The teacher

`WRSoftmaxTeacher.mean_probs` returns `softmax(card_WR / tau)` over the pack — a valid distribution
(0 at pads). It's built from the **training sets'** 17lands ratings via
`eval.winrate.build_global_wr_targets(standardize=True)` (z-scored relative WR + a rated mask).
Unrated real cards sit at the pack-mean score (unrated ≠ weak); packs with <2 rated cards carry no
WR signal and fall back to uniform. `tau` sets target sharpness; the KD `distill_temp` still softens
the *student* (default 1.0 here — match a sharp WR target directly). Same `mean_probs` interface as
`EnsembleTeacher`, so it reuses `pack_distillation_kl` and the `train_loop` hook unchanged — the
payoff of building the plumbing teacher-agnostic.

## The experiment

`run_wr_distill` trains three students (same seed, same data) and compares on a held-out set:

| policy | win-rate signal | meaning |
|---|---|---|
| baseline (CE) | none | hard label only |
| **WR-KD (dense)** | KL to softmax(WR/τ) over the pack | reshape the target — the candidate |
| advantage (scalar) | `exp(advantage/tau)` example weight | the existing method (`pick_advantage_weights`) |

The win-rate signal injected on the *training* sets is evaluated on an *unseen* set (does it
generalize through the content encoder). Read **WR-agreement** and **avg-pick-WR** against the
holdout's real ratings, with the human pick as the floor; the report prints `dense − scalar` on both
(>0 ⇒ reshaping the target beats reweighting the example). Top-1 is not the metric here — by design
both methods *deviate* from the human pick toward higher-WR cards.

## Running it

```bash
uv run python scripts/pod_wr_distill.py                                   # GIH WR
uv run python scripts/pod_wr_distill.py --wr-field drawn_improvement_win_rate --distill-topk 5
```

## Caveats

- GIH/IWD win rate is a confounded proxy (favors controlling decks); `build_global_wr_targets`
  z-scores within set to remove per-format baselines, but archetype confounds remain — the dense
  target inherits them.
- It can be combined with the ensemble teacher (denoise *and* bias toward winning) by averaging the
  two `mean_probs`; left as a follow-up to keep the first comparison clean.

---

# Leaky-feature → release-day distillation (DD-004 #3)

**Status:** scaffold (`distill/leaky.py`, runner `scripts/pod_leaky_distill.py`; reuses the KD term,
the `train_loop` hook, and `EnsembleTeacher` as a single-model wrapper).
**Question it answers:** a teacher given per-card win rate as an input feature is more powerful — but
win rate is a post-hoc aggregate that doesn't exist on a new set's release day and leaks outcome
information. Can its win-rate-informed *contextual* policy be distilled into a student that sees only
release-day inputs, so the student needs no win-rate features at inference?

## Distinct from the WR-softmax teacher

WR-softmax makes the target the *pure* win-rate ordering (ignores pool synergy). The leaky teacher
*learns* a full policy that blends win rate with the pool context — "high-WR **and** fits your pool" —
a richer target. That's why DD-004 #3 is separate from #1.

## Mechanism

The teacher is a `ContentDraftModel` on a **win-rate-augmented content matrix**:
`augment_with_winrate(base, wr_z, wr_mask)` appends two columns — z-scored win rate (0-filled where
unrated) and a rated flag (so the encoder can tell "WR≈0 because average" from "0 because unrated").
The teacher's encoder is two inputs wider; everything downstream (pool encoder, pick head) is
unchanged, so it produces pack logits over the same vocab and wraps in `EnsembleTeacher([teacher])`
to expose the standard `mean_probs`. The student trains on the **base** matrix (release-day) with the
KD term — no architecture change, no win-rate features at inference.

## The experiment

`run_leaky_distill` compares on a held-out set:

| policy | sees win rate? | meaning |
|---|---|---|
| baseline (CE) | no | release-day floor |
| **distilled (CE+KD)** | no (distilled from a teacher that did) | the candidate |
| teacher (leaky) | yes (incl. the holdout's real WR) | the ceiling the student chases |

Read WR-agreement / avg-pick-WR; the report prints `distilled − baseline` and the fraction of the
teacher gap closed. The teacher's holdout eval uses the holdout's real win-rate columns — an offline
upper reference; a real release-day deploy wouldn't have them, which is exactly why the student must
inherit the knowledge instead.

## Running it

```bash
uv run python scripts/pod_leaky_distill.py
uv run python scripts/pod_leaky_distill.py --wr-field drawn_improvement_win_rate --distill-topk 5
```

## Composing teachers

`CompositeTeacher([ensemble, wr_or_leaky], weights=[1, 2])` averages teachers' distributions into one
KD target — denoise *and* bias toward winning in a single pass, since every teacher shares the
`mean_probs` interface. Drops into the same `train_loop` hook unchanged.

## Caveat

The win rate the teacher learns from is a confounded proxy (favors controlling decks); the student
inherits that bias along with the signal. The leaky teacher's *ceiling* on the holdout also depends
on the holdout's real ratings, so read "fraction of gap closed" as "how transferable was the
win-rate-informed policy," not an absolute deployable number.

---

# Cold-start LLM-teacher distillation (DD-004 #4)

**Status:** experiment scaffold (code in `src/mtg_draft_ml/distill/`, runner `scripts/pod_coldstart.py`).
**Question it answers:** on a brand-new set's release day there is *zero* 17lands data — can an LLM
teacher reading oracle text + stats provide enough pick guidance to be worth deploying before human
data accumulates?

## Why this is the one niche with an LLM advantage

The project's two ceilings are settled (`docs/SUMMARY.md`): the content encoder already generalizes
to unseen **cards** (~0.57 top-1), and ~0.58 is a human-noise ceiling, so "predict humans better" is
a dead end. The remaining real win is "pick *better*", and IWD advantage-weighting already does that
**when 17lands data exists**. The gap it can't close is the new-**format** cold start: the content
encoder scores a new set's cards structurally, but *which* cards are actually strong in this format
is set-specific knowledge that only accrues as drafts are logged. That's the single slice where an
LLM that has read the cards has signal nothing else has on day one (DD-004 pattern #4 — distinct from
the new-*card* problem DD-001 already solves).

## The design: the teacher is a stand-in ratings file

The whole scaffold rests on one decision. A 17lands ratings file is a list of
`[{"name", "ever_drawn_win_rate"}, ...]`, consumed everywhere via `eval.winrate.align_winrates`,
`WRMeter`, and the `deploy.Drafter` quality-blend dial. So the teacher emits the **same shape** —
`[{"name", "llm_quality"}, ...]` — and becomes a drop-in for the ratings file that doesn't exist yet.
No change to the locked eval/inference path; the LLM is just "a ratings provider that needs no draft
data".

```
Scryfall card (text + stats, NO meta/WR)  ──►  Teacher.rate()  ──►  llm_quality ∈ [0,10]
        per card                                                          │
                                                  build_teacher_ratings ──┘──►  17lands-shaped JSON
                                                                                      │
                          align_winrates / WRMeter / quality-blend dial  ◄───────────┘
```

`distill/teacher.py`:
- `card_brief(record)` — the **release-day-only** view handed to the teacher: name, mana cost, type,
  P/T, rarity, oracle text. Deliberately *excludes* win rate / pick rate / meta — the teacher must
  judge from the card alone, exactly as a human would on day one.
- `HeuristicTeacher` — deterministic stats-only proxy, no API key. The offline/test path and a floor
  baseline (mirrors the `hash` text-embedder stand-in pattern).
- `AnthropicTeacher` — Claude (`claude-opus-4-8`, adaptive thinking) rates each card's Limited power
  via structured outputs, batched, **disk-cached per (teacher, set, card)** so re-runs are free and
  an interrupted run resumes. Needs the `[distill]` extra (`anthropic`, `pydantic`) + `ANTHROPIC_API_KEY`.

## The experiment: baseline vs LLM vs oracle

`distill/coldstart.py::run_coldstart` reuses the leave-one-set-out frame: train the content model on
N sets, then treat a held-out set as "the new set on release day" and compare three pick policies on
the **same trained model** — all scored against the real 17lands signal, which we *do* have for the
held-out set (that's what makes this a clean offline test):

| policy | blend signal | meaning |
|---|---|---|
| no-data baseline | none (α=0) | pure content-encoder zero-shot |
| **LLM cold-start** | teacher `llm_quality` | what we could ship on day one |
| real-data oracle | real 17lands WR | the upper bound the LLM is chasing |

Both quality vectors are z-scored (ignoring missing) so `blend_alpha` means the same thing across
sources (same rationale as `deploy.Drafter`'s scale-invariant dial). Headline outputs:
- **% of the oracle gap closed** on WR-agreement: `(LLM − baseline) / (oracle − baseline)`.
- **teacher↔real rank correlation** (Spearman/Pearson over cards rated by both) — does the LLM even
  order the set correctly?
- the human WR-agreement floor, for the "good, not just human" framing.

A large fraction closed ⇒ day-one deployment is justified. A small fraction ⇒ cleanly ruled out
(and the correlation tells you whether the teacher's *ranking* or only the *blend strength* was the
problem).

## Running it

```bash
# 1. smoke-test the whole pipeline offline (free, deterministic):
uv run python scripts/pod_coldstart.py --teacher heuristic

# 2. the real teacher (needs the [distill] extra + ANTHROPIC_API_KEY):
uv run python scripts/pod_coldstart.py --teacher anthropic
```

Or build a ratings file standalone and feed it to any rating-aware path:

```bash
uv run python -m mtg_draft_ml.distill.teacher \
    --manifest data/hf/manifests/DSK.PremierDraft.sample60000.json \
    --scryfall data/hf/scryfall/dsk.json --teacher anthropic --set-code DSK \
    --out data/teacher_cache/DSK.ratings.json
```

## Caveats / next steps

- The current teacher produces a **per-card quality** signal (the cheapest, highest-leverage form —
  ~250 cards/set, reuses all existing machinery). DD-004 pattern #1 is richer: **per-pack soft-label
  KL** distillation where the LLM ranks each actual pack given the pool. That captures
  context/synergy the per-card signal can't, at the cost of one LLM call per pick (~tens of thousands)
  and a new KL term in `training.losses` — a clear follow-up if the per-card result is promising.
- The teacher is blind to the format's eventual metagame by construction; it estimates *raw* card
  power, which GIH win rate confounds with archetype/speed. Expect the teacher↔real correlation to be
  capped by that confound, not only by the LLM.
