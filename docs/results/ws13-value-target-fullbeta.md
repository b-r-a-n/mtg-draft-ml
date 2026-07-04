# WS1.3 — Re-test `deck_value`-as-target at 19 sets with reliable β

**Why.** The "deck_value target subsumed by corpus breadth" verdict
([game-data-value-model.md](game-data-value-model.md), compounding test) was reached with the noisy
β (split-half ρ=0.644). The concern: if the ruler misstates the outcome signal, the arm comparison
misstates which teacher wins. WS1.1 rebuilt β on full game_data (ρ 0.644 → **0.860**;
[full-data-deck-value.md](full-data-deck-value.md)) and WS1.2 established the new outcome-eval
baselines ([ws12-rebaseline.md](ws12-rebaseline.md)). Per
[breakthrough-plan.md](../breakthrough-plan.md) WS1.3, the decisive re-test is:
`scripts/pod_game_value_target.py` at nested19 (19 sets; ground rule 6), holdout DSK, big net
(emb512/h1024/L4), ≥3 seeds, measured on both (a) WR-agree:GIH and (b) the outcome eval under
full-β. Adopt only if either prong improves > +0.01 beyond seed noise; otherwise REFUTED.

## Method

**Two RunPod RTX A5000 sessions (~$2 total).**

*Run A (seeds 0,1,2):* `scripts/pod_game_value_target.py`, targets `gih`, `+value`, `value`,
arms `good_base / good_comp_gih / good_comp_+value / good_comp_value`. This run also refuted the
pure `value` target (WR-agree:GIH δ = −0.0026, −0.0047, −0.0143 vs gih → mean **−0.0072**), so
Run B dropped it.

*Run B (seeds 3,4,5):* confirmation run, targets `gih` and `+value` only. Added
`--export-onnx-dir` (commit 4aadf48) so the trained arms could be exported to ONNX and the
outcome-eval prong could run locally without re-spinning a pod.

*Outcome eval (local, seeds 3–5):* `scripts/run_outcome_eval.py` on each of the 6 exported ONNX
files (3 seeds × 2 arms: `good_comp_gih`, `good_comp_+value`), 800 held-out DSK drafts, playprob
build, full-β. The `--webapp-dir` flag points at a temporary directory containing the arm ONNX.

Raw artifacts: `ws13/ws13_results.json` (seeds 0–2), `ws13/ws13b_results.json` (seeds 3–5),
`ws13/out.good_comp_{gih,plusvalue}.s{3,4,5}.json` (outcome eval, 6 files).

## Results

### Prong 1 — WR-agree:GIH (paired per-seed, +value − gih)

| seed | gih | +value | **delta** |
|---|---|---|---|
| 0 | 0.3058 | 0.3097 | **+0.0039** |
| 1 | 0.3303 | 0.3413 | **+0.0110** |
| 2 | 0.3229 | 0.3368 | **+0.0138** |
| 3 | 0.3302 | 0.3184 | **−0.0117** |
| 4 | 0.3209 | 0.3299 | **+0.0090** |
| 5 | 0.3112 | 0.3245 | **+0.0133** |
| **mean ± sd** | | | **+0.0065 ± 0.0097** |
| **paired t (5 df)** | | | **t = 1.66** |
| **positive** | | | **5/6** |

**Gate: FAILS.** Requires > +0.01 beyond seed noise. Mean is +0.0065 and the paired t does not
clear a meaningful threshold (p ≈ 0.16).

### The 3-seed head-fake

After seeds 0,1,2 the picture looked like: mean **+0.0096**, sd 0.0051, paired **t = 3.23**, 3/3
positive — a borderline near-pass that reversed the old noisy-β conclusion (which had read
flat-to-negative). This matched the critical region where the plan's gate is defined (+0.01). But
project history predicted exactly this pattern: good-player reversal, Step-2 single-seed ceiling
break, the compounding test itself. Seed 3 returned −0.0117 and the distribution regressed to
+0.0065. The confirmation run was worth its cost.

### Prong 2 — Outcome eval (headline metric)

800 held-out DSK drafts, playprob build, full-β. These are HOLDOUT models (DSK unseen, trained on
19 sets) scored against the deployed baseline (+0.0567; [ws12-rebaseline.md](ws12-rebaseline.md)).
The arm-vs-arm comparison is protocol-matched; the smaller absolute margins (+0.00–+0.01) vs the
deployed model reflect that the deployed model was trained WITH DSK in distribution — do not
interpret the small absolutes as a regression.

| seed | +value model δ vs human | gih model δ vs human | **paired delta** |
|---|---|---|---|
| 3 | +0.0110 | +0.0122 | **−0.0011** |
| 4 | +0.0027 | +0.0030 | **−0.0002** |
| 5 | +0.0037 | −0.0003 | **+0.0040** |
| **mean ± sd** | | | **+0.0009 ± 0.0027** |
| **positive** | | | **1/3** |

**Gate: FAILS decisively.** Mean paired delta +0.0009, 1/3 positive — indistinguishable from zero.

### Secondary: WR-agree:value (circular metric, discounted)

WR-agree:value measures agreement with `deck_value` scores — the same scorer used as the additional
training target in `+value`. This metric is circular: a model trained to predict deck_value will
agree with deck_value more, independent of whether that translates to better actual outcomes.
Per the plan's metric policy, wins here are discounted.

| seed | delta (+value − gih) |
|---|---|
| 0 | +0.0074 |
| 1 | +0.0237 |
| 2 | +0.0083 |
| 3 | +0.0011 |
| 4 | +0.0075 |
| 5 | +0.0140 |
| **mean ± sd** | **+0.0103 ± 0.0077** |
| **paired t (5 df)** | **t = 3.28, 6/6 positive** |

Consistent and statistically clear — but expected by construction. Reported for completeness only.

### top-1 pick accuracy

Per-seed +value − gih top-1 deltas: +0.0028, −0.0015, +0.0034, +0.0041, +0.0001, −0.0043.
Mean **+0.0008** — flat. Not a gate metric; confirms no top-1 cost or gain.

## Verdict

**REFUTED.** `deck_value`-as-teacher adds nothing over the GIH-composite teacher when measured
honestly: the WR-agree:GIH signal is +0.0065 (t=1.66, noise) and the outcome prong is +0.0009
(1/3 positive, decisive null). The 3-seed +0.0096 result was a head-fake; the confirmation run
resolved it correctly.

The pure `value` target (no composite) is also refuted at mean −0.0072 (3 seeds, Run A).

**Keep the GIH-composite teacher.** Corpus breadth (nested19) subsumes the deck_value training
signal even with a reliable ruler (ρ 0.87). `deck_value` remains the pick-time dial and outcome
eval metric — it is not a training target.

## Repro

**Pod training (Run B, seeds 3–5, with ONNX export):**

```
PYTHONPATH=src .venv/bin/python scripts/pod_game_value_target.py \
  --train-sets BLB,OTJ,WOE,MKM,LCI,MH3,MOM,FDN,DFT,TDM,FIN,EOE,ONE,BRO,DMU,SNC,NEO,MID,LTR \
  --holdout DSK --targets gih,+value --seeds 3,4,5 \
  --emb-dim 512 --enc-hidden 1024 --enc-layers 4 \
  --epochs 10 --export-onnx-dir /tmp/ws13_onnx
```

**Local outcome eval (per arm per seed):**

```
PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py \
  --set DSK --n-drafts 800 --l2 30 --build playprob \
  --game-npz data/game/game.DSK.PremierDraft.full.npz \
  --webapp-dir /tmp/ws13_onnx/good_comp_gih_s3 \
  --out docs/results/ws13/out.good_comp_gih.s3.json
```

Repeat for each of the 6 arm/seed combinations (`good_comp_gih` and `good_comp_plusvalue`,
seeds 3–5). **Gotcha: `--game-npz` must be passed explicitly** (same as WS1.2 — the default glob
matches both full and sample caches in arbitrary order).
