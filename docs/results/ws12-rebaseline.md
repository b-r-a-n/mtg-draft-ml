# WS1.2 — re-baseline the outcome eval on the reliable full-data β

**Why.** The outcome eval ([outcome-eval.md](outcome-eval.md)) is the headline metric, but its ruler —
the `deck_value` β — was fit on 150k-game samples with split-half ρ=0.644. WS1.1 rebuilt β on the
full game_data (ρ 0.644 → **0.860**; [full-data-deck-value.md](full-data-deck-value.md)), so per
[breakthrough-plan.md](../breakthrough-plan.md) WS1.2 the eval must be re-run with the reliable β to
produce the margins all later workstreams are judged against. This is a re-baseline, not a new
method: same script (`scripts/run_outcome_eval.py`, `--l2 30`), same 800 held-out DSK drafts, the
currently deployed big-net ONNX — a 2×2 of build ∈ {playprob, naive top-23} × β ∈ {full, sample-150k}.

## Result — new official baselines (full-β, current deployed model)

**playprob: model +0.0567 est. deck-WR vs human, beating the human seat in 92% of drafts;
naive top-23: +0.0717 / 98%.** These replace the June numbers (+0.049/84%, +0.063/95%,
[outcome-eval.md](outcome-eval.md)) as the reference baselines.

Mean estimated deck-WR, all five policies (JSONs: `ws12/outcome-eval.<build>.<beta>.json`):

| build × β | human | **model** | gih_greedy | deckvalue_greedy (oracle) | random |
|---|---|---|---|---|---|
| playprob × **full** | 0.6325 | **0.6892** | 0.7232 | 0.7504 | 0.5095 |
| playprob × sample | 0.6032 | 0.6646 | 0.7078 | 0.7423 | 0.4714 |
| naive × **full** | 0.6728 | **0.7446** | 0.7750 | 0.7886 | 0.5745 |
| naive × sample | 0.6539 | 0.7299 | 0.7620 | 0.7884 | 0.5459 |

Δ vs human (model also shows % of drafts beating the human seat):

| build × β | **model** | gih_greedy | deckvalue_greedy | random |
|---|---|---|---|---|
| playprob × **full** | **+0.0567 (92%)** | +0.0908 | +0.1180 | −0.1230 |
| playprob × sample | +0.0614 (90%) | +0.1046 | +0.1391 | −0.1318 |
| naive × **full** | **+0.0717 (98%)** | +0.1021 | +0.1158 | −0.0984 |
| naive × sample | +0.0760 (98%) | +0.1081 | +0.1345 | −0.1080 |

## What it means

1. **New official baselines:** playprob **+0.0567 (92%)**, naive top-23 **+0.0717 (98%)** — full-β,
   current deployed model. All later workstreams (WS1.3, WS2.x, WS3.x, WS4.1) are judged against
   these, not the June numbers.
2. **The June verdict survives the reliable ruler.** The model's decks still beat the humans'
   decisively — 92–98% of seats — so "imitation-style drafting yields outcome-better decks" was not
   a noise artifact of the ρ=0.644 β.
3. **The noisy ruler inflated the self-referential policies most.** `deckvalue_greedy` — greedy on
   the ruler itself — loses ~0.02 of margin under full-β (+0.1391 → +0.1180 playprob): classic
   winner's curse, where a noise-inflated β gets picked *and* credited by the same β. The model,
   which never sees β, trims only ~0.005 (+0.0614 → +0.0567); `gih_greedy` sits in between. A
   cleaner ruler compresses exactly the margins you'd expect it to.
4. **Decomposition vs June:** the June baselines used the *older* deployed net. Current-net +
   sample-β gives +0.0614/90% (playprob), so most of the June→now margin growth (+0.049 → +0.0567)
   is the big-net model upgrade (commit 701b4f6); the ruler swap slightly *trims* measured margins.
   Note the playprob beat-rate *rises* 90→92% under full-β despite the smaller mean margin —
   de-noising reduces variance as well as bias.
5. **Known caveat, unchanged:** the metric is a card-power sum, so greedy rating policies
   (`gih_greedy`) still outscore the model policy — it measures "picks winning-er cards than
   humans", not "drafts optimally" ([outcome-eval.md](outcome-eval.md)).

## Repro

```
PYTHONPATH=src .venv/bin/python scripts/run_outcome_eval.py --set DSK --n-drafts 800 --l2 30 \
  --build playprob --game-npz data/game/game.DSK.PremierDraft.full.npz \
  --out docs/results/ws12/outcome-eval.playprob.fullbeta.json
```

…and the other three combinations (`--build naive`, `--game-npz …sample150000.npz`; the file names
in `docs/results/ws12/` spell out the sweep). **Gotcha: `--game-npz` MUST be passed explicitly** —
the script's default glob `data/game/game.<SET>.*.npz` now matches *both* the full and sample caches
in arbitrary order.

**Scope:** single set (DSK), 800 drafts — same as the June baseline; multi-set replication is cheap
if a later workstream needs it.
