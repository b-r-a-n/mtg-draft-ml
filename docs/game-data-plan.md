# Next work — game_data value model (the less-confounded signal)

**Status:** **Step 0 done → PROCEED** (2026-06-24). The go/no-go ran on DSK (80k-game sample): the
per-game β_c is a genuinely different, less-confounded card-value ranking — Spearman(β, IWD) = **0.60**
(≪ 0.95 stop-threshold), flat across L2 ∈ [1, 1000], well-sampled-stable, face-plausible (bombs/removal
top, lands/filler bottom). Full result: [`docs/results/game-data-value-model.md`](results/game-data-value-model.md).
**Step 1 (full value model) is greenlit.** The one remaining lever with real headroom after the
distillation + data investigation (see `docs/results/`).

## Why this, why now

Every "good, not just human" result plateaus at **WR-agreement 0.29–0.31**: average humans 0.298,
good humans 0.315, our best model (good players + composite-WR target) 0.297. We've now shown the
ceiling is **not** in the model, the imitation data, or the human labels:

- depth saturates (60k→240k flat), more sets help only baseline top-1, ensemble denoising is null;
- good-player *labels* genuinely help WR (+0.0057 vs volume-matched random) but can't pass the bar;
- the WR *target* (GIH-WR / IWD) is the binding constraint — it's a **confounded proxy** (a card's
  win rate is entangled with the decks/archetypes/players it rides with).

So the next move is a **less-confounded card-value signal**, built from the raw match outcomes we've
never touched: 17lands **game_data** (the Phase-4 "deck-strength value model", roadmap-gated on
"rich enough outcome data" — which exists).

## The data (confirmed available)

`game_data_public.<SET>.<EVENT>.csv.gz` on the 17lands S3 bucket — one row per game, ~1400 wide:
- **`won`** (binary outcome) — the label.
- per-card: **`deck_<card>`** (count in deck), `drawn_/opening_hand_/tutored_/sideboard_<card>` (~270 cards).
- controls: **`on_play`, `num_mulligans`, `num_turns`, `opp_colors`, `user_game_win_rate_bucket`** (player skill).

## The idea

Fit a **regularized logistic regression per set**:

```
P(won) = σ( Σ_c β_c · deck_count_c  +  γ·on_play  +  δ·player_skill  +  … )
```

`β_c` is card *c*'s **marginal** contribution to winning, **controlling for the rest of the deck**
(de-confounds "good cards ride with good cards"), for player skill, and for on-the-play. This is a
stronger de-confounding than GIH-WR (raw, fully confounded) or IWD (controls only for *having* the
card, not for *what else is in the deck*). `β_c` aligned to the manifest = a new card-value field
that drops straight into `composite_card_quality` / the WR-softmax teacher.

It also unlocks a **better eval metric**: instead of "does the bot agree with the confounded GIH-WR
ranking", score the bot's drafted pool by its **model-estimated deck win rate** — the outcome we
actually care about (roadmap "estimated deck-WR").

## Staged plan — with a go/no-go gate (don't build big on an uncertain payoff)

**Step 0 — go/no-go (≈1 day, one set, cheap). ✅ DONE → PROCEED.** Added `download_17lands_game`
(`data/download.py`), a slim game→deck-matrix preprocessor (`data/game_preprocess.py`), the L2
logistic fit + ratings comparison (`eval/game_value.py`), and the driver `scripts/game_value_gonogo.py`.
Ran on DSK (80k games). **Spearman(β, IWD) = 0.60, Spearman(β, GIH) = 0.72** — well below the 0.95
stop-line, signal-not-noise (flat across L2, well-sampled-stable), face-plausible. β demotes
"swingy-when-drawn" cards IWD over-rates and promotes solid contributors IWD under-rates. Full
writeup: [`results/game-data-value-model.md`](results/game-data-value-model.md). Original gate:
- `β_c` ≈ IWD (corr ≳ 0.95) → game_data adds little de-confounding beyond IWD we already use → **stop**.
- `β_c` meaningfully differs (demotes mediocre-but-ride-along cards, promotes genuinely impactful
  ones) → **proceed**. This gates the expensive ingest on evidence the new signal is actually new.

**Step 1 — full value model. ✅ DONE (2026-06-24).** Built `deck_value` for all 8 corpus sets (150k
games/set, l2=30) via `scripts/game_value_build.py` → `gamevalue/<SET>.PremierDraft.gamevalue.json`
(ratings-shaped, pushed to HF; `align_winrates(field="deck_value")` / `composite_card_quality` consume
it unchanged). CPU-only, ~3 min. **Validated:** face-plausible in every set (top = the format's bombs:
Oko/OTJ, Gruff Triplets/WOE, Elesh Norn/MOM, …), Sp(β,IWD) mean **0.625** (all ≪ 0.95 → its own signal
in every set, DSK wasn't special), split-half stability ρ mean **0.644**, reprint consistency ρ=0.86.
Full writeup: [`results/game-data-value-model.md`](results/game-data-value-model.md#step-1--the-full-per-set-value-field-done-2026-06-24).

**Step 2 — use it. ✅ DONE (2026-06-24, 4 seeds).** Plugged `deck_value` into the composite-WR target,
re-trained the best config (good players + composite, 7-set big net), evaluated vs GIH and vs
deck_value. **Result: `deck_value` is a better target** (top-1 +0.009, WR-agree-vs-deck_value +0.023
[4/4 seeds]) **but the single-seed GIH-ceiling-break did NOT survive multi-seed** (GIH-agree +0.007,
3/4, within the ±0.005–0.01 noise band — below the bar). Headline question ("does it push past the
0.29–0.31 GIH ceiling?") → **not confirmed**. Code `scripts/pod_game_value_target.py`; full writeup
[`results/game-data-value-model.md`](results/game-data-value-model.md#step-2--the-de-confounded-target-re-trained-4-seeds-2026-06-24).

**Step 3 — outcome eval. ✅ DONE (2026-06-25).** Replay held-out drafts, score each policy's drafted
pool by the game_data deck-value model (`scripts/run_outcome_eval.py`, `eval/outcome.py`). **The
deployed (GIH-trained) model's decks beat the humans' by +0.063 est. deck-WR in 95% of drafts — and
non-circularly** (GIH-trained, deck_value-scored). Full writeup
[`results/outcome-eval.md`](results/outcome-eval.md). **Honest gap:** the metric is σ(Σ top-23 β) — a
card-power sum that ignores **mana curve / color distribution / synergy**, so greedy rating policies
beat it; the next rung is a constrained best-deck builder (pick the best 2-color pair, enforce a curve
+ creature count from the `ci`/`cmc`/`t` fields in cards.json) so the score reflects a playable deck.

## Prereq (cheap, do first): multi-seed confirmation — ✅ DONE (2026-06-24)

Every recent delta (+0.0057 good-vs-rand, +0.0053 composite, +0.0068 more-sets) is ~0.005 — within
single-seed noise; we've trusted the *consistent ordering*, not the magnitude. Before investing in
game_data, **rotate 3–5 seeds** on the current best config (good players + composite) to confirm the
orderings and get error bars. ~30 min on a pod; bankable either way. `scripts/rotate_seeds.py` exists.

**Done.** Ran a fresh **4-seed** pass (RunPod A5000, ~3h, ~$0.87) over both scales via
`scripts/pod_skill.py --seeds 0,1,2,3` — 4-set small net and the 7-set big-net best config. It
**reproduces [`results/good-players.md`](results/good-players.md) to the digit**, so those numbers are
solid, not single-run flukes. Key facts for the game_data go-decision:
- **Seed noise floor ≈ ±0.005** (1σ on WR-agreement). A Step-2 game_data lift must clear **~0.01
  (≈2σ)** to be unambiguously real.
- **The ~0.30 WR-agreement ceiling is robust across seeds.** Best confirmed config (good + composite,
  7-set big net) = **0.2988 ± 0.0052 ≈ average human (0.298)**; even the strong levers plateau there.
- **Lever orderings confirmed with error bars:** composite-WR target is large & robust (~0.26→0.30,
  ≫ noise); good-player *labels* are real but small and **scale-dependent** (good−rand WR
  +0.0018±0.0070 [3/4] at 4-set-small → **+0.0158±0.0039 [4/4]** at 7-set-big); more-sets/capacity
  lifts good/comp +0.011 and flips top1-on-good positive. Nothing breaks ~0.30 → the ceiling is the
  confounded GIH-WR target, which is exactly what the game_data β_c is meant to replace.

## Risks / honest caveats

- **Still observational, not causal.** The regression de-confounds co-occurrence + skill + play/draw,
  not everything (card sequencing, in-game decisions, opponent). It's *less* confounded than GIH-WR,
  not ground truth — temper expectations.
- **`β_c` may correlate highly with IWD** (IWD is 17lands' own lighter de-confounding). That's exactly
  what Step 0 checks; if so, we've cheaply ruled out the whole branch.
- **Collinearity** (cards co-occur) → L2 regularization required; interpret coefficients as relative.
- **Data volume / cost** — game_data is large; dev on samples, full ingest only after Step 0 passes.

## Alternative track (if not game_data)

If the value-model payoff looks marginal after Step 0, the higher-value move is the **product surface**
(deploy the model + an aggressiveness dial; the `Drafter` bundle already exists) — a different kind of
progress than chasing the last point of WR-agreement.
