# Next work — game_data value model (the less-confounded signal)

**Status:** scoped, not started. The one remaining lever with real headroom after the
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

**Step 0 — go/no-go (≈1 day, one set, cheap).** Add `download_17lands_game` (mirror the draft
downloader, different URL) + a slim game→deck-matrix preprocessor. Fit the regression on a *sample*
of DSK games. **Compare `β_c` to GIH-WR and IWD** (Spearman + which cards move most). Decision:
- `β_c` ≈ IWD (corr ≳ 0.95) → game_data adds little de-confounding beyond IWD we already use → **stop**.
- `β_c` meaningfully differs (demotes mediocre-but-ride-along cards, promotes genuinely impactful
  ones) → **proceed**. This gates the expensive ingest on evidence the new signal is actually new.

**Step 1 — full value model (≈2–3 days).** Ingest game_data for the train sets (sample mode first —
full is GBs/set), fit per-set value models, emit a per-card adjusted-value file shaped like a ratings
JSON (so it's a drop-in field). Validate face-plausibility (bombs/removal high, filler low) + stability.

**Step 2 — use it (≈1–2 days).** Plug the adjusted value into the composite-WR target and re-run the
best config (good players + composite). Evaluate on **estimated deck-WR** *and* WR-agreement-vs-adjusted.
Headline: does the de-confounded target push past the 0.29–0.31 GIH-WR ceiling?

## Prereq (cheap, do first): multi-seed confirmation

Every recent delta (+0.0057 good-vs-rand, +0.0053 composite, +0.0068 more-sets) is ~0.005 — within
single-seed noise; we've trusted the *consistent ordering*, not the magnitude. Before investing in
game_data, **rotate 3–5 seeds** on the current best config (good players + composite) to confirm the
orderings and get error bars. ~30 min on a pod; bankable either way. `scripts/rotate_seeds.py` exists.

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
