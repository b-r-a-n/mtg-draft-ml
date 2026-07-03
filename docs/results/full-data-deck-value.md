# Full-data `deck_value` — WS1.1, rebuild β on all games (gate PASSED, 2026-07-03)

**What/why.** Every downstream judgment (the outcome eval, the webapp dial, the linearity bound, the
"deck_value target subsumed" verdict) is filtered through a β fit on **150k-game samples** with
split-half reliability ρ = 0.644 — the noisy ruler [breakthrough-plan.md](../breakthrough-plan.md)
WS1 exists to fix. Per **WS1.1**, this re-runs Step 1 of
[game-data-value-model.md](game-data-value-model.md) on the **full** 17lands game_data for the 8
webapp sets (`--sample-rows 0`, same `--l2 30`, same controls, same output shape — consumers
unchanged). Purely a reliability play: 5–8× the games, no modeling change.

(`docs/results/game-value-full.json` is the machine-readable output.)

## Result — the ruler is fixed (mean split-half ρ 0.644 → 0.860)

Same table as [game-data-value-model.md](game-data-value-model.md) Step 1, plus the old 150k ρ for
direct comparison:

| set | games | cards valued | train acc | Sp(β,IWD) | split-half ρ | 150k ρ (old) |
|---|---|---|---|---|---|---|
| BLB | 931,230 | 270 | 0.595 | 0.774 | **0.890** | 0.717 |
| OTJ | 1,172,393 | 370 | 0.587 | 0.694 | **0.861** | 0.635 |
| WOE | 1,024,028 | 305 | 0.592 | 0.650 | **0.838** | 0.519 |
| MKM | 964,377 | 301 | 0.593 | 0.719 | **0.836** | 0.606 |
| DSK | 1,011,949 | 277 | 0.589 | 0.764 | **0.909** | 0.779 |
| LCI | 823,614 | 288 | 0.600 | 0.714 | **0.855** | 0.662 |
| MH3 | 767,772 | 314 | 0.598 | 0.623 | **0.815** | 0.621 |
| MOM | 1,246,850 | 346 | 0.592 | 0.736 | **0.873** | 0.611 |
| **mean** | | | | **0.709** | **0.860** | 0.644 |

**Gate verdict (PASSED, both prongs):**

- **Mean split-half ρ ≥ 0.80 → 0.860** (was 0.644). Not just the mean: *every individual set*
  clears 0.80 (min 0.815 MH3, max 0.909 DSK). WOE — the old noise poster child at 0.52 — jumps to
  0.838, confirming its instability was sample size, not the format.
- **Reprint cross-set ρ ≥ 0.86 → 0.867** (n=9 reprints well-sampled in ≥2 sets; was 0.86 on n=8),
  and the mean cross-set spread tightens **0.048 → 0.0225** — the field measures a real card
  property, now with half the per-set jitter.

**Still its own signal.** Sp(β,IWD) mean **0.709** (0.62–0.77, all ≪ the 0.95 stop-threshold). It
rises from 0.625 at 150k, as expected — both estimators de-noise toward their true rankings, which
are correlated but distinct — without approaching re-derivation.

**Face-plausible everywhere.** Tops are each format's recognized bombs (BLB *Sword of Fire and
Ice* / *Valley Questcaller*; DSK *Overlord of the Mistmoors*; WOE *Gruff Triplets*; LCI *Aclazotz* /
*Bonehoard Dracosaur*; MOM *Elesh Norn* / *Sunfall*); bottoms are utility lands and constructed-only
cards (*Doubling Season*, *Parallel Lives*) — correct under the "marginal value of one more copy"
reading.

## Downstream refresh — webapp dial + HF

The new fields were pushed to HF `b-r-a-n/mtg-draft` (`gamevalue/<SET>.PremierDraft.gamevalue.json`,
same shape as before), and the webapp dial values (`deck_value`/`q` in `webapp/data/<SET>.cards.json`)
were patched **in place** via the new `scripts/refresh_deck_values.py` — no retrain, no ONNX
re-export, other card metadata (`mc` etc.) untouched. Old↔new dial Spearman per set (high but not
1.0, as expected — the old dial carried the 150k noise):

| set | old↔new Spearman | dial non-null coverage |
|---|---|---|
| BLB | 0.912 | 253 → 265 |
| DSK | 0.928 | 259 → 272 |
| LCI | 0.895 | 272 → 283 |
| MH3 | 0.888 | 272 → 309 |
| MKM | 0.870 | 266 → 296 |
| MOM | 0.874 | 321 → 341 |
| OTJ | 0.875 | 322 → 365 |
| WOE | 0.852 | 270 → 300 |

Coverage rises everywhere because at 5–8× the games more cards clear the min-support threshold —
the dial now rates 20–40 more cards per set.

## Corrected plan assumptions (cheaper than feared)

The plan's WS1.1 sizing was wrong by ~50×: full `game_data` CSVs are **~60–100 MB gz each**
(~625 MB for all 8 sets), not "up to ~5 GB each". No CPU pod, no 200 GB disk, no one-set-at-a-time
staging — the whole thing (download + npz cache + 8 float64 L-BFGS fits + split-half + summary) ran
**locally in ~20 minutes**. The float64/mean-loss L-BFGS gotcha still applies and is honored by
`scripts/game_value_build.py`. Raw CSVs were deleted after the run; the durable per-set caches are
`data/game/game.<SET>.PremierDraft.full.npz`.

## Repro

```
PYTHONPATH=src .venv/bin/python scripts/game_value_build.py --sample-rows 0 --summary docs/results/game-value-full.json
PYTHONPATH=src .venv/bin/python -m mtg_draft_ml.data.hf push --repo b-r-a-n/mtg-draft --processed-dir data/hf
PYTHONPATH=src .venv/bin/python scripts/refresh_deck_values.py
```

(`--sample-rows 0` = full CSV, added for this task; `scripts/refresh_deck_values.py` is new.)

## Implications for WS1.2–1.4

- **WS1.2 (re-baseline the outcome eval) must point at the full-data caches**:
  `scripts/run_outcome_eval.py` **refits β from the game npz** rather than reading the gamevalue
  JSONs, so pass `--game-npz data/game/game.<SET>.PremierDraft.full.npz --l2 30` — otherwise it
  silently re-derives the old sample-β and the re-baseline is void.
- **WS1.3/WS1.4 are unblocked**: both the "subsumed by corpus breadth" verdict and the linearity
  bound can now be re-adjudicated against a β whose split-half reliability is 0.86, not 0.64. If
  linearity still holds at ~10× data, the bound is real.
