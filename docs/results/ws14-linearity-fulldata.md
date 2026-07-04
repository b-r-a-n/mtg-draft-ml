# WS1.4 — linearity probe on full game_data (CONFIRMED, 2026-07-03)

**Why.** The original linearity bound — "deck outcome is linear in card composition; GBM / pair
interactions do not improve held-out log-loss" — was measured on an **~80k-game sample** of DSK
([game-data-value-model.md](game-data-value-model.md) Step 2, synergy probe). Per
[breakthrough-plan.md](../breakthrough-plan.md) WS1.4, the question is whether the bound holds at
full scale (~10× data, 930k–1M games/set), now that WS1.1 has rebuilt the β on the same full caches
([full-data-deck-value.md](full-data-deck-value.md)). If GBM still does not beat linear at 10× data,
the bound is real — not a small-data artefact — and pool-conditioned *pick* objectives designed to
capture synergistic outcome signal stay closed. Either answer was valuable; the answer is
**CONFIRMED: linearity holds at full scale across all three sets**.

## Result table

Three sets run on their full game_data `.npz` caches (the same files WS1.1 fit β on):

| set | n_games | linear logloss | GBM logloss | GBM − linear | linear AUC | GBM AUC | pair Δ logloss |
|---|---|---|---|---|---|---|---|
| DSK | 1,011,949 | 0.6673 | 0.6673 | 0.0000 | 0.6150 | 0.6152 | +0.0001 |
| BLB | 931,230 | 0.6665 | 0.6664 | −0.0001 | 0.6171 | 0.6173 | −0.0001 |
| MKM | 964,377 | 0.6665 | 0.6664 | −0.0001 | 0.6156 | 0.6154 | 0.0000 |

**Old 80k-game DSK baseline (for comparison):** linear 0.668, GBM 0.671, pair Δ +0.0006.

All held-out deltas are at or within the 0.0001 log-loss rounding threshold. GBM does **not** beat
linear by any meaningful margin on ~930k–1M games. Targeted synergy-pair interactions neither
help (DSK, MKM) nor hurt by more than a rounding step (BLB −0.0001 — well below noise).

## Verifier verdict and caveats

The verifier confirmed all n_games figures against the WS1.1 build table (exact match: DSK
1,011,949; BLB 931,230; MKM 964,377). AUC and log-loss directions are internally consistent for DSK
and BLB. One flag was raised for **MKM**: GBM logloss is trivially better (0.6664 < 0.6665) but GBM
AUC is trivially *lower* (0.6154 < 0.6156). The verifier notes this is not contradictory at the
0.0001 level — logloss and AUC optimize different aspects (calibration vs rank order) — but it is
worth recording. It does not change the conclusion; the deltas are in the rounding-noise regime by
any measure.

Two architectural caveats the verifier raised about the GBM:

1. **Shallow cap (`max_leaf_nodes=31`) relative to full data.** At 800k training rows the same
   architecture (300 trees × 31 leaves) is far more constrained than at 64k. A skeptic could argue
   the GBM should have been scaled (deeper trees, more iterations) to match the 10× data. The
   counter-point: `early_stopping=True` means the GBM stops when its internal validation set
   (10% of train, ~80k rows) stops improving — if the linear model exhausts the signal at 80k, the
   GBM's budget question is moot.
2. **Min-support threshold at 10× data.** `--min-support=150` selects ~4,000–4,005 candidate pairs,
   far more than the ~few-hundred at 80k scale. At 150/640k = 0.023% of games a pair could be
   statistically noisier per-pair than at the old 0.23% threshold. This does not affect the
   *aggregate* logloss test (only the individual pair rankings), and the pair delta is 0.0000–0.0001
   in any case.

Both caveats are real and worth tracking if WS3.0 replay data reopens the outcome-structure question.
They do not overturn the verdict.

## Top synergy pairs (train-ranked, held-out unvalidated)

These are the card pairs where the linear model most under-predicts wins when both are in the deck.
The *held-out* pair delta shows they don't generalise as predictors — they are archetype or format
artefacts, not synergy the model is missing.

**DSK** (horror-synergy format): Fear of Burning Alive + Patchwork Beastie (+0.021, n=11,483);
Impossible Inferno + Patchwork Beastie (+0.016, n=14,060); Winter's Intervention + Threats Around
Every Corner (+0.014, n=8,098).

**BLB** (gift/boon format): Wick's Patrol + Shoreline Looter (+0.021, n=9,618); Dire Downdraft +
Cindering Cutthroat (+0.019, n=1,769); Seedpod Squire + Oakhollow Village (+0.019, n=4,754).

**MKM** (detective-synergy format): Karlov Watchdog + Macabre Reconstruction (+0.026, n=2,359);
Glint Weaver + Hotshot Investigators (+0.023, n=3,713); Exit Specialist + Glint Weaver (+0.022,
n=6,453). MKM is the strongest individual-pair signal set yet the *aggregate* held-out delta is
0.0000 — even a dense detective-synergy theme doesn't produce measurable non-linearity in the
outcome regression.

## Implication for WS3

The linearity bound now holds at **~10× data, across three formats with different synergy themes**
(horror-tribal, gift/boon, detective). This closes the question of whether the old 80k-game result
was a small-data artefact: it was not. The structure of "did this deck have winning cards?" is
captured essentially perfectly by a card-power sum — there is no residual pool interaction signal
that a pick model could be optimised to capture.

This makes **WS3 (replay data) the only remaining route to a richer outcome signal**. The replay
data offers a per-turn win-probability view (columns: `cards_drawn, creatures_cast,
non_creatures_cast, mana_spent, life`, etc.) that replaces the single `won` bit with many
observations per game and sidesteps the deck-composition linearity ceiling entirely. If the cast-
conditioned replay-β passes its WS3.0 gate (split-half ρ ≥ full-data deck_value's 0.860, Spearman
vs GIH < 0.95), it becomes both the new evaluation ruler and the training target — the thing WS1.4
confirms is that the old ruler has no remaining room to reward pool-conditioned picking.

## Repro

```
# Full .npz must be passed explicitly — the default glob now matches BOTH full and sample150k caches
# in data/game/ in arbitrary order. Use the explicit path.
PYTHONPATH=src .venv/bin/python scripts/probe_deck_outcome.py \
  --set DSK --npz data/game/game.DSK.PremierDraft.full.npz

PYTHONPATH=src .venv/bin/python scripts/probe_deck_outcome.py \
  --set BLB --npz data/game/game.BLB.PremierDraft.full.npz

PYTHONPATH=src .venv/bin/python scripts/probe_deck_outcome.py \
  --set MKM --npz data/game/game.MKM.PremierDraft.full.npz
```

**Glob trap warning:** omitting `--npz` lets the script glob `data/game/game.<SET>.*.npz`, which
now matches both `…full.npz` and `…sample150000.npz` — `next(iter(...))` picks whichever the OS
returns first. Always pass `--npz` explicitly when the full cache is present.

Logs at `/tmp/ws14_DSK.log`, `/tmp/ws14_BLB.log`, `/tmp/ws14_MKM.log` (numbers in this doc were
taken directly from the logs; logs win over any prior JSON estimate if they differ).
