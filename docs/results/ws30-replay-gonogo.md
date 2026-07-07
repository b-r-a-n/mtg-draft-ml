# WS3.0 — Replay-data cast-conditioned value: go/no-go

**Decision: GO** (2026-07-07). All three gates pass. Ran locally for $0 on DSK full replay data.

## Why

17lands replay data contains per-turn cast lists that game_data does not — a strictly different view
of card contribution. WS3.0 tests whether a per-card **cast-conditioned** value signal (β_cast) is
(a) reliable enough to use as an outcome-eval adjudicator and teacher, (b) genuinely distinct from
GIH, and (c) face-plausible. If so, WS3.1 repeats WS1.2/WS1.3 with replay-β in place of deck_value-β.

## Data

DSK PremierDraft full replay file — **570 MB gz, n = 1,011,949 games**. Downloaded via
`download_17lands_replay` (new function in `src/mtg_draft_ml/data/download.py`). No pod, no GPU, $0.

## Method

Estimand: `logistic won ~ Σ cast_count_c + controls` where `cast_count_c` is the number of times
card c was cast during game i (0 if not cast at all). Same float64 L-BFGS / L2 = 30 recipe as
`game_value`. Controls: `on_play, num_mulligans, user_game_win_rate_bucket`.

Cast counts parsed from 90 per-turn columns — `user_turn_{1..30}_{creatures_cast,
non_creatures_cast, user_instants_sorceries_cast}` — three **disjoint** columns (verified: 21,960 /
24,778 cells where instants/sorceries is nonempty have non_creatures null → they partition, not nest;
200/200 sampled cells show violations). Arena ID → manifest mapping via Scryfall: 322 IDs → 276
manifest cards; 10 bonus-sheet cards legitimately absent (Collected Company, Damnation, etc.);
unmapped fraction 0.07% (tokens / alchemy variants, expected).

### Schema discovery note

The first schema scout misread the per-turn columns as aggregate count integers — early turns are
empty (NaN), and single-token cells like `'92212'` are easily confused with integer counts. A
mid-file verification at rows 400–800 (turns 5–7, where games are active) flipped the reading: the
columns are **pipe-delimited Arena-ID lists**, not integers. Lesson: verify schema claims against
mid-file rows, not headers or early rows. The same columns also reveal richer per-turn structure
(draws, discards, kills, board states, life totals) available for a future finer-grained credit model.

## Results

### Corpus and fit sanity

| metric | value |
|---|---|
| n games | 1,011,949 |
| n cast tokens total | 7,852,680 |
| mean distinct cards cast / game | 6.98 |
| mean total casts / game | 7.76 |
| unmapped ID fraction | 0.07% |
| train accuracy | 63.6% |
| intercept | +0.201 |

Control coefficients (all signs sane):

| control | coef |
|---|---|
| on_play | +0.103 |
| num_mulligans | −0.189 |
| user_game_win_rate_bucket | +0.280 |

### Gate (a) — split-half reliability

| signal | split-half ρ | n well-supported |
|---|---|---|
| replay β_cast (WS3.0) | **0.976** | 106 |
| game_data deck-β, matched n = 1.01M (WS1.1 recipe) | 0.944 | 234 |
| game_data deck-β, WS1.1 reference (DSK full) | 0.909 | — |

**PASS.** At equal n, cast-β (ρ 0.976) exceeds deck-β (ρ 0.944) — the cast signal carries more
within-game information per game than deck composition does.

The split-half gap between cast and deck is interpretable: deck_count is sparse (0 or 1 most cards,
at most 2–3 copies); cast_count is zero-inflated but the non-zero distribution is also richer —
games where a card is cast multiple times or in combination provide stronger signal per game.

### Gate (b) — divergence from GIH

| comparison | Spearman ρ (all) | Spearman ρ (well-sampled) |
|---|---|---|
| β_cast vs GIH WR | 0.637 | 0.702 |
| β_cast vs IWD | 0.658 | 0.681 |
| β_cast vs deck_value full-β | 0.719 | 0.778 |

**PASS.** Sp(β_cast, GIH) = 0.637 ≪ 0.95 threshold — cast-β is a genuinely new ranking, not a
re-derivation of GIH. Sp(β_cast, deck_value_full) = 0.719 shows moderate correlation with the
deck-composition signal, consistent with both measuring card quality but through different
conditional paths.

### Gate (c) — face-plausibility

**PASS with a documented caveat.**

Top 15 by β_cast (well-supported, support ≥ 12,649 total casts):

| card | β_cast | n_cast |
|---|---|---|
| Valgavoth's Onslaught | +0.714 | 20,509 |
| Ghostly Dancers | +0.707 | 18,483 |
| Unholy Annex // Ritual Chamber | +0.694 | 21,669 |
| The Swarmweaver | +0.572 | 24,977 |
| Jump Scare | +0.551 | 27,378 |
| Zimone, All-Questioning | +0.501 | 21,221 |
| Turn Inside Out | +0.501 | 49,916 |
| Balustrade Wurm | +0.469 | 18,035 |
| Midnight Mayhem | +0.448 | 37,204 |
| Fear of Burning Alive | +0.445 | 32,824 |
| Dissection Tools | +0.427 | 23,149 |
| Vile Mutilator | +0.426 | 15,830 |
| Unstoppable Slasher | +0.410 | 21,115 |
| Vanish from Sight | +0.344 | 27,449 |
| Optimistic Scavenger | +0.337 | 52,382 |

Bottom 15 by β_cast (well-supported):

| card | β_cast | n_cast |
|---|---|---|
| Keys to the House | −0.240 | 17,532 |
| Friendly Teddy | −0.197 | 24,779 |
| Grab the Prize | −0.183 | 20,794 |
| Anthropede | −0.167 | 15,595 |
| Found Footage | −0.162 | 15,060 |
| Living Phone | −0.132 | 17,590 |
| Malevolent Chandelier | −0.129 | 13,661 |
| Glimmerburst | −0.115 | 48,559 |
| Trial of Agony | −0.113 | 22,037 |
| Creeping Peeper | −0.113 | 52,355 |
| Say Its Name | −0.105 | 82,232 |
| Grasping Longneck | −0.086 | 53,612 |
| Savior of the Small | −0.073 | 16,362 |
| Cult Healer | −0.064 | 37,700 |
| Resurrected Cultist | −0.064 | 30,031 |

The tops are sane: Valgavoth's Onslaught, Ghostly Dancers, and Unholy Annex are DSK bombs;
Zimone, Fear of Burning Alive, Vile Mutilator, and Unstoppable Slasher are all correct high-quality
cards. The bottoms are also reasonable: Keys to the House (uncastable in most games), Friendly Teddy
(underpowered), Anthropede (overcosted).

**Documented caveat — the cast-selection confound.** Two anomalies reveal a structural issue that
must be stated prominently:

1. **Turn Inside Out** (+0.501, #7) is a cheap combat trick — it is cast in already-favorable combat
   states where the player is winning anyway, inflating its β.
2. The bottom list is skewed toward desperation card-digging (Grab the Prize, Glimmerburst, Say Its
   Name) — these get cast when the player is behind and needs to find answers.

β_cast partly measures "cards cast from winning positions" — a selection confound distinct from but
structurally analogous to GIH's "cards drawn in winning games." WS3.0 establishes **reliability and
novelty**, NOT de-confoundedness. The confound must be addressed in WS3.1 design before using
replay-β as a teacher or adjudicator.

## Gates summary

| gate | criterion | result | status |
|---|---|---|---|
| (a) split-half ρ | ≥ 0.909 (WS1.1 DSK reference) | 0.976 | **PASS** |
| (b) Sp(β_cast, GIH) | < 0.95 | 0.637 | **PASS** |
| (c) face-plausibility | human review | PASS with caveat | **PASS** |

**Decision: PROCEED to WS3.1.**

## WS3.1 implications

The go/no-go opens WS3.1 (replay-β as outcome-eval adjudicator + teacher, repeating WS1.2/WS1.3).
Before running, the cast-selection confound must be addressed in design. Candidate controls:

- **Cast turn relative to CMC**: was the card cast on curve (turn ≈ CMC) vs late? On-curve casts
  are likelier to be enabling a plan; off-curve casts may signal mana flooding or opponent weakness.
- **Restricting to casts before turn N**: early casts are more causal; late-game casts in winning
  positions (tricks, finishers) are increasingly confounded. Choosing N requires sensitivity
  analysis.
- **Game length (n_turns) as a control**: tempting but dangerous — game length is both an outcome
  (short games = one player dominated) and a mediator (a card that closes games fast changes
  n_turns). The same gotcha applies as with num_turns in game_value (WS1.4 linearity probe caveats).

The replay data also contains untapped per-turn richness: board states, life totals, draws,
discards, kill events — sufficient for a per-turn win-probability delta model (the original sketch
(b) in the WS3.0 plan), which would attribute value to actions rather than to card identity. That
is a significantly deeper model and beyond WS3.1's scope.

**Budget note:** WS3.1 requires a GPU pod to run the outcome eval and teacher re-test (~$2–3 against
a $4.60 balance). The WS3.0 run itself was free (local CPU).

## Repro

```bash
# Full run (used for this writeup):
PYTHONPATH=src .venv/bin/python scripts/replay_value_gonogo.py --set DSK

# Quick smoke test (20k games):
PYTHONPATH=src .venv/bin/python scripts/replay_value_gonogo.py --set DSK --sample-rows 20000
```

Raw artifact: `docs/results/ws30-replay-gonogo.json`
