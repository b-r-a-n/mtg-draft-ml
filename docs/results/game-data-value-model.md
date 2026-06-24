# game_data value model — Step 0 go/no-go (is β_c a *new* signal vs GIH-WR / IWD?)

**Question:** every WR-agreement result plateaus at ~0.29–0.31, capped by the **confounded** GIH-WR
proxy (see [README](README.md)). Before investing in a full game_data ingest, the cheap test
([game-data-plan.md](../game-data-plan.md) Step 0): does a regression on raw **per-game outcomes**
give a *less-confounded*, *genuinely different* card-value signal than the GIH-WR / IWD ratings we
already use — or does it just re-derive IWD?

**The signal.** 17lands `game_data_public.<SET>.<EVENT>` is one row per game (`won` + per-card
`deck_<card>` counts + game controls). Fit one L2-regularized logistic regression per set:

```
P(won) = σ( Σ_c β_c·deck_count_c  +  γ·on_play  +  γ·num_mulligans  +  γ·player_skill  +  b )
```

`β_c` is card *c*'s **marginal** win contribution, holding the rest of the deck + player skill +
mulligans + on-the-play fixed — a stronger de-confounder than GIH-WR (raw, fully confounded) or IWD
(controls only for *having* the card drawn, not for the rest of the deck). Every deck is ~40 cards,
so the `deck_*` columns share a constant-sum direction that only L2 identifies → **interpret β_c
relatively** (a ranking), which is exactly what we compare against the ratings.

**Setup.** One set (**DSK**), CPU, ~1 minute end-to-end. 80k-game sample, aligned to the DSK manifest
(286 cards; all 286 present in game_data). Controls deliberately **exclude `num_turns`** — turn count
is a *consequence* of the game (a mediator), so conditioning on it would bias the card coefficients;
`on_play` / mulligans / skill are pre-outcome. Reproduce:

```
uv run python scripts/game_value_gonogo.py --set DSK --sample-rows 80000 --l2 30
```

(`docs/results/game-data-value-model.json` is the machine-readable output.)

## Result — β_c is a genuinely different ranking (PROCEED)

| comparison | Spearman | Pearson | well-sampled Spearman (n=254, ≥1000 copies) |
|---|---|---|---|
| β vs **IWD** (`drawn_improvement_win_rate`) | **0.597** | 0.656 | **0.629** |
| β vs **GIH-WR** (`ever_drawn_win_rate`) | 0.718 | 0.731 | 0.743 |

**Decision: PROCEED.** Spearman(β, IWD) = **0.60 ≪ 0.95** stop-threshold. game_data is *not* a
re-derivation of IWD — it is a meaningfully distinct card-value ranking worth the Step-1 full model.

**It is signal, not noise.** The worry is that a low correlation could be an artifact of noisy
rarely-played cards. It is not: restricting to the 254 **well-sampled** cards (≥1000 deck-copies
across the games, where β is reliably estimated) the correlation barely moves (IWD 0.60→0.63), and an
**L2 sweep is flat** — across l2 ∈ {1, 3, 10, 30, 100, 300, 1000}, Spearman(β, IWD) stays **0.57–0.62**
and Spearman(β, GIH) stays **0.71–0.73**. The gap to IWD is structural, not a regularization choice.

**β tracks GIH more than IWD** (0.72 vs 0.60). Intuitive once stated: β and GIH both rank by *"decks
with this card win"*; IWD measures *"the deck wins more in the games it's drawn"* — a different
quantity that rewards swingy, high-variance cards. β keeps GIH's deck-outcome grounding but
**de-confounds across cards** (it holds the rest of the deck fixed; GIH does not).

**Controls behave** (standardized coefs): `num_mulligans` **−0.207**, player-skill bucket **+0.251**,
`on_play` **+0.101** — mulligans hurt, skill helps, on-the-play helps, all the right sign and sane
magnitudes; the fit is healthy (train acc 0.591, well above the 0.546 base rate).

## Face validity — β ranks the format correctly

**Top β** (most win-positive per marginal copy): the DSK bomb cycle — *Overlord of the Mistmoors,
Ghostly Dancers, Valgavoth's Onslaught, The Swarmweaver, Abhorrent Oculus, the Overlords* — and
premium removal (*Sheltered by Ghosts*, well-sampled at 10k copies). **Bottom β:** the dual lands
(*Razortrap Gorge, Blazemire Verge, Hushwood Verge, Etched Cornfield*) and low-impact filler — exactly
right under a "marginal value of one more copy, holding the 40-card deck fixed" reading (an extra land
displaces a spell).

## Where β and IWD disagree (well-sampled cards, ≥1000 copies)

These movers are the *point* — the cards whose value the de-confounding actually changes:

| direction | cards | story |
|---|---|---|
| **β promotes** (IWD under-rates) | Altanak the Thrice-Called (n=7427), Fear of Exposure, Fear of Impostors, Hardened Escort, Trial of Agony | solid contributors that don't *swing* games when drawn (low IWD) but whose decks win — β credits them, IWD doesn't |
| **β demotes** (IWD over-rates) | Twitching Doll (IWD +0.054 → β −0.020), Nashi Searcher in the Dark, Fanatic of the Harrowing, dual lands (Gloomlake Verge, Central Elevator) | high-variance "swingy when drawn" cards + lands — IWD rewards the draw-time swing; β says one more copy doesn't move deck win-rate |

This is the de-confounding hypothesis confirmed in miniature: **IWD rewards "swingy when drawn"; β
rewards "decks with this card win, holding the rest of the deck fixed."**

## Caveats (honest)

- **Still observational, not causal.** β de-confounds co-occurrence + skill + play/draw, *not* card
  sequencing, in-game decisions, or the opponent. Less confounded than GIH-WR, not ground truth.
- **Relative, not absolute.** The ~40-card constant-sum makes β identifiable only up to the L2 prior;
  read it as a ranking. (Lands sitting low is a feature of this framing, not a bug — but it does mean
  β is a *spell-quality* signal, not a deckbuilding land-count signal.)
- **One set, one sample.** DSK, 80k games. Step 1 should confirm stability across sets and a larger
  sample, and sanity-check the low-support tail (rare bombs like *Damnation*, n=203, get noisy/negative
  β — correctly excluded from the well-sampled movers above).

## Verdict → Step 1

**PROCEED to the Step-1 full value model.** β_c is a real, less-confounded, face-plausible card-value
signal that differs from IWD where theory says it should (≈0.6 Spearman, flat under regularization,
well-sampled-stable). It is a candidate to drop into `composite_card_quality` / the WR-softmax teacher
and—per the plan—to power an **estimated deck-WR** eval. The open question Step 1 must answer is not
"is it different" (it is) but **"is the de-confounded target the one that finally pushes WR-agreement
past the 0.29–0.31 GIH-WR ceiling."**
