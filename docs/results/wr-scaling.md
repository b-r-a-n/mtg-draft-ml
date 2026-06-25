# WR-agreement data-scaling curve — corpus size breaks the GIH ceiling

**Question.** Phase-4 found the data-scaling curve **saturates at ~3–4 sets** — but *on top-1*, and we
capped the training corpus at 7. The levers with real headroom (good-player labels, `deck_value`)
both showed "grows with scale" up to 7 sets and were **never tested beyond**. So: does adding sets
past 7 lift **WR-agreement** (the axis we care about), even though top-1 saturated?

**Setup.** Expanded the corpus 8 → **16 sets** (`scripts/ingest_set.py` added FDN, DFT, TDM, FIN, EOE,
ONE, BRO, DMU). Trained the established best recipe — **good players + composite-WR teacher
(GIH+IWD+ALSA), big net (emb512/h1024/L4)** — at nested corpus sizes (4/7/11/15, each a superset of
the last), holding out **DSK**, **3 seeds** each. Read WR-agreement vs **GIH** (the metric the
0.29–0.31 ceiling is defined on) and vs `deck_value`. Reproduce:
`uv run python scripts/pod_wr_scaling.py --sizes 4,7,11,15 --seeds 0,1,2`.

## Result — WR-agreement keeps climbing; the ceiling breaks from data alone

| train sets | global cards | top-1 | **WR-agree:GIH** | WR-agree:deck_value |
|---|---|---|---|---|
| 4 | 1294 | 0.5225 | 0.2902 ± 0.0047 | 0.3068 ± 0.0098 |
| 7 | 2240 | 0.5333 | 0.3027 ± 0.0039 | 0.3077 ± 0.0045 |
| 11 | 3421 | 0.5304 | 0.3070 ± 0.0051 | 0.3030 ± 0.0041 |
| **15** | 4564 | **0.5437** | **0.3203 ± 0.0050** | 0.3109 ± 0.0035 |
| *human* | | | 0.2981 | |

**Δ WR-agree:GIH vs the 7-set point:** 11 sets **+0.0043**, 15 sets **+0.0176**.

- **The corpus is NOT saturated on WR-agreement.** 7 → 15 sets lifts WR-agree:GIH **+0.018 (≈3.5σ**,
  seed std ~0.005, 3 seeds) — **0.303 → 0.320, a clean break of the 0.29–0.31 GIH ceiling, from more
  data alone.** The Phase-4 "saturates at 3–4 sets" conclusion was **top-1-specific**; it does not hold
  for the win-rate axis.
- **This is the cleanest ceiling-break found.** It beats the `deck_value` *target* (Step 2: GIH lift
  +0.007, within noise, 3/4 seeds) — here corpus size gives **+0.018 at 3.5σ**. The lever that moves
  the binding metric is **data breadth**, more than the de-confounded target.
- **top-1 isn't fully saturated either** — it creeps 0.533 → 0.544 (7 → 15), smaller than the WR gain
  but real (consistent with diminishing-but-nonzero returns; Phase-4 measured 4→7 only).
- **`deck_value`-agreement stays flat (~0.31)** — expected: the *target* here is the GIH-composite, so
  GIH-agreement is what improves; `deck_value`-agreement is incidental. (Compounding the deck_value
  *target* with 15 sets is untested — see below.)

## Confirmed across holdouts (not DSK-specific)

DSK is an *easy* holdout (Phase-4), so the 7→15 slope was re-run on three more holdouts (2 seeds each):

| holdout | 7-set WR:GIH | 15-set WR:GIH | **Δ (7→15)** | human |
|---|---|---|---|---|
| DSK | 0.3027 | 0.3203 | **+0.018** | 0.298 |
| OTJ | 0.2974 | 0.3221 | **+0.025** | 0.285 |
| MOM | 0.2793 | 0.2935 | **+0.014** | — |
| FDN | 0.2796 | 0.2997 | **+0.020** | 0.292 |

**4/4 holdouts positive, mean +0.019.** The corpus-scaling lift on WR-agreement is **general, not a DSK
artifact** — and at 15 sets the model meets-or-beats the human WR-agreement on every holdout measured
(e.g. DSK 0.320 > 0.298, OTJ 0.322 > 0.285, FDN 0.300 > 0.292). (`scripts/pod_wr_scaling.py --holdout
<SET> --sizes 7,15`.)

## Why more sets helps the WR axis (hypothesis)

The model is content-based; more diverse sets = broader coverage of card *types* the encoder must
value, so on an unseen set it better identifies the genuinely high-WR cards. Top-1 (matching the human
pick) is capped by human disagreement, but WR-agreement (taking the highest-WR card) has room to
improve as the encoder's value sense sharpens with corpus breadth.

## Caveats

- ~~One holdout~~ **Confirmed on 4 holdouts** (DSK/OTJ/MOM/FDN, all +0.014–0.025) — see above.
- **Diminishing returns past 15 unknown** — the curve is still rising at 15 (our full corpus); whether
  it continues needs more sets (17lands has ~30+; ingest is cheap via `ingest_set.py`).
- Single target/config; `deck_value`-target × 15-set is the obvious compounding test.

## Implications

Corpus size is the **cleanest lever found for WR-agreement** — and it was sitting unused because the
saturation conclusion was read off the wrong metric. Next: (1) rotate the holdout to confirm; (2)
push past 15 sets to find the real WR-agreement plateau; (3) re-run the **`deck_value` target at 15
sets** to see if the two levers compound; (4) the outcome eval (estimated deck-WR) remains the
honest adjudicator. Recipe note: train the deployable on **as many diverse sets as available**, not 7.
