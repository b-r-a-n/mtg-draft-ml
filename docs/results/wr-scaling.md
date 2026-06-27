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

## Pushing past 15 — the curve PEAKS at ~19, then turns over

Ingested 7 more sets (SNC, NEO, MID, LTR, STX, SIR, PIO → 23 total, 22 train) and extended the curve
(DSK holdout, 2 seeds):

| sets | WR-agree:GIH | per-seed |
|---|---|---|
| 15 | 0.3171 ± 0.0036 | [.314, .321] |
| **19** | **0.3260 ± 0.0051** | [.321, .331] |
| 22 | 0.3051 ± 0.0074 | [.298, .313] |

**It keeps rising to ~19 sets (peak ≈0.326), then DECLINES at 22 (−0.021, ~3σ, seed-consistent 2/2).**
So it's **not raw count** — the sets added 19→22 are **STX, SIR, PIO** (Strixhaven 2021, Shadows
*remastered*, *Pioneer Masters*): older / remaster / reprint-Masters sets with off-distribution draft
environments. Padding the corpus with low-relevance sets actively *hurts* the encoder's value sense.

**Refined recipe:** more *relevant, diverse* draft sets helps WR-agreement up to ~19; **don't pad with
remaster/Masters/old sets** — quality of the corpus matters, not just quantity. The genuine WR-agreement
peak is ≈**0.326** at ~19 recent-Standard-ish sets (vs the ~0.30 at 7 and the 0.29–0.31 GIH "ceiling").

## Curation: relevance, not count (resolves "curated-19 vs 22")

The nested-prefix curve pinned STX/SIR/PIO **last**, so "19 vs 22" conflated *which* sets with *how
many*. To separate them, trained the same recipe on **explicit** corpora (holdout DSK, 3 seeds,
`scripts/pod_corpus_curation.py`). Set composition (scryfall) flags **SIR** (2016 Innistrad *remaster* —
cards resolve to soi/emn/inr) and **PIO** (Pioneer *Masters*, 2012–19 reprints) as genuinely
off-distribution; **STX** is a real 2021 Standard expansion caught in the same ingest batch.

| arm | sets | WR-agree:GIH | vs all22 |
|---|---|---|---|
| **nested19** (−STX,SIR,PIO) | 19 | **0.3251 ± 0.0044** | +0.018 |
| drop_reprint20 (keep STX) | 20 | 0.3215 ± 0.0017 | +0.014 |
| relevance_ctrl19 (keep SIR,PIO; −FIN,EOE,TDM) | 19 | 0.3151 ± 0.0064 | +0.008 |
| all22 (full) | 22 | 0.3071 ± 0.0067 | — |
| *human* | | 0.2981 | |

- **It's relevance, not count.** The control `relevance_ctrl19` is a *19-set* corpus that **keeps**
  SIR/PIO — and scores **−0.010 below** `nested19` at *identical count* (0.315 vs 0.325). Hitting "19"
  is not the lever; *which* sets is.
- **SIR + PIO are the culprits.** Dropping just those two (`drop_reprint20`, 0.3215) recovers nearly
  all the gain; keeping them (`all22`, 0.307) is worst. The off-distribution remaster/Masters
  environments dilute the encoder's value sense.
- **STX is ~neutral** (keep 0.3215 vs drop 0.3251 — within ~1σ): a real but older Standard set neither
  clearly helps nor hurts. No free set to reclaim, but no harm either.
- **Curation recovers and *explains* the ~0.326 peak; it does not exceed it.** The lever is "exclude
  reprint/remaster/Masters sets," and 0.325–0.326 is the corpus-breadth ceiling for this recipe/holdout.
  (Replicates the 2-seed curve's 19=0.326 / 22=0.305.)

**Curated recipe:** train on the **~19 relevant expansion/draft-innovation sets**; explicitly exclude
remaster (SIR), Masters/reprint (PIO), and the oldest off-Standard sets. STX optional (marginal).

## Why more sets helps the WR axis (hypothesis)

The model is content-based; more diverse sets = broader coverage of card *types* the encoder must
value, so on an unseen set it better identifies the genuinely high-WR cards. Top-1 (matching the human
pick) is capped by human disagreement, but WR-agreement (taking the highest-WR card) has room to
improve as the encoder's value sense sharpens with corpus breadth.

## Caveats

- ~~One holdout~~ **Confirmed on 4 holdouts** (DSK/OTJ/MOM/FDN, all +0.014–0.025) — see above.
- ~~Past 15 unknown~~ **Resolved: peaks ~19, declines at 22** when padded with remaster/Masters/old
  sets (STX/SIR/PIO) — corpus *relevance* matters, not just count.
- ~~A curated ~19 may do even better~~ **Resolved: it's relevance, not count** (curation section above).
  Excluding the off-distribution sets recovers the peak (0.325) but does **not** exceed it; SIR/PIO are
  the culprits, STX is neutral. 0.325–0.326 is the corpus-breadth ceiling for this recipe/holdout.
- Single target/config; `deck_value`-target × ~19-set is the obvious compounding test.

## Implications

Corpus size is the **cleanest lever found for WR-agreement** — and it was sitting unused because the
saturation conclusion was read off the wrong metric — but it **peaks ~19 *relevant* sets (≈0.326)**,
not unboundedly. Next: (1) ✅ rotated holdout (confirmed); (2) ✅ pushed past 15 (peaks ~19); (3) ✅
`deck_value` target × 19 sets — **no compounding; corpus breadth subsumes the deck_value target**
([game-data-value-model.md](game-data-value-model.md)); (4) the outcome eval (estimated deck-WR) remains the
honest adjudicator. Recipe note: train the deployable on **as many diverse sets as available**, not 7.
