# Good players, not the average drafter (skill-filtered training)

**Question:** the ~0.58 top-1 ceiling is human *disagreement*, and the WR-agreement ceiling (~0.29)
is the average drafter's mediocre win-rate selection. Good players disagree less and take higher-WR
cards — so does training on *them* (filtering the same data to high-skill drafters) lift either
ceiling? Is any gain real label quality or just data volume? And does it depend on **scale** (model
capacity × set diversity)?

**Setup:** 17lands tags every pick with the drafter's skill (`user_game_win_rate_bucket`,
`user_n_games_bucket`, `rank`), already in our parquets (`data.dataset.skill_filter_indices`). Best
recipe (content encoder + MiniLM, Set Transformer, in-pack CE + composite-WR KD target), LOSO →
holdout DSK. Good-player bar `winrate ≥ 0.55 & n_games ≥ 50` keeps ~36% of picks. A 2×2 — {all, good}
× {CE, CE + composite-WR} — plus a **volume-matched control** (`rand`: a *random* 36% subsample, same
volume, mixed quality), scored on the **full** holdout (top-1 vs the average human) **and** the
**good-player** holdout (the fair top-1 test). **Multi-seed (4 seeds)** for error bars, across two
scales: 4 vs 7 train sets, and a small (emb256/h512/L3, ~2M) vs big (emb512/h1024/L4, ~8M) net.

Reproduce: `scripts/pod_skill.py --min-winrate 0.55 --min-games 50 --volume-control --seeds 0,1,2,3
[--train-sets …] [--emb-dim 512 --enc-hidden 1024 --enc-layers 4]`.

> **⚠ Single seed misleads here.** A single-seed run gave `good−rand WR +0.0057`, `top1-on-good
> −0.0146`, `good/comp 0.2974` — which over-stated the WR win and "refuted" the cleaner-label
> hypothesis. Multi-seed at scale reversed both. Trust the multi-seed numbers below; the deltas are
> ~0.005–0.015, so sign-consistency across seeds matters more than any single value.

## Multi-seed results (4 seeds; good−rand isolates label quality at matched volume)

| config | good/comp WR-agree | good−rand WR (base) | good−rand top1-on-good |
|---|---|---|---|
| small net, **4 sets** | 0.2877 ± 0.0087 | +0.0018 ± 0.0070 (3/4) | −0.0076 ± 0.0051 (**0/4**) |
| small net, **7 sets** | 0.2955 ± 0.0087 | +0.0049 ± 0.0058 (4/4) | −0.0018 ± 0.0032 (2/4) |
| **big net, 7 sets** | **0.2988 ± 0.0052** | **+0.0158 ± 0.0039 (4/4)** | **+0.0091 ± 0.0064 (4/4)** |
| *human (all) / (good)* | 0.2981 / 0.3147 | — | — |

(WR-agreement vs 17lands GIH-WR; `(n/4)` = seeds with a positive delta.)

## Findings

1. **The good-player lever is real, and it *grows with scale*.** good−rand WR (the clean
   quality-vs-quantity test) climbs **+0.0018 → +0.0049 → +0.0158** as we add sets then capacity, and
   goes from 3/4 → 4/4 → 4/4 seeds positive. At the largest scale a bigger net extracts ~3× more value
   from the cleaner good-player labels than the small net does. So good-player *labels* are genuinely
   better — the model just needs enough capacity + diversity to exploit them.

2. **The "cleaner label → better top-1" hypothesis is vindicated *at scale* (and refuted at small
   scale).** good−rand top1-on-good goes **−0.0076 (0/4) → −0.0018 (2/4) → +0.0091 (4/4)**. At the big
   net + 7 sets, training on good players reliably predicts good drafters *better* — the cleaner-label
   effect the small-scale run couldn't see. The earlier "refuted" was an artifact of too-small scale.

3. **Capacity does NOT raise the WR-agreement ceiling.** big good/comp (0.2988 ± 0.0052) ≈ small
   (0.2955 ± 0.0087) — a +0.0033 difference, **inside the noise.** The single-seed 0.3033 that prompted
   this test was seed luck plus the 4→7-set effect. The ~0.29–0.30 ceiling is **signal-bound (GIH-WR's
   confound), not capacity-bound** — more parameters can't extract signal the label lacks. Capacity
   helps the *levers* (amplifies good-player quality, flips top1-on-good positive), not the absolute bar.

4. **Best confirmed "good, not just human" config: big net + good players + composite-WR + 7 sets →
   WR-agreement 0.299 ± 0.005**, matching the average human (0.298), now with good-player labels
   *helping* both WR and top1-on-good. Still short of the good-human bar (0.315).

## Interpretation / caveats

- **Single seed at small scale was doubly misleading** — it over-stated the WR win *and* hid the
  top-1 benefit. Both only appear with multi-seed *and* enough capacity/diversity. Lesson: confirm
  small effects with multiple seeds, and don't conclude a lever is dead from one under-scaled run.
- **One capacity step** (small vs big), single bar (winrate ≥ 0.55). The amplification is credible
  (4/4 sign-consistency, tight stds) but not a capacity *curve*.
- **GIH-WR is still the binding ceiling.** Everything clusters at 0.29–0.30: avg human 0.298, our best
  0.299, good human 0.315. Capacity + good players sharpen the levers but cannot pass the confounded
  *target signal*. Breaking past 0.30 needs a less-confounded value signal — **game_data**
  ([../game-data-plan.md](../game-data-plan.md)).

## Next

- Bankable recipe: **big + good-players + composite-WR + 7 sets** (WR 0.299 ± 0.005).
- The remaining lever with headroom is the **game_data value model** (Phase 4) — a match-outcome card
  value less confounded than GIH-WR. Capacity, good players, richer fields, and more sets each add
  small WR gains; depth and the top-1 ceiling are dead ends.
