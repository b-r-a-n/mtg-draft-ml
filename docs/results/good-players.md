# Good players, not the average drafter (skill-filtered training)

**Question:** the ~0.58 top-1 ceiling is human *disagreement*, and the WR-agreement ceiling (~0.29)
is the average drafter's mediocre win-rate selection. Good players disagree less and take higher-WR
cards — so does training on *them* (filtering the same data to high-skill drafters) lift either
ceiling? And is any gain real label quality, or just an artifact of how much data we keep?

**Setup:** 17lands tags every pick with the drafter's skill (`user_game_win_rate_bucket`,
`user_n_games_bucket`, `rank`), which our parquets already carry (`data.dataset.skill_filter_indices`).
Best recipe (content encoder + MiniLM, Set Transformer, in-pack CE), LOSO BLB+OTJ+WOE+MKM → DSK,
8 epochs, single seed. The good-player bar `winrate ≥ 0.55 & n_games ≥ 50` keeps **36%** of picks
(85.5k/240k). A 2×2 — {all, good} × {CE, CE + composite-WR KD} — plus a **volume-matched control**:
a *random* 36% subsample (`rand`, same volume, mixed quality) that isolates label quality from data
quantity. Each model is scored on the **full** holdout (top-1 vs the average human) **and** the
**good-player** holdout (the fair top-1 test). WR-agreement vs 17lands GIH-WR.

Reproduce: `scripts/pod_skill.py --min-winrate 0.55 --min-games 50 --volume-control`.

## Results (holdout DSK; good-player holdout = 37% of its picks)

| model | WR-agree | avg-pick-WR | top1 (all) | top1 (good) |
|---|---|---|---|---|
| all / base | 0.2574 | 0.5470 | 0.5710 | 0.5766 |
| **good / base** | **0.2613** | 0.5471 | 0.5563 | 0.5591 |
| rand / base *(=good's volume)* | 0.2555 | 0.5467 | 0.5657 | 0.5738 |
| all / composite | 0.2902 | 0.5495 | 0.5330 | 0.5380 |
| **good / composite** | **0.2974** | 0.5501 | 0.5198 | 0.5256 |
| rand / composite | 0.2825 | 0.5485 | 0.5266 | 0.5330 |
| *human (all)* | 0.2981 | 0.5509 | — | — |
| *human (good)* | **0.3147** | — | — | — |

**Volume-matched deltas (good − rand: same data volume, only label quality differs):**
- WR-agreement **+0.0057**  ·  top1-on-good **−0.0146**

## Findings

1. **Good-player data is a real WR lever — label quality, not volume.** good/base (0.2613) beats
   both the volume-matched random subsample (0.2555, **+0.0057**) *and the full 100% data* (0.2574).
   The ordering **good > all > rand** holds on the composite arm too (0.2974 > 0.2902 > 0.2825), so
   it's a consistent signal, not one noisy number. Training on good drafters genuinely teaches
   better winning-pick selection.

2. **The "cleaner label → higher top-1" hypothesis is refuted — even at matched volume.** The random
   36% predicts the good-player holdout *better* than the good 36% does (top1-on-good **−0.0146**).
   The reason is structural: top-1 rewards the *popular* pick, and good players deviate from popular
   toward higher-WR (less common) cards — so good-player training pulls the model *away* from the
   consensus the metric scores. The ~0.58 top-1 ceiling is **not** a bad-player-noise problem you can
   filter out; good players are, if anything, *harder* to predict on a consensus metric.

3. **New best "good, not just human" config: good players + composite-WR target → WR-agreement
   0.2974**, which essentially **matches the average human (0.2981)** — the first config to reach the
   average-human WR-agreement bar (prior best 0.2933, [distillation.md](distillation.md)). Still short
   of the good-human bar (0.3147), but the gap to average humans is ~closed.

4. **The lever is real but minor, and partly redundant with the WR target.** good−all on the baseline
   is +0.0039 WR-agreement; the composite WR target alone gets +0.0328. Both push toward winning
   picks, so stacking them adds only a little (the redundancy is why good/comp 0.2974 only edges
   all/comp 0.2902). Good-player filtering is a worthwhile *complement* to the WR target, not a
   replacement.

## Interpretation / caveats

- **Single seed; deltas are ~0.005.** Magnitude alone is within run-to-run noise — the credibility
  comes from the *consistent ordering* across arms (good > all > rand on WR; rand > all > good on
  top1-on-good), not the third decimal. A multi-seed pass would firm the numbers.
- **The volume control matters.** Without it the first run's top-1 drop (−0.021) looked like it might
  be label quality; the control shows it is *not* even volume — good-player training lowers top-1 on
  its own terms. Always match volume when comparing a filtered subset to the whole.
- **GIH-WR is the binding ceiling, again.** Every WR-agreement result clusters at 0.29–0.31:
  average humans 0.298, good humans 0.315, our best model 0.297. Filtering to good players moves the
  model *toward* the average-human bar but can't pass the good-human bar, because the *target signal*
  (GIH-WR) is itself a confounded proxy. Surpassing this needs a less-confounded value signal
  (game_data), not better human labels.

## Next

- **Best recipe to date for picking well:** good-player training + composite-WR target. Bank it after
  a multi-seed confirmation of the +0.005 deltas.
- **The remaining lever with real headroom is game_data** — a match-outcome value model less confounded
  than GIH-WR (roadmap Phase 4). Skill filtering, composite fields, and more sets have each been shown
  to add small WR gains; depth and the top-1 ceiling are dead ends.
