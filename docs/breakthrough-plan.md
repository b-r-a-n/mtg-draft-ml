# Breakthrough plan — re-opening the modeling track

**Status: ACTIVE (2026-07-03). Owner: agent team. Read this whole file before starting any task.**

`docs/SUMMARY.md` concluded the modeling track is bounded out at three walls. A 2026-07-03 review
found that **two of the three walls were measured with a noisy ruler**, and that the highest-leverage
representation mitigation flagged by our own research was never run. This plan re-adjudicates the
walls cheaply first, then attacks the axes with real headroom. Work the workstreams **in order** —
WS1 is cheap and its results change whether WS3/WS4 are worth doing at all.

## The three walls, and which ones we contest

| wall | verdict | why |
|---|---|---|
| top-1 ≈0.58 (human-noise ceiling) | **ACCEPT — do not attack** | human-disagreement probe is decisive ([results/human-disagreement-probe.md]) |
| WR-agreement ≈0.326 (corpus-breadth ceiling) | **CONTEST the metric** | it's agreement with GIH-WR, a *confounded proxy*; past some point more agreement isn't better. Pivot the headline metric (see Metric policy) |
| deck-WR ≈linear in composition (no pool-conditioned signal) | **CONTEST the measurement** | the adjudicator `deck_value` has split-half reliability **ρ=0.644** (0.52–0.78) because it's fit on 150k-game samples. Part of the "interactions below the noise floor" result IS that noise |

## Ground rules (every agent, every task)

1. **Multi-seed or it didn't happen.** Seed noise floor is **±0.005** on WR-agreement
   ([results/good-players.md]). Run **≥3 seeds** for any model-training claim; report mean±sd.
   Single-seed results have misled this project twice (good-players, Step-2 ceiling break).
2. **Baselines are in this doc and in `docs/results/`** — compare against them, never re-derive.
3. **Where to run:** GPU training → RunPod per `docs/runpod-runbook.md` (`scripts/runpod_launch.sh`,
   `scripts/runpod_bootstrap.sh`). CPU-only jobs (sklearn/pandas/L-BFGS) → **use
   `scripts/runpod_bootstrap_cpu.sh`** and invoke with `PYTHONPATH=src .venv/bin/python` (NOT
   `uv run` — its auto-sync once pulled a 2–3 GB CUDA torch and hung a session for 1.5 h).
4. **Locally**, the venv is `.venv`; run `PYTHONPATH=src .venv/bin/python`.
5. **Deliverable per task:** a writeup in `docs/results/<task-slug>.md` (numbers + verdict + exact
   repro command), a row appended to `docs/results/README.md`, and the checkbox flipped **in this
   file**. Persist raw numbers to JSON next to the writeup — do not leave results only in stdout.
6. **Corpus:** the curated corpus is **nested19** = the 22-set ingest minus `STX, SIR, PIO`
   (see `ALL`/`NESTED_DROP` in `scripts/pod_corpus_curation.py`). **Never** add remaster/Masters
   sets ([results/wr-scaling.md]).
7. **Don't do:** global-negative InfoNCE, feature standardization, per-card ID embeddings,
   corpus padding past ~19 relevant sets, pick-time buildability blending, the curve-aware picker
   arm — all tested and dead ([results/README.md], [results/play-prob-pick-time.md]).

## Metric policy (applies to all workstreams)

- **Headline metric: the outcome eval** — replay held-out drafts, score each policy's pool with the
  deck-value model (`scripts/run_outcome_eval.py`, prefer `--build playprob`). It is non-circular
  (GIH-trained, deck_value-scored) and still has room to register gains.
  Current baseline: model beats humans **+0.0567 est. deck-WR (92% of drafts)** with the playprob
  builder; +0.0717 (98%) with top-23 ([results/ws12-rebaseline.md]).
- **Secondary:** WR-agreement vs GIH (human ref 0.298, model 0.3251±0.0044 on nested19) and
  rotated-LOSO top-1 (honest **0.5405±0.0229**, range 0.510–0.575 by holdout set).
- Report all three; never claim a win on the secondary metrics alone.

---

## WS1 — Fix the ruler: full-data `deck_value` (CPU-only, do FIRST)

**Why:** every downstream judgment (outcome eval, webapp dial, the "linearity" bound, the
"deck_value target subsumed" verdict) is filtered through β fit on 150k-game samples with split-half
ρ=0.644. Full 17lands game_data is 10–30× larger. The fit is CPU L-BFGS (~3 min/set at 150k);
skill controls (`on_play, num_mulligans, user_game_win_rate_bucket`) are already in
(`src/mtg_draft_ml/data/game_preprocess.py::DEFAULT_CONTROLS`). This is purely a reliability play.

- [x] **WS1.1 — Rebuild β on full game_data for the 8 webapp sets.** *(2026-07-03 — gate PASSED: mean split-half ρ 0.860 (was 0.644), all 8 sets ≥ 0.815; see [results/full-data-deck-value.md])*
  - `src/mtg_draft_ml/data/download.py::download_17lands_game` already supports full downloads
    (`sample_rows=None`). Full CSVs are small (~60–100 MB gz each, ~625 MB for all 8 — the
    "~5 GB each" fear was ~50× off); a local run takes ~20 min end-to-end, no pod needed.
    Delete each raw CSV after the npz cache is built.
  - Extend `scripts/game_value_build.py` to accept `--sample-rows 0` = full (it currently defaults
    to 150 000). Keep `--l2 30` (flat in [1,1000], [results/game-data-value-model.md]).
  - Memory check: X is [n_games × n_cards] float32; 2 M games × 300 cards ≈ 2.4 GB — fine. The fit
    itself must stay **float64 L-BFGS, mean loss** (float32/summed overflows — known gotcha).
  - Output: `gamevalue/<SET>.PremierDraft.gamevalue.json` (same shape; consumers unchanged); push to
    HF `b-r-a-n/mtg-draft` like Step 1 did. Then rebuild the webapp dial values
    (`scripts/export_webapp.py` assets) and sanity-check old↔new β Spearman per set (expect high
    but not 1.0 — report it).
  - **Gate: mean split-half ρ ≥ 0.80** (was 0.644) and reprint cross-set ρ ≥ 0.86 (was 0.86).
    Report the same table as [results/game-data-value-model.md] Step 1. If full data does NOT
    reach ρ≥0.80, report the achieved curve (ρ vs n_games) — that itself bounds WS1.3/1.4.
- [x] **WS1.2 — Re-baseline the outcome eval with the new β.** *(2026-07-03 — new baselines: playprob +0.0567/92%, top-23 +0.0717/98% (full-β, current net); June verdict survives; see [results/ws12-rebaseline.md])*
  - Rerun `scripts/run_outcome_eval.py --build playprob` (and the top-23 build) on the same
    held-out drafts as [results/outcome-eval.md], scoring with full-data β.
  - Deliverable: the new "model vs human" margins (old-β baselines: +0.049/84% playprob,
    +0.063/95% top-23). These become the numbers all later work is judged against.
- [ ] **WS1.3 — Re-test `deck_value`-as-target at 19 sets with reliable β.**
  - The "subsumed by corpus breadth" verdict ([game-data-value-model.md] compounding test) was
    reached with noisy β. Rerun `scripts/pod_game_value_target.py` with `--train-sets <nested19>`,
    holdout DSK, targets `gih,+value,value`, **3–4 seeds** (GPU pod, big net:
    `--emb-dim 512 --enc-hidden 1024 --enc-layers 4`).
  - **Adopt only if** GIH-agree or the WS1.2 outcome-eval margin improves > +0.01 beyond seed noise;
    otherwise record REFUTED and keep the GIH-composite teacher.
- [ ] **WS1.4 — Re-run the linearity probe on full data.**
  - `scripts/probe_deck_outcome.py --set DSK` (then ≥2 more sets) using the full game_data cache
    (vs the old 80k sample) and full-β.
  - Question: does GBM / pair-interaction now beat linear on held-out log-loss? If linearity still
    holds at ~10× data, the bound is real — pool-conditioned *pick* objectives stay dead and WS3
    becomes the only route to a richer outcome signal. Either answer is valuable; write it up.

---

## WS2 — Representation / day-0 generalization (the user-facing axis)

**Why:** the deployed text encoder is a **frozen 2020-era MiniLM (384d)** (`src/mtg_draft_ml/cards/
text_embed.py`, cached by `cards/content_table.py`). Our own research flagged the fix we never ran:
*"fine-tune the encoder on MTG text or on interaction data (see beeFormer)"*
([research/findings/card-representation.md]). Day-0 (new-set release) is when the webapp matters
most and the model is weakest. Harness for everything here: **rotated LOSO** (3 seeds × 5 holdouts,
`scripts/rotate_seeds.py` pattern) + WR-agreement on the holdout.

- [ ] **WS2.1 — Modern embedder swap (do first, cheapest).**
  - Make the sentence-transformer model configurable in `cards/text_embed.py` (currently MiniLM);
    add a cache key per model so embeddings don't collide.
  - Arms: MiniLM-384 (control), and 2 modern embedders (e.g. `bge-large-en-v1.5`,
    `gte-large-en-v1.5`; pick by MTEB rank at run time — must be runnable offline on the pod).
    Note: larger dim (1024) changes encoder input width; keep total params comparable
    (the capacity sweep showed params don't matter — [results/README.md] Phase 4).
  - **Gate: rotated-LOSO top-1 +0.01 over 0.5405, or holdout WR-agree +0.01, mean over 3 seeds.**
- [ ] **WS2.2 — LLM-annotated card tags as structured features.**
  - Generate once per set with a cheap LLM from Scryfall oracle text: is_removal, is_sweeper,
    is_card_advantage, is_ramp/fixing, is_evasive, is_combat_trick, is_bomb_rate, archetype_role
    (payoff/enabler/filler), expected_speed. Cache as `data/hf/tags/<SET>.tags.json`; validate
    ~30 cards/set by hand before training.
  - Append to the 73 structured features in `cards/content_table.py`. Same harness/gate as WS2.1.
  - This directly tests the open research question: do text embeddings transfer to *novel
    mechanics*, or do we need explicit function tags?
- [ ] **WS2.3 — Interaction-tuned text encoder (beeFormer-style; only if 2.1/2.2 show signal).**
  - Fine-tune the text tower (LoRA or last-2-layers) so card-embedding dot products predict
    co-pick/co-play on nested19 (positive pairs: same-deck cards from game_data `deck_` columns;
    in-pack negatives — NOT global negatives, see ground rule 7).
  - Freeze after tuning; rebuild cache; same harness/gate.
- [ ] **WS2.4 — Run the cold-start scaffold to a number.**
  - `src/mtg_draft_ml/distill/teacher.py` + `coldstart.py` + `scripts/pod_coldstart.py` exist as
    scaffolds (DD-004 #4) but were never run to a result. Define day-0 eval: ZERO target-set picks,
    model + LLM-teacher blend vs model alone, on the rotated holdouts.
  - Report even a null result — day-0 is the product's weakest moment and we have no number for
    the blend at all.

---

## WS3 — Replay-data credit assignment (moonshot; needs WS1 first)

**Why:** 17lands publishes a third view we've never used — **replay data**, per-turn actions per
game (columns confirmed: `cards_drawn, lands_played, creatures_cast, non_creatures_cast, mana_spent,
life`, user+oppo, plus `drawn_*/tutored_*/opening_hand_*` per card — schema at
`https://17lands-public.s3.amazonaws.com/analysis_data/helper_files/replay_dtypes.py`).
It attacks both remaining walls at once: a value signal that isn't GIH (different confounds) and
many observations per game instead of one `won` bit (higher effective sample size).

- [ ] **WS3.0 — Go/no-go probe (bounded: one set, one week).**
  - Download ONE set's replay data (DSK), map the schema, and fit the simplest credit model:
    per-card **cast-conditioned** value (logistic `won ~ Σ cast_count_c + controls`, mirroring
    `eval/game_value.py`) and/or a per-turn win-prob model whose deltas attribute to cards cast
    that turn.
  - **Gate (mirrors the Step-0 gonogo):** replay-β must (a) have split-half ρ ≥ full-data
    deck_value's (WS1.1 result), (b) Spearman vs GIH **< 0.95** (it's a new signal, not a
    re-derivation), and (c) be face-plausible (spot-check top/bottom 15 cards).
  - If NO-GO: write it up and stop — do not sink further time.
- [ ] **WS3.1 — (gated on 3.0) replay-β as eval + target.** Repeat WS1.2/WS1.3 with replay-β.

---

## WS4 — Skill-contrast objective (optional, small)

**Why:** good-player *filtering* ([results/good-players.md]) throws weak players' picks away; the
lever grew with scale (+0.0158 WR at big net), so the skill axis isn't exhausted — just crudely
used. `src/mtg_draft_ml/distill/skill.py` is a starting scaffold.

- [ ] **WS4.1 — DPO-style pairwise objective.** On matched (pool, pack) states where a
  top-bucket and bottom-bucket player picked differently, prefer the good pick (pairwise logistic
  on the two logits, weight by skill gap; keep the in-pack CE as the base loss).
  **Gate: holdout WR-agree +0.01 over the good+composite baseline (3 seeds), no top-1 loss > 0.01,
  and outcome-eval margin not worse.** If flat, write it up and close the axis.

---

## Sequencing and effort

| order | task | compute | expected wall-clock |
|---|---|---|---|
| 1 | WS1.1–1.2 | CPU pod, ~$1–3 | 1–2 days (mostly download) |
| 2 | WS2.1 | 1 GPU pod session | 1–2 days |
| 3 | WS1.3–1.4 | GPU ~$1 + CPU | 1 day |
| 4 | WS2.2 → 2.4 | LLM calls + GPU | 2–4 days |
| 5 | WS3.0 | CPU pod, big download | ≤1 week, hard-capped |
| 6 | WS2.3, WS4.1 | GPU | gated on earlier signal |

**Stop conditions:** if WS1.3, WS1.4, WS2.1 AND WS2.2 all come back null, the walls are confirmed
with a clean ruler — update `docs/SUMMARY.md` to say so and close the modeling track for good, this
time with the measurement caveats resolved.
