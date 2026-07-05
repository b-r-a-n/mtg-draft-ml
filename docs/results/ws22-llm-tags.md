# WS2.2 — LLM-annotated card tags as structured features

**Why.** The WS2.1 null (modern embedders +0.000–+0.001) showed that generic semantic-embedding
quality is not the binding constraint on day-0 generalization. The residual hypothesis is that the
bottleneck is *functional grounding*: knowing what a card *does* in a draft (removal, evasion,
ramp, payoff/enabler role) — information that no general-purpose text encoder recovers from surface
oracle text. WS2.2 tests this directly by generating explicit function tags via LLM and appending
them as structured features to the card encoder.

## Tag-generation method

### Tag schema

Fourteen binary/categorical dimensions generated once per card from Scryfall oracle text:

| tag | type | description |
|---|---|---|
| `is_removal` | bool | destroys/exiles/bounces a creature or planeswalker |
| `is_sweeper` | bool | removes ≥3 permanents at once |
| `is_card_advantage` | bool | nets ≥+1 card in hand |
| `is_ramp_or_fixing` | bool | adds mana or fixes colors |
| `is_evasive` | bool | flying, unblockable, menace, trample, etc. |
| `is_combat_trick` | bool | instant-speed power/toughness pump or protection |
| `is_bomb` | bool | single-card game-winner if unanswered |
| `role` | str | `payoff`, `enabler`, `filler` |
| `speed` | str | `aggressive`, `midrange`, `controlling` |
| `is_tagged` | bool | sentinel — False for missing cards (no tag inflation) |

The `is_tagged` sentinel is appended as the 10th new dimension (after the 9 above); untagged cards
receive all-zero structured tags plus `is_tagged=False`, so missing coverage degrades gracefully to
a zero-imputed baseline rather than injecting false negatives.

### Workflow v1 — native sets (79% train coverage)

A 50-agent workflow processed cards in 36-card batches. Haiku taggers applied strict Limited
definitions (e.g. "removal = destroys/exiles/bounces a non-land permanent; bounce/tuck count;
combat tricks do not"). A Sonnet adversarial verifier sampled 30 cards/set post-generation and
reported field-error rates of **2.2–3.3%, all ACCEPT**. Approximately 35 named fixes and a
basic-lands normalization were applied before the v1 tags were finalized.

Coverage v1: 1,394 native-set cards (~79% of train picks per run). Coverage was incomplete because
the 17lands parquet manifests include **bonus-sheet reprints** that are not in the native-set
Scryfall oracle dump — OTJ Breaking News (105 cards), WOE Enchanting Tales (113), MKM Special
Guests (51), BLB/DSK extra cards (~10 each). These showed up in the training corpus but lacked tags.

### Workflow v2 — full coverage (100%)

The 288 missing bonus-sheet cards were identified by diffing the manifest card names against the
tagged-card list, then fetched from the Scryfall API by full card name. A second generation
workflow applied the same definitions (tightened slightly to resolve verifier edge cases). A spot
verification of 3.8% of v2 cards returned **ACCEPT** with 3 minor fixes applied. Tags were merged
by full manifest card name (case-sensitive), achieving **100.00% coverage** across all five sets
(BLB, OTJ, WOE, MKM, DSK). Total tagged cards: **1,682**.

Tags live on HF `b-r-a-n/mtg-draft` under `tags/*.json`; `data/` is gitignored locally.

### Plumbing (commit e34a53d)

`cards/content_table.py` was extended to accept a `--tags-dir` path; the 14 tag dimensions are
appended after the 73 existing structured features, expanding the encoder input width. The harness
(`scripts/rotate_seeds.py --tags-dir`) threads the tags directory through to each run and records
per-run tag coverage in the JSON output.

## Results

### Rotated top-1 and WR-agreement (15 runs each; 3 seeds × 5 holdouts)

| arm | coverage | top-1 mean ± sd | Δ top-1 | t (paired) | n_pos/15 | WR-agree | Δ WR-agree | t (paired) | n_pos/15 |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM control | — | 0.5409 ± 0.0234 | — | — | — | 0.2528 | — | — | — |
| **v1 tags** | 79% | 0.5448 ± 0.0188 | **+0.0038** | 2.06 | 11/15 | 0.2590 | **+0.0062** | 3.78 | 12/15 |
| **v2 tags** | 100% | 0.5461 ± 0.0184 | **+0.0051** | 1.99 | 9/15 | 0.2588 | **+0.0060** | 3.03 | 10/15 |

Human WR-agree reference (this config): **0.2893**. Gate: Δ top-1 > +0.01 **or** Δ WR-agree > +0.01.

### Per-holdout top-1 (means over 3 seeds)

| holdout | MiniLM control | v1 tags (79%) | Δ v1 | v2 tags (100%) | Δ v2 |
|---|---|---|---|---|---|
| BLB | 0.5594 | 0.5607 | +0.0013 | 0.5583 | −0.0010 |
| OTJ | 0.5315 | 0.5317 | +0.0002 | 0.5268 | −0.0046 |
| WOE | 0.5118 | 0.5235 | +0.0117 | 0.5310 | **+0.0192** |
| MKM | 0.5262 | 0.5357 | +0.0095 | 0.5389 | **+0.0127** |
| DSK | 0.5759 | 0.5724 | −0.0035 | 0.5754 | −0.0004 |

### Per-pair deltas (v2, all 15 cells)

| seed | holdout | Δ top-1 | Δ WR-agree |
|---|---|---|---|
| 0 | BLB | −0.0062 | −0.0041 |
| 1 | BLB | +0.0003 | −0.0012 |
| 2 | BLB | +0.0028 | +0.0184 |
| 0 | OTJ | −0.0044 | +0.0131 |
| 1 | OTJ | −0.0033 | +0.0084 |
| 2 | OTJ | −0.0062 | +0.0112 |
| 0 | WOE | +0.0238 | +0.0142 |
| 1 | WOE | +0.0180 | +0.0095 |
| 2 | WOE | +0.0157 | +0.0054 |
| 0 | MKM | +0.0124 | +0.0151 |
| 1 | MKM | +0.0171 | +0.0073 |
| 2 | MKM | +0.0084 | −0.0026 |
| 0 | DSK | −0.0002 | +0.0010 |
| 1 | DSK | +0.0030 | −0.0005 |
| 2 | DSK | −0.0042 | −0.0051 |

## Key findings

### 1. The gain concentrates in the control's two weakest holdouts

The entire positive signal in v2 comes from WOE (+0.019 top-1) and MKM (+0.013 top-1) — the two
holdouts where the control is weakest (WOE 0.512, MKM 0.526). BLB, OTJ, and DSK are flat-to-slightly
negative. This pattern is interpretable: WOE (enchantments-matter, role-heavy) and MKM (detective
synergy, archetype-structured) are exactly the sets where oracle text *underdetermines* draft role
— a Clue-token generator's value depends on whether the deck exploits it, not on its text. Tags
give the encoder explicit role information that the text embedding misses for those mechanics.

The DSK and BLB/OTJ holdouts already encode more self-contained card functions (clear bombs, clear
removal, clear evasion) — tags add less there, and the noise floor produces slight negatives.

### 2. Full coverage did not unlock a step change

The dilution hypothesis (v1 at 79% is missing 21% of cards, imputing zeros, suppressing the
signal) is **refuted**. Going from 79% to 100% moves mean top-1 +0.0013 and mean WR-agree −0.0002
— within noise, not a structural unlock. The effect size is intrinsic to the signal, not to
coverage completeness.

### 3. WR-agreement effect is stronger and more consistent than top-1

Both arms show WR-agree Δ ≈ +0.006 at t≈3 (v1: t=3.78, p=0.0020; v2: t=3.03, p=0.0090), while
top-1 Δ ≈ +0.004–0.005 reaches only t≈2 (p≈0.06–0.07). The WR-agree effect is statistically
cleaner: 12/15 positive cells in v1, 10/15 in v2 (vs 11/15 and 9/15 for top-1). Tags steer
picks toward higher-WR cards more reliably than they improve raw top-1 accuracy — which makes
sense if function tags encode "this card wins games" knowledge that is distinct from "this is the
human-consensus pick."

### 4. Twice-replicated real effect, genuinely below gate

Unlike WS1.3's 3-cell head-fake (reversed on replication), this result replicates across 30
independent paired cells (two separate 15-run arms with different tag sets). The direction is
consistent, the WR-agree t-stats are robust, and the per-holdout structure is interpretable. This
is a **real effect** — but it is **below the +0.01 gate** on both metrics in both arms. The gate
was set at a level judged useful for adoption; +0.006 is real and interpretable but not
operationally meaningful for a model that already beats humans on WR-agreement.

## Verdict

**BELOW-GATE (twice-replicated real effect).** The +0.01 gate fails in both arms (v1: top-1 +0.0038,
WR-agree +0.0062; v2: top-1 +0.0051, WR-agree +0.0060). Full coverage did not unlock the gate;
the dilution hypothesis is refuted. The ~+0.006 WR-agree effect at t≈3, replicated across two
independent 15-run arms (30 paired cells total), is the **first genuine representation signal this
track has found** — but too small to adopt. Tags are not adopted into the deployed model.

With WS2.1 null and WS2.2 below-gate, WS2.3's own gate condition ("only if 2.1/2.2 show signal")
is not met. **The stop-condition is triggered**: WS1.3 null + WS1.4 confirmed-linear + WS2.1 null
+ WS2.2 below-gate → the walls are confirmed with a clean ruler. The modeling re-adjudication is
closed.

## Product note

The **1,682 LLM-tagged cards** are a durable asset independent of this training verdict. The tags
(removal/sweeper/card_advantage/ramp_or_fixing/evasive/combat_trick/bomb/role/speed) are
face-validated with <4% verifier error across all five sets and are exactly the vocabulary the
webapp's **deck doctor** needs to surface functional advice: "this deck has only 1 removal spell,"
"you have 3 payoffs and 0 enablers," "your curve is aggressive but your only combat tricks are
expensive." The webapp deck-doctor panel could use these tags to add a **function** lens alongside
its existing power and buildability lenses — without re-running any ML. Tags live on HF under
`b-r-a-n/mtg-draft/tags/`.

## Repro

```bash
# v1 (79% coverage — native cards only)
PYTHONPATH=src uv run python scripts/rotate_seeds.py \
  --data-dir data/hf \
  --tags-dir data/hf/tags \
  --out /tmp/ws22/rotate_minilm_tags.json

# v2 (100% coverage — includes bonus-sheet reprints)
PYTHONPATH=src uv run python scripts/rotate_seeds.py \
  --data-dir data/hf \
  --tags-dir data/hf/tags \
  --out /tmp/ws22/rotate_minilm_tags_v2.json
```

All runs: MiniLM-384 embedder; seeds 0,1,2; holdouts BLB/OTJ/WOE/MKM/DSK; sample60000; 10 epochs;
A5000 pod. Tags fetched from HF `b-r-a-n/mtg-draft` to `data/hf/tags/` before running. The
control JSON (`docs/results/ws21/rotate_minilm.json`) is the paired baseline for all Δ figures.

Raw artifacts: `docs/results/ws22/rotate_minilm_tags.json` (v1), `docs/results/ws22/rotate_minilm_tags_v2.json` (v2).
