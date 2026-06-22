# Probe: is the ~0.58 ceiling irreducible human disagreement?

**Thesis under test:** three levers (more data, more capacity, sequence/inputs) all plateaued at
~0.58 held-out top-1, suggesting ~0.58 is a *task-noise floor* — i.e., humans themselves disagree on
the "right" pick, so no model can predict the human choice much better. This probes that directly.

**Method.** Exact pack contexts almost never repeat (a 15-card pack from a ~270-card set), so we
can't measure modal-pick fraction on identical packs. Instead we measure **pairwise preference
disagreement**: for every pair of cards that co-occur in packs, how often do drafters take the same
one? (Data: BLB 60k-pick sample. A pair "votes" for whichever card was taken when both were present.)

## Findings

### 1. Humans genuinely disagree — and disagreement explodes mid-pack
Pairwise human agreement (freq-weighted; 1.0 = always same choice, 0.5 = coinflip):

| region | pairwise agreement | share of decisions near-coinflip (<0.6) |
|---|---|---|
| early (picks 1–3) | **0.851** | 3% |
| mid-pack (picks 4–10) | **0.675** | **31%** |

Early picks: humans strongly agree (clear bombs). Mid-pack: only ~68% head-to-head agreement, with
~a third of matchups near coin-flips. And these are *pairwise* numbers — on a full ~10-card mid-pack
pack the modal-pick fraction is necessarily **lower** than 0.675, landing right in the ~0.45–0.55
range.

### 2. The model's error profile mirrors human disagreement
Our model is strong on early picks and weakest mid-pack (sequence model mid-pack top-1 ≈ 0.45) —
**exactly where humans themselves coin-flip.** The accuracy curve tracks the human-agreement curve.

### 3. The model is genuinely contextual (not just popularity)
A context-blind "popularity-favorite" model (always take the globally most-picked card in the pack)
scores only **0.41** top-1 — far below our content model (~0.57 LOSO / ~0.62 in-set). So the model is
doing real contextual work (pool synergy, signals), not regurgitating card tiers; the remaining gap
to 1.0 is not "missing popularity signal."

## Conclusion

**The thesis holds, directionally and strongly.** Much of the 1.0 − 0.58 gap is *irreducible human
inconsistency*, concentrated in mid-pack where even head-to-head agreement falls to ~0.68 (and full
multiway modal fraction lower still). The model is strong precisely where humans agree (early) and
plateaus precisely where they don't (mid). Combined with the three-lever plateau, this is convincing
evidence we are near the achievable ceiling for *predicting the human pick*.

**Honest caveats:** (a) exact contexts don't repeat, so this is a pairwise proxy, not the exact Bayes
top-1 ceiling; (b) one set (BLB); (c) late-pick pairs were too sparse (≥30 votes) to report. The
*shape* (agreement collapses mid-pack, model errors concentrate there) is the robust result.

**Implication (reinforces the project conclusion):** raising *human-pick* top-1 further is largely
chasing noise. The lever with real headroom is the **objective** — optimizing for win-rate (the
pick-time quality blend, especially with the less-confounded IWD target) lets the bot make
*better-than-human* picks even though it can't predict humans better. That's the right goal for a
drafting bot, and it's where effort should go.

---

## CORRECTION / EXTENSION: the context-blind probe overstated irreducibility

The pairwise probe above is **context-blind** — it lumps together drafters in different archetypes
facing the same pack. A red-aggro and a blue-control drafter *should* pick differently; that's pool
context, not noise. Conditioning on the pool changes the picture a lot.

**Pool-conditioned agreement (the right test).** Among mid-pack cross-color contested decisions where
the drafter's pool *clearly leans* toward one card's colors, the human takes the **pool-favored card
89.3%** of the time — vs 67.5% unconditioned, vs 50% if colors were irrelevant. Humans are highly
consistent *given their pool*.

**How much does pool context explain?** A simple pool-color-aware popularity heuristic vs context-blind:

| model | overall top-1 | mid-pack |
|---|---|---|
| context-blind popularity | 0.412 | 0.315 |
| + pool-color awareness | 0.504 | 0.425 |
| our content model | ~0.62 in-set | ~0.45 |

Pool *colors alone* add ~+9 pts overall / +11 mid-pack. Our model (0.62) sits above the color
heuristic, so its pool encoder already uses color **and** more (synergy/curve/quality).

**Revised conclusion.** The ~0.58 plateau is **not** pure irreducible noise — much of the apparent
mid-pack disagreement is pool-explainable structure, and the model captures a large part of it. The
plateau means the model is near *its* ceiling under the current framing, but the task has more
exploitable structure than the context-blind probe implied. **Open, decisive question:** does the
model exploit pool context *as well as humans* (who follow pool-lean ~0.89), or is it leaving
pool-conditioning signal on the table? That requires measuring the **trained model's** pool-follow
rate on the same color-lean decisions — the next experiment. If the model lags humans there, the
real headroom lever is **better pool-conditioning** (explicit archetype/color-commitment signal),
not more data/capacity/sequence (all already shown flat).
