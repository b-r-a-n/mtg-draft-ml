# Distillation — soft-label targets (DD-004)

**Question:** the one-hot human pick collapses a whole pack to one bit and is itself noisy (the
~0.58 top-1 ceiling is genuine drafter disagreement). Does a *softer/denser* target — a denoised
ensemble, a win-rate ranking, or a privileged-feature teacher — extract more from the **same data**,
specifically on *pick quality* (WR-agreement) rather than human-imitation (top-1)? Three KD arms,
each `(1-λ)·CE + λ·KL(teacher ‖ student)` over the pack (`training.losses.pack_distillation_kl`).

**Setup:** best recipe (content encoder + MiniLM, Set Transformer pool, in-pack CE), LOSO
BLB+OTJ+WOE+MKM → DSK (~98% novel cards), 8 epochs, batch 512, single seed (students seed 0,
ensemble teachers seeds 1–3). **RunPod RTX 2000 Ada GPU** (`scripts/pod_distill_suite.sh`). WR
metrics over DSK packs with ≥2 rated cards (17lands `ever_drawn_win_rate`), same eval as
[phase3-winrate.md](phase3-winrate.md). The **human reference (WR-agree 0.2981, avg-pick-WR 0.5509)
matches Phase 3's 0.2990/0.5509** — a consistency anchor that the eval is the same.

Reproduce: `uv run python scripts/pod_distill.py` / `pod_wr_distill.py` / `pod_leaky_distill.py`
(`--device cuda --epochs 8`). λ=0.5; temp 2.0 (ensemble, leaky) / 1.0 (WR); `wr_tau=1.0`,
`adv_tau=0.03`, `topk=0`.

## 1. Ensemble-of-seeds — control (denoise the same labels, no new information)

| policy | top1 | top3 | top5 | MTPD | **WR-agree** | avg-pick-WR |
|---|---|---|---|---|---|---|
| baseline (CE) | 0.5760 | 0.8935 | 0.9709 | 0.855 | **0.2597** | 0.5471 |
| distilled (CE+KD) | 0.5787 | 0.8929 | 0.9698 | 0.858 | 0.2538 | 0.5468 |
| ensemble (target) | 0.5789 | 0.8917 | 0.9698 | 0.858 | 0.2517 | 0.5465 |

**No help — WR-agreement slightly *regresses*** (0.2597 → 0.2538). top1 +0.0027 is within noise;
ranking (top3/5, MTPD) is flat. Averaging independent seeds pulls the target toward **human
consensus**, so the student gets marginally more human-like but slightly *worse* at taking the
high-WR card. Denoising the same labels adds nothing — corroborating the label-noise ceiling from
the *objective* side (it was the control arm).

## 2. WR-softmax — dense win-rate target vs the existing scalar method ⭐ headline

| policy | **WR-agree** | avg-pick-WR | top1 | top5 | MTPD |
|---|---|---|---|---|---|
| baseline (CE) | 0.2597 | 0.5471 | 0.5760 | 0.9709 | 0.855 |
| **WR-KD (dense target)** | **0.2885** | **0.5493** | 0.5550 | 0.9580 | 0.980 |
| advantage-weight (scalar) | 0.2725 | 0.5485 | 0.5564 | 0.9602 | 0.960 |
| *human reference* | 0.2981 | 0.5509 | — | — | — |

**Reshaping the target beats reweighting the example.** Making the target the softmax-over-win-rate
ranking (`WRSoftmaxTeacher`) lifts WR-agreement **+0.0288 over baseline** and **+0.0159 over the
scalar IWD-advantage method** (`pick_advantage_weights`), closing **~75% of the human WR-agreement
gap** (0.2597→0.2885 vs human 0.2981) where the scalar method closes ~33%. avg-pick-WR also reaches
the highest of the three, near human. The cost is top1 (0.5760 → 0.5550): it deviates from the human
pick *on purpose*, more than the scalar method — the intended "good, not just human" trade. For
comparison, Phase 3's gentle win-weighting moved WR-agreement only +0.008; the **dense target is a
~3.5× larger WR-agreement gain** than win-weighting on the same holdout.

## 3. Leaky-feature → release-day — strong teacher, failed transfer (this config)

| policy | **WR-agree** | avg-pick-WR | top1 | top5 | MTPD |
|---|---|---|---|---|---|
| baseline (CE) | 0.2597 | 0.5471 | 0.5760 | 0.9709 | 0.855 |
| distilled (CE+KD, no WR) | 0.2590 | 0.5470 | 0.5805 | 0.9706 | 0.846 |
| teacher (leaky, sees WR) | **0.3303** | 0.5540 | 0.5998 | 0.9736 | 0.798 |
| *human reference* | 0.2981 | 0.5509 | — | — | — |

**The leaky teacher is excellent; its knowledge did not transfer.** Giving the model per-card win
rate as an *input feature* (`augment_with_winrate`) makes a genuinely strong drafter — WR-agreement
0.3303, **above human** (0.2981), and the best top1 (0.5998). But the release-day student (no WR
feature) distilled from it is **flat vs baseline** (0.2590 ≈ 0.2597; ~0% of the teacher gap closed).
The teacher's edge lives in its *argmax*; its temperature-2 softened distribution — trained on the
same human picks, just with WR as an extra feature — looks too much like the baseline's to carry the
WR signal across. The **direct WR-softmax target (#2) is the better vehicle for the same signal.**

## Findings

1. **The dense WR-softmax target is the keeper.** It is the only arm that improves winning-pick
   selection *beyond the existing method*: +0.0159 WR-agreement over scalar advantage-weighting, and
   the largest WR-agreement gain of any "good, not just human" lever tried so far (vs Phase 3
   win-weighting / aux-WR). Recommended for the "good, not just human" objective.
2. **Ensemble-of-seeds does nothing** (slight WR regression). Denoising the same labels with no new
   information can't beat the label-noise ceiling — exactly the predicted control outcome, and an
   independent confirmation of the ceiling from the objective side.
3. **Privileged features make a strong teacher but don't distill as-is.** The win-rate-augmented
   teacher beats humans on WR-agreement, yet a temp-2 / λ=0.5 distillation transfers ~none of it.
   Either the soft policy doesn't express the WR knowledge, or it needs a sharper target — a temp/λ/
   top-k sweep is the open question. For now, inject win rate as a *target* (#2), not via a teacher's
   softened policy.
4. **Top-1 is the wrong scoreboard here, as expected.** Every arm sits ~0.55–0.58 top-1 (the noise
   ceiling); the signal is entirely in WR-agreement / avg-pick-WR. Reading top-1 alone would call all
   three "flat" and miss the WR-softmax win.

## Interpretation / caveats

- **Single seed, sampled data, one holdout (DSK), 8 epochs.** Treat sub-0.01 deltas as suggestive.
  The ensemble's −0.006 WR-agreement and leaky's −0.0006 are within run-to-run noise — read them as
  "no effect," not "harmful."
- **GIH-WR is a confounded proxy** (favors controlling decks; entangles archetype/skill). WR-agreement
  is *directional*, not ground truth — that humans hit only ~0.30 confirms the highest-GIH-WR card is
  often not the right pick. So "above human WR-agreement" (the leaky teacher) means "takes the high-WR
  card more often," not provably "drafts better." Same caveat as Phase 3.
- **Raw result JSONs were not saved off-pod** (the pod was deleted before the `scp` completed — a
  zsh word-split slip on the SSH options, the gotcha in [../runpod-runbook.md](../runpod-runbook.md)).
  These tables are transcribed verbatim from the pod run log; re-run `pod_distill_suite.sh` to
  regenerate the `data/*.json` if needed.

## Next

- **Tune WR-softmax** (`--wr-tau`, `--distill-lambda`, `--distill-topk`) — it's the lever that works;
  find its frontier on the WR-agreement ↔ top1 trade.
- **Unlock leaky transfer** — sweep temp (→1.0) / λ (↑) / top-k on the leaky teacher; if it transfers,
  it's a release-day-deployable upgrade.
- **Compose** — `CompositeTeacher([wr, leaky])` to combine the dense WR target with the leaky policy.
