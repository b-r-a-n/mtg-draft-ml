"""Cold-start LLM-teacher distillation (DD-004 pattern #4).

A brand-new set has zero 17lands data on release day, yet that's exactly when a draft assistant
is most valuable. The content encoder already scores the *cards* (DD-001), but the set's *format*
signal — which cards are actually strong here — only accrues as human data arrives. An LLM that
reads oracle text + stats can provide that signal provisionally.

The whole subpackage rests on one idea: a teacher produces a per-card quality rating in the **same
JSON shape 17lands card ratings use** (`[{"name", "llm_quality"}]`). That makes the LLM a drop-in
stand-in for the missing ratings file — it flows through `eval.winrate.align_winrates`, the
`WRMeter`, and the inference quality-blend dial unchanged. The experiment then asks one question:
how much of the achievable cold-start benefit does the LLM teacher capture before real data exists?
"""
from .ensemble import CompositeTeacher, EnsembleModel, EnsembleTeacher, run_ensemble_distill
from .leaky import augment_with_winrate, run_leaky_distill
from .wr import WRSoftmaxTeacher, run_wr_distill
from .teacher import (
    AnthropicTeacher,
    HeuristicTeacher,
    Teacher,
    build_teacher_ratings,
    card_brief,
)

__all__ = [
    # cold-start (DD-004 #4): teacher as a stand-in ratings file for a brand-new set
    "Teacher",
    "HeuristicTeacher",
    "AnthropicTeacher",
    "card_brief",
    "build_teacher_ratings",
    # soft-label ranking distillation (DD-004 #1): denser target, same data
    "EnsembleTeacher",
    "EnsembleModel",
    "run_ensemble_distill",
    # win-rate soft target (DD-004 #1): good-not-just-human as a dense target, not a scalar reweight
    "WRSoftmaxTeacher",
    "run_wr_distill",
    # leaky-feature -> release-day (DD-004 #3): teacher sees win rate, student doesn't
    "augment_with_winrate",
    "run_leaky_distill",
    # compose KD signals (denoise + good-not-just-human + leaky) into one target
    "CompositeTeacher",
]
