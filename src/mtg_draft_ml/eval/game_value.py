"""Per-card marginal win-value from per-game outcomes (game_data Step 0 go/no-go).

See docs/game-data-plan.md and docs/results/game-data-value-model.md.

Fit a single L2-regularized logistic regression over per-game decks:

    P(won) = sigma( sum_c beta_c * deck_count_c  +  gamma . controls  +  b )

`beta_c` is card *c*'s **marginal** contribution to winning, holding the rest of the deck, player
skill, mulligans and on-the-play fixed. This is a stronger de-confounding than GIH-WR (raw, fully
confounded) or IWD (controls only for *having* the card, not for the rest of the deck).

Because every deck has ~40 cards, the `deck_*` columns are near-collinear with the intercept (they
sum to a near-constant), so L2 on the card weights is required and `beta_c` is only meaningful
*relative* to the others — exactly the ranking we compare against the ratings. We penalize only the
card weights (controls + intercept are free) and fit full-batch with L-BFGS (torch, CPU).
"""
from __future__ import annotations

import json

import numpy as np

from .winrate import align_winrates


def fit_card_values(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray | None = None,
    l2: float = 10.0,
    max_iter: int = 300,
    standardize_controls: bool = True,
) -> dict:
    """Fit L2 logistic regression won ~ decks + controls. Returns beta/control coefs + diagnostics.

    `l2` penalizes only the card weights (`0.5*l2*||beta||^2`, sklearn-style on the *summed* loss so
    the penalty is on a comparable scale to the data term); controls and intercept are unpenalized.
    Every deck sums to ~40 cards, so the card columns share a constant-sum direction that is only
    identified by this penalty — interpret `beta_c` *relatively*. The card columns are mean-centered
    for conditioning (centering shifts only the intercept, leaving slopes exact) and the bias is
    warm-started at `logit(mean y)`. Returns
    {"beta" [n_cards], "control_coef" [n_ctrl], "intercept", "loss", "n", "train_acc"}.
    """
    import torch

    # float64 throughout: the fit is tiny (a few hundred params) and L-BFGS' strong-Wolfe line
    # search overflows in float32 on the summed loss. Double is robust and costs nothing here.
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    col_mean = Xc.mean(axis=0, keepdims=True)
    Xt = torch.from_numpy(Xc - col_mean)               # centering changes only the intercept
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float64))
    n, n_cards = Xt.shape

    if C is not None and C.shape[1] > 0:
        Cn = np.asarray(C, dtype=np.float64).copy()
        if standardize_controls:                       # put controls on a common scale (free params)
            mu = Cn.mean(axis=0, keepdims=True)
            sd = Cn.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            Cn = (Cn - mu) / sd
        Ct = torch.from_numpy(np.ascontiguousarray(Cn))
        n_ctrl = Ct.shape[1]
    else:
        Ct = None
        n_ctrl = 0

    p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    beta = torch.zeros(n_cards, dtype=torch.float64, requires_grad=True)
    gamma = (torch.zeros(n_ctrl, dtype=torch.float64, requires_grad=True) if n_ctrl else None)
    bias = torch.tensor([np.log(p0 / (1 - p0))], dtype=torch.float64, requires_grad=True)
    params = [beta, bias] + ([gamma] if gamma is not None else [])
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
    # mean loss keeps gradients O(1) (a summed loss over ~80k games overflows/diverges even in
    # float64). Penalty `0.5*(l2/n)*||beta||^2` is sklearn's `0.5/C*||w||` divided by n, so `l2`
    # plays the role of sklearn's alpha = 1/C on the per-sample objective.
    pen = l2 / n
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Xt @ beta + bias
        if gamma is not None:
            logits = logits + Ct @ gamma
        loss = bce(logits, yt) + 0.5 * pen * (beta * beta).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        logits = Xt @ beta + bias
        if gamma is not None:
            logits = logits + Ct @ gamma
        loss = float(bce(logits, yt) + 0.5 * pen * (beta * beta).sum())
        train_acc = float(((logits > 0).float() == yt).float().mean())
    return {
        "beta": beta.detach().numpy().astype(np.float32),
        "control_coef": (gamma.detach().numpy().astype(np.float32)
                         if gamma is not None else np.zeros(0, np.float32)),
        "intercept": float(bias.detach()),
        "loss": loss,
        "n": int(n),
        "train_acc": train_acc,
    }


def build_value_ratings(
    beta: np.ndarray,
    support: np.ndarray,
    card_names: list[str],
    set_code: str | None = None,
    event_type: str = "PremierDraft",
    min_support: float = 1000.0,
) -> list[dict]:
    """Shape per-card beta into a 17lands-ratings-style record list (a drop-in card-value field).

    Each record is `{name, deck_value, deck_value_support}` — the same `{name, <field>}` shape the
    17lands ratings JSONs use, so `align_winrates(..., field="deck_value")` and
    `composite_card_quality` consume it unchanged (and `deck_value_support` is its shrink/confidence
    count, registered in `winrate._COUNT_FOR`). Cards below `min_support` total deck-copies get
    `deck_value=None` (beta is noisy there) so they're treated as missing, not as a real low value.
    """
    recs = []
    for i, name in enumerate(card_names):
        s = int(support[i])
        recs.append({
            "name": name,
            "set": set_code,
            "event_type": event_type,
            "deck_value": (round(float(beta[i]), 6) if s >= min_support else None),
            "deck_value_support": s,
        })
    return recs


def merged_ratings_with_value(ratings_path, gamevalue_path, out_path):
    """Write a copy of a 17lands ratings JSON with each card's `deck_value`/`deck_value_support`
    merged in by name, so `composite_card_quality` can use `deck_value` as one of its fields (it reads
    all fields from a single ratings file per set). Returns out_path."""
    import json
    import pathlib

    ratings = json.load(open(ratings_path))
    gv = {r["name"]: r for r in json.load(open(gamevalue_path))}
    for c in ratings:
        g = gv.get(c["name"])
        if g is not None:
            c["deck_value"] = g.get("deck_value")
            c["deck_value_support"] = g.get("deck_value_support")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ratings))
    return str(out_path)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho over the finite-in-both entries (rank then Pearson)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = _rankdata(a[m]); rb = _rankdata(b[m])
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy-free)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i + 1
        while j < len(sa) and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def compare_to_ratings(
    beta: np.ndarray,
    manifest_path,
    ratings_path,
    card_names: list[str] | None = None,
    support: np.ndarray | None = None,
    min_support: float = 1000.0,
    n_movers: int = 15,
) -> dict:
    """Correlate per-card beta against GIH-WR and IWD, and list the biggest rank movers vs IWD.

    `beta` is in manifest-index order. Returns Spearman/Pearson vs each field over all rated cards
    *and* over the well-sampled subset (`support >= min_support`, where `support` is each card's
    total copies across the fitted games) — the well-sampled number guards against a low-correlation
    being mere noise from rarely-played cards. Movers are restricted to the well-sampled subset so
    the example cards are trustworthy. `support=None` disables the subset (treats all as sampled).
    """
    gih = align_winrates(manifest_path, ratings_path, field="ever_drawn_win_rate")
    iwd = align_winrates(manifest_path, ratings_path, field="drawn_improvement_win_rate")
    if card_names is None:
        cards = json.load(open(manifest_path))["cards"]
        card_names = [c["name"] for c in sorted(cards, key=lambda d: d["index"])]
    card_names = list(card_names)
    well = (support >= min_support) if support is not None else np.ones(len(beta), bool)
    bw = np.where(well, beta, np.nan)

    out = {
        "spearman_gih": _rank_corr(beta, gih),
        "pearson_gih": _pearson(beta, gih),
        "spearman_iwd": _rank_corr(beta, iwd),
        "pearson_iwd": _pearson(beta, iwd),
        "spearman_gih_well": _rank_corr(bw, gih),
        "spearman_iwd_well": _rank_corr(bw, iwd),
        "n_compared_gih": int((np.isfinite(beta) & np.isfinite(gih)).sum()),
        "n_compared_iwd": int((np.isfinite(beta) & np.isfinite(iwd)).sum()),
        "n_well_sampled": int((well & np.isfinite(iwd)).sum()),
        "min_support": float(min_support),
    }

    # rank movers vs IWD (well-sampled only): positive delta = beta ranks the card HIGHER (promoted)
    m = well & np.isfinite(beta) & np.isfinite(iwd)
    idx = np.flatnonzero(m)
    br = _rankdata(beta[m]) / (m.sum() - 1)      # normalized rank in [0,1]
    ir = _rankdata(iwd[m]) / (m.sum() - 1)
    delta = br - ir
    order = np.argsort(delta)
    movers = []
    for k in [*order[:n_movers], *order[-n_movers:]]:
        gi = idx[k]
        movers.append({
            "name": card_names[gi],
            "beta": float(beta[gi]),
            "beta_rank": float(br[k]),
            "iwd": float(iwd[gi]),
            "iwd_rank": float(ir[k]),
            "gih": float(gih[gi]) if np.isfinite(gih[gi]) else None,
            "support": int(support[gi]) if support is not None else None,
            "delta_rank": float(delta[k]),
        })
    out["promoted_vs_iwd"] = sorted(movers, key=lambda d: -d["delta_rank"])[:n_movers]
    out["demoted_vs_iwd"] = sorted(movers, key=lambda d: d["delta_rank"])[:n_movers]
    return out
