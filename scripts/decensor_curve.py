"""De-censoring experiment: does deck CURVE/CASTABILITY have MARGINAL predictive power for WINNING,
once we (a) widen the curve distribution with skill-diverse / bad-player decks and (b) control for the
deck_value card-power sum (and player skill)?

This is the EMPIRICAL counterpart to the mechanistic castability model (src/mtg_draft_ml/eval/
castability.py). castability.py predicts P(deck functions) from pure mana math with NO outcome data;
this script tests that prediction against REAL win/loss. They are the two ends of one bridge.

Why this is needed (the censoring problem, stated in probe_deck_outcome.py / game_value.py):
game_data only contains decks humans ACTUALLY BUILT — all curve-sane and castable. The bad-curve
region has no examples, so the won~composition regression can't see curve. "deck-WR is ~linear in
composition / curve doesn't matter" was therefore never really tested. Bad players build worse decks
and have real outcomes, so they are the negative examples that make curve MEASURABLE.

THE POWER-CONTROLLED TEST (the whole point): bad players lose for many reasons (mostly weak cards).
To avoid confounding curve with general badness we test the curve descriptor's MARGINAL effect on
`won` AFTER controlling for:
  - power   = CROSS-FIT (out-of-fold) deck_value sum over the built deck (the card-power control).
              MUST be out-of-fold: deck_value is itself fit on `won`, so an in-sample power launders the
              outcome into the control and biases the test toward "curve doesn't matter". See crossfit_power.
  - skill   = user_game_win_rate_bucket                       (the player's general strength)
  - on_play, num_mulligans                                    (pre-outcome game controls)
Two readouts:
  (1) nested logistic regression: add the curve descriptor to won ~ power+skill+play+mull and report
      its coefficient, the held-out log-loss / AUC lift, and a likelihood-ratio test.
  (2) power-banded stratification: within narrow power x skill cells, compare win rate across curve
      buckets (a model-free version of the same marginal test).

CURVE / CASTABILITY DESCRIPTORS (per built deck), all computed from the RAW CSV (which has basic
lands -> the real land count) + Scryfall (mana_cost -> colored pips):
  - avg_cmc            : mean cmc of the nonland deck cards
  - n_lands            : total lands in the deck (basics + nonbasics)
  - pip_concentration  : Herfindahl over colored-pip shares (1 = mono, ->0 = many colors): a splash /
                         mana-greed proxy
  - castability_score  : eval/castability.py P(deck functions) — the MECHANISTIC descriptor; this is
                         the column that, if it has marginal lift, validates the castability model.

DATA SLICE (all LOCAL, no pod):
  - data/raw/game.<SET>.PremierDraft.sample{60000|150000}.csv   (deck_*/sideboard_* incl. basics,
    won, on_play, num_mulligans, user_game_win_rate_bucket, rank, user_n_games_bucket)
  - data/hf/scryfall/<set>.json                                  (mana_cost per card)
  - webapp/data/<SET>.cards.json                                 (deck_value per card -> power sum)
  - manifest under data/hf/manifests/<SET>.PremierDraft*.json    (name -> index; optional, only used
    to reuse a cached npz; this script reads the CSV directly so the manifest is not required)

INTERPRETATION (state up front, before looking):
  (a) curve descriptor has MARGINAL lift after power+skill (coef significant, AUC up, banded win-rate
      gap across curve buckets > noise): curve really does matter for winning once de-censored. This
      VALIDATES castability.py against real outcomes AND reopens the "pick-time curve is valueless"
      verdict (it was bounded out on censored data; the bound is no longer safe).
  (b) STILL flat with bad / skill-diverse decks in (coef ~0, no AUC lift, banded gap ~0): curve is
      dominated by card power even when we look at worse decks. The mechanistic castability model is a
      build-time nicety (it still helps the *deckbuilder* choose among a fixed pool, per play_prob's
      +0.049), but it is not a hidden lever on win rate, and the curve-blind pick verdict stands.

CAVEAT THIS SCRIPT IS HONEST ABOUT: built decks are STILL curve-self-censored even for bad players —
bad players mostly play weaker CARDS, not broken curves (their avg_cmc spread is barely wider). So a
(b) result on built decks does NOT prove curve is irrelevant in the truly broken region; it proves
humans rarely enter that region. The COUNTERFACTUAL arm (--counterfactual) probes the broken region
directly by re-building each pool with a deliberately curve-blind sort and asking whether the LINEAR
power model (which is all we can score an unplayed deck with) thinks it lost anything — a model-vs-
model bound, not a real-outcome test, and flagged as such.

    uv run python scripts/decensor_curve.py --set DSK
    uv run python scripts/decensor_curve.py --set DSK --counterfactual
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

from mtg_draft_ml.eval.castability import castability_score

COLORS = ("W", "U", "B", "R", "G")
BASICS = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G", "Wastes": "C"}


# --------------------------------------------------------------------------------------------------
# card metadata: deck_value (power), cmc, type, mana_cost (pips) keyed by card NAME
# --------------------------------------------------------------------------------------------------
def _name_index(records: list[dict]) -> dict[str, dict]:
    """Scryfall by name, also indexing the front face of split/MDFC `A // B` names (mirrors
    cards.content_table._name_index so lookups match the rest of the codebase)."""
    idx: dict[str, dict] = {}
    for c in records:
        if c.get("name"):
            idx.setdefault(c["name"], c)
            if " // " in c["name"]:
                idx.setdefault(c["name"].split(" // ", 1)[0], c)
    return idx


def load_card_meta(set_code: str, cards_json: str | None, scryfall_path: str | None) -> dict:
    """Per-card-NAME metadata: {name: {deck_value, cmc, t, mana_cost, ci}}. deck_value/cmc/t/ci from
    webapp/data/<SET>.cards.json (the same fields play_prob/_card_arrays use); mana_cost from
    Scryfall (cards.json has no mana_cost). Missing deck_value stays None (treated as 0 power)."""
    cards_json = cards_json or f"webapp/data/{set_code}.cards.json"
    scryfall_path = scryfall_path or f"data/hf/scryfall/{set_code.lower()}.json"
    cj = json.load(open(cards_json))["cards"]
    sc = _name_index(json.load(open(scryfall_path)))
    meta = {}
    for c in cj:
        rec = sc.get(c["name"]) or {}
        meta[c["name"]] = {
            "deck_value": c.get("deck_value"),
            "cmc": c.get("cmc") or 0.0,
            "t": c.get("t"),
            "ci": c.get("ci") or "",
            "mana_cost": rec.get("mana_cost"),
        }
    return meta


# --------------------------------------------------------------------------------------------------
# stream the raw CSV into per-deck rows: power, curve descriptors, controls, won
# --------------------------------------------------------------------------------------------------
def _resolve_csv(set_code: str, raw_dir: str) -> str:
    cands = sorted(glob.glob(str(pathlib.Path(raw_dir) / f"game.{set_code}.*.csv")),
                   key=lambda p: -int(p.split("sample")[-1].split(".")[0]))  # biggest sample first
    if not cands:
        raise SystemExit(f"no raw game CSV for {set_code} under {raw_dir}")
    return cands[0]


def build_deck_table(set_code: str, raw_dir: str, meta: dict, max_decks: int,
                     counterfactual: bool, n_lands_cf: int = 17) -> dict:
    """One row per (deduplicated) built deck. Returns arrays:
      power [n]            sum of deck_value over the built deck (card-power; the linear model's input)
      avg_cmc, n_lands, pip_conc, cast [n]    curve/castability descriptors of the REAL built deck
      C [n,3]              controls: on_play, num_mulligans, user_game_win_rate_bucket
      y [n]               won
      cf_power [n]         (if counterfactual) power of a curve-BLIND rebuild of the same pool
                          (deck+sideboard), built by greedy deck_value to the same spell count
    """
    import pandas as pd

    csv = _resolve_csv(set_code, raw_dir)
    head = pd.read_csv(csv, nrows=0).columns.tolist()
    deck_cols = [c for c in head if c.startswith("deck_")]
    side_cols = ["sideboard_" + c[len("deck_"):] for c in deck_cols]
    side_cols = [c for c in side_cols if c in head]
    names = [c[len("deck_"):] for c in deck_cols]

    # per-deck-column vectors aligned to deck_cols order
    dv = np.array([(meta.get(nm, {}).get("deck_value") or 0.0) for nm in names], dtype=np.float64)
    cmc = np.array([(meta.get(nm, {}).get("cmc") or 0.0) for nm in names], dtype=np.float64)
    is_land = np.array([(nm in BASICS) or (meta.get(nm, {}).get("t") == "land") for nm in names])
    is_nonland = ~is_land
    # colored pips per card name (for pip concentration), W/U/B/R/G
    from mtg_draft_ml.eval.castability import parse_pips
    pip = np.zeros((len(names), 5))
    for j, nm in enumerate(names):
        p = parse_pips(meta.get(nm, {}).get("mana_cost"))
        pip[j] = [p.get(c, 0) for c in COLORS]

    usecols = ["draft_id", "won", "on_play", "num_mulligans",
               "user_game_win_rate_bucket"] + deck_cols + side_cols
    usecols = [c for c in usecols if c in head]

    rows_power, rows_cmc, rows_land, rows_pip, rows_cast, rows_C, rows_y, rows_cf = (
        [], [], [], [], [], [], [], [])
    rows_X = []   # full deck-count vectors, retained so power can be CROSS-FIT (leak-free) downstream
    seen_drafts: set = set()
    for chunk in pd.read_csv(csv, usecols=usecols, chunksize=20_000):
        if "draft_id" in chunk:
            chunk = chunk[~chunk["draft_id"].isin(seen_drafts)].drop_duplicates("draft_id")
            seen_drafts.update(chunk["draft_id"].tolist())
        D = chunk[deck_cols].fillna(0).to_numpy(np.float64)        # [m, n_deckcols]
        S = (chunk[side_cols].fillna(0).to_numpy(np.float64)
             if side_cols else np.zeros_like(D))
        for r in range(len(chunk)):
            d = D[r]
            nl_mask = d * is_nonland
            n_nl = nl_mask.sum()
            if n_nl < 10:        # not a real built deck
                continue
            rows_X.append(d.astype(np.float32))               # retain composition for cross-fit power
            rows_power.append((d * dv).sum())
            rows_cmc.append((nl_mask * cmc).sum() / n_nl)
            rows_land.append((d * is_land).sum())
            pip_tot = (nl_mask[:, None] * pip).sum(0)              # total pips per color in deck
            s = pip_tot.sum()
            rows_pip.append(float((pip_tot / s) @ (pip_tot / s)) if s > 0 else 0.0)  # Herfindahl
            # mechanistic castability over the deck's NONLAND spells, with the REAL land count
            deck_idx = [j for j in np.flatnonzero(nl_mask > 0)]
            cards_meta = [meta.get(names[j], {"cmc": cmc[j], "t": ("land" if is_land[j] else "spell"),
                                              "mana_cost": None}) for j in range(len(names))]
            n_lands = int((d * is_land).sum()) or n_lands_cf
            rows_cast.append(castability_score(deck_idx, cards_meta, n_lands=n_lands))
            ch = chunk.iloc[r]
            rows_C.append([ch.get("on_play", 0.0), ch.get("num_mulligans", 0.0),
                           ch.get("user_game_win_rate_bucket", np.nan)])
            rows_y.append(ch["won"])
            if counterfactual:
                # rebuild the full pool (deck+side) ignoring curve: greedy top-deck_value nonland
                # spells to the same spell count, then the SAME power sum tells us if the linear
                # power model thinks the curve-blind deck is any weaker.
                pool = d + S[r]
                pool_nl = pool * is_nonland
                # expand to a per-copy list of (deck_value), take best `n_nl` by value
                vals = []
                for j in np.flatnonzero(pool_nl > 0):
                    vals += [dv[j]] * int(pool_nl[j])
                vals = sorted(vals, reverse=True)[:int(n_nl)]
                rows_cf.append(float(np.sum(vals)))
        if len(rows_y) >= max_decks:
            break

    out = {
        "power_insample": np.array(rows_power), "avg_cmc": np.array(rows_cmc),
        "n_lands": np.array(rows_land), "pip_conc": np.array(rows_pip),
        "cast": np.array(rows_cast), "C": np.array(rows_C, dtype=np.float64),
        "y": np.array(rows_y, dtype=np.float64), "X": np.array(rows_X, dtype=np.float32),
    }
    if counterfactual:
        out["cf_power"] = np.array(rows_cf)
    # impute the rare missing skill control with column mean (matches game_preprocess)
    c = out["C"]
    bad = np.isnan(c[:, 2])
    if bad.any():
        c[bad, 2] = np.nanmean(c[:, 2])
    # THE GATING FIX: leak-free power. deck_value (cards.json) was fit on the SAME `won` these rows
    # carry, so an in-sample power = sum(deck_value) over-absorbs win-correlated variance (incl. any
    # curve signal aligned with card identities) and biases the marginal test toward (b)/flat. Use a
    # cross-fit out-of-fold deck_value instead.
    out["power"] = crossfit_power(out["X"], out["y"], out["C"])
    return out


def crossfit_power(X, y, C, k: int = 5, l2: float = 30.0, seed: int = 0) -> "np.ndarray":
    """Leak-free per-deck card-power: K-fold OUT-OF-FOLD deck_value. For each fold, refit the
    won~composition logistic (game_value.fit_card_values) on the OTHER folds and score this fold's
    power = X @ beta. Each deck's power thus comes from a deck_value model that never saw that deck,
    so it cannot launder this row's `won` into the control."""
    from mtg_draft_ml.eval.game_value import fit_card_values

    Xf = np.asarray(X, dtype=np.float64)
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    power = np.zeros(n)
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(k) if g != f])
        beta = fit_card_values(Xf[tr], y[tr], C[tr], l2=l2)["beta"]
        power[te] = Xf[te] @ beta
    return power


# --------------------------------------------------------------------------------------------------
# (1) nested logistic regression: marginal lift of each curve descriptor after power+skill+controls
# --------------------------------------------------------------------------------------------------
def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)


def marginal_test(t: dict, descriptors=("avg_cmc", "n_lands", "pip_conc", "cast"), seed=0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    y = t["y"]
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    # base controls: power (card strength), skill, on_play, mulligans
    base = np.column_stack([_z(t["power"]), _z(t["C"][:, 2]), t["C"][:, 0], _z(t["C"][:, 1])])

    def fit_eval(Xtr, Xte):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        return m, log_loss(y[te], p), roc_auc_score(y[te], p)

    m0, ll0, auc0 = fit_eval(base[tr], base[te])
    out = {"n": int(n), "win_rate": float(y.mean()), "base_logloss": ll0, "base_auc": auc0,
           "descriptors": {}}
    for name in descriptors:
        full = np.column_stack([base, _z(t[name])])
        m1, ll1, auc1 = fit_eval(full[tr], full[te])
        # likelihood-ratio chi-square on TRAIN (1 dof): 2*(LL_full - LL_base) on the training fit
        ptr0 = np.clip(m0.predict_proba(base[tr])[:, 1], 1e-9, 1 - 1e-9)
        ptr1 = np.clip(m1.predict_proba(full[tr])[:, 1], 1e-9, 1 - 1e-9)
        bce = lambda p: -(y[tr] * np.log(p) + (1 - y[tr]) * np.log(1 - p)).sum()
        lr_chi2 = 2 * (bce(ptr0) - bce(ptr1))
        out["descriptors"][name] = {
            "coef_after_controls": float(m1.coef_[0, -1]),   # std-scaled log-odds per 1 SD
            "logloss_lift": ll0 - ll1,                       # >0 = held-out improvement
            "auc_lift": auc1 - auc0,
            "lr_chi2_1dof": float(lr_chi2),                  # >3.84 ~ p<0.05 (single dof)
            "corr_with_power": float(np.corrcoef(t[name], t["power"])[0, 1]),
            "descriptor_std": float(np.std(t[name])),
        }
    return out


# --------------------------------------------------------------------------------------------------
# (2) power-banded stratification: win rate across curve buckets within power x skill cells
# --------------------------------------------------------------------------------------------------
def banded_stratify(t: dict, descriptor="cast", n_power=8, n_skill=3, n_curve=3) -> dict:
    """Win-rate gap across curve buckets within power x skill cells. Within each cell the win-rate is
    RESIDUALIZED on power (a linear y~power fit) before the high-vs-low-curve comparison, so the
    residual-power confound the wide bins leave inside a cell (high-castability decks carry slightly
    more deck_value) is removed; we also report the leftover power gap so it's auditable. Uses the
    cross-fit (leak-free) power. n_power=8 (finer than the old 5) further shrinks within-cell power."""
    power, skill, curve, y = t["power"], t["C"][:, 2], t[descriptor], t["y"]
    pe = np.quantile(power, np.linspace(0, 1, n_power + 1))
    se = np.quantile(skill, np.linspace(0, 1, n_skill + 1))
    pb = np.clip(np.digitize(power, pe[1:-1]), 0, n_power - 1)
    sb = np.clip(np.digitize(skill, se[1:-1]), 0, n_skill - 1)
    gaps, raw_gaps, pwr_gaps, weights = [], [], [], []
    for pi in range(n_power):
        for si in range(n_skill):
            cell = (pb == pi) & (sb == si)
            if cell.sum() < 200:
                continue
            cv, yc, pc = curve[cell], y[cell], power[cell]
            lo = cv <= np.quantile(cv, 1.0 / n_curve)
            hi = cv >= np.quantile(cv, 1.0 - 1.0 / n_curve)
            if lo.sum() < 30 or hi.sum() < 30:
                continue
            # residualize win on power within the cell, then compare high-curve vs low-curve residuals
            b = np.polyfit(pc, yc, 1)
            resid = yc - np.polyval(b, pc)
            gaps.append(resid[hi].mean() - resid[lo].mean())
            raw_gaps.append(yc[hi].mean() - yc[lo].mean())          # un-residualized (old metric)
            pwr_gaps.append(pc[hi].mean() - pc[lo].mean())          # leftover power confound (audit)
            weights.append(min(lo.sum(), hi.sum()))
    gaps, raw_gaps = np.array(gaps), np.array(raw_gaps)
    pwr_gaps, weights = np.array(pwr_gaps), np.array(weights, float)
    agg = lambda a: float(np.average(a, weights=weights)) if len(a) else float("nan")
    return {
        "descriptor": descriptor,
        "n_cells": int(len(gaps)),
        "mean_gap": agg(gaps),                                      # power-RESIDUALIZED (the real one)
        "mean_gap_raw": agg(raw_gaps),                              # before residualizing (old, inflated)
        "mean_residual_power_gap": agg(pwr_gaps),                   # power still differing hi vs lo (~0 ideally)
        "gap_std": float(gaps.std()) if len(gaps) else float("nan"),
        "frac_cells_positive": float((gaps > 0).mean()) if len(gaps) else float("nan"),
        "note": ("power-residualized high-minus-low-castability win-rate gap over power x skill cells; "
                 "mean_gap is the de-confounded effect, mean_gap_raw the old inflated one, "
                 "mean_residual_power_gap the leftover power (should be ~0)."),
    }


# --------------------------------------------------------------------------------------------------
# counterfactual (model-vs-model, NOT real outcome): curve-blind rebuild vs the real built deck
# --------------------------------------------------------------------------------------------------
def counterfactual_summary(t: dict) -> dict:
    """How much power does the LINEAR model think a curve-BLIND rebuild of the same pool gains/loses
    vs the human-built (curve-sane) deck? If ~0, the power model is INDIFFERENT to curve (it can't
    see it) — which is exactly the censoring: an unplayed bad-curve deck has no real `won`, so this
    is a BOUND on what the power model could even register, not evidence about real win rate."""
    # cf_power = greedy-best-by-deck_value over the deck+sideboard SUPERSET; the real deck is one subset
    # of that superset, so cf_power >= power BY CONSTRUCTION. The old "frac_curveblind_higher = 1.00" was
    # therefore a tautology, not a finding — dropped. Only the MAGNITUDE (how much power a curve-blind
    # drafter would gain by ignoring castability) carries information; report its distribution.
    d = t["cf_power"] - t["power_insample"]   # vs in-sample deck_value (the metric the rebuild optimizes)
    qs = np.quantile(d, [0.25, 0.5, 0.75])
    return {
        "mean_power_delta_curveblind_minus_real": float(d.mean()),
        "median_power_delta": float(qs[1]), "p25": float(qs[0]), "p75": float(qs[2]),
        "note": ("curve-blind greedy-deck_value rebuild (over deck+sideboard) vs the real deck, scored "
                 "by the LINEAR power sum. cf_power >= power by construction (greedy over a superset), so "
                 "only the magnitude is informative: it is how much card-power a curve-blind drafter "
                 "could grab that the power model CANNOT penalize for being unbuildable. A crude "
                 "model-vs-model BOUND on the censoring, NOT a real-outcome test (the rebuild was never "
                 "played)."),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_code", default="DSK")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cards-json", default=None)
    ap.add_argument("--scryfall", default=None)
    ap.add_argument("--max-decks", type=int, default=60000)
    ap.add_argument("--counterfactual", action="store_true",
                    help="also run the model-vs-model curve-blind rebuild bound")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    meta = load_card_meta(a.set_code, a.cards_json, a.scryfall)
    t = build_deck_table(a.set_code, a.raw_dir, meta, a.max_decks, a.counterfactual)
    print(f"{a.set_code}: {len(t['y'])} built decks  win_rate={t['y'].mean():.3f}  "
          f"avg_cmc={t['avg_cmc'].mean():.2f}(sd {t['avg_cmc'].std():.2f})  "
          f"cast={t['cast'].mean():.3f}(sd {t['cast'].std():.3f})")

    mt = marginal_test(t)
    print("\n[1] MARGINAL TEST (after power + skill + on_play + mulligans), held-out 80/20:")
    print(f"    base: logloss={mt['base_logloss']:.4f}  AUC={mt['base_auc']:.4f}")
    print(f"    {'descriptor':<14}{'coef/SD':>9}{'AUC lift':>10}{'logloss lift':>14}"
          f"{'LR chi2':>9}{'corr(power)':>12}")
    for name, d in mt["descriptors"].items():
        print(f"    {name:<14}{d['coef_after_controls']:>9.4f}{d['auc_lift']:>10.4f}"
              f"{d['logloss_lift']:>14.5f}{d['lr_chi2_1dof']:>9.1f}{d['corr_with_power']:>12.3f}")

    print("\n[2] POWER x SKILL-BANDED win-rate gap across curve buckets (power-residualized):")
    bands = {}
    for desc in ("cast", "avg_cmc"):
        b = banded_stratify(t, descriptor=desc)
        bands[desc] = b
        print(f"    {desc:<9} mean_gap={b['mean_gap']:+.4f} (raw {b['mean_gap_raw']:+.4f}, "
              f"resid-power {b['mean_residual_power_gap']:+.4f})  cells={b['n_cells']}  "
              f"frac+={b['frac_cells_positive']:.2f}")

    result = {"set": a.set_code, "marginal": mt, "banded": bands}
    if a.counterfactual:
        cf = counterfactual_summary(t)
        result["counterfactual"] = cf
        print("\n[CF] curve-blind rebuild (MODEL-vs-MODEL bound, not real outcome):")
        print(f"     power delta (curve-blind - real): mean {cf['mean_power_delta_curveblind_minus_real']:+.4f}"
              f"  median {cf['median_power_delta']:+.4f}  (cf>=power by construction; magnitude only)")

    # verdict helper for the mechanistic descriptor (`cast`) — the castability-model validation
    cast = mt["descriptors"]["cast"]
    band = bands["cast"]
    decensored = (cast["lr_chi2_1dof"] > 3.84 and cast["auc_lift"] > 0.001
                  and band["frac_cells_positive"] > 0.6 and abs(band["mean_gap"]) > 0.005)
    print("\nVERDICT (castability_score marginal value):",
          "(a) CURVE MATTERS once de-censored -> validates castability.py, reopens pick-time verdict"
          if decensored else
          "(b) STILL FLAT -> curve dominated by card power; castability.py is a build-time nicety only")
    print("     (built decks are self-censored even for bad players; see --counterfactual for the "
          "broken-region bound)")

    if a.out:
        json.dump(result, open(a.out, "w"), indent=2, default=float)
        print(f"\nwrote {a.out}")
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
