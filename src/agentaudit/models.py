"""Policy models and association tests."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import SETTINGS


def separation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Identify action categories that an activation cause predicts perfectly.

    A category observed under only one trigger level admits no finite maximum
    likelihood estimate, so it must be described by exact counts rather than
    entered into a regression. Reporting the check keeps the exclusion visible
    instead of leaving it to be inferred from a missing row.
    """
    d = df.dropna(subset=["completion", "action", "trigger", "stage"])
    if d.empty:
        return pd.DataFrame()
    triggers = sorted(d["trigger"].unique())
    rows = []
    for action in [a for a in SETTINGS.action_order if a in set(d["action"])]:
        sub = d[d["action"] == action]
        present = sorted(sub["trigger"].unique())
        rows.append({
            "action": action,
            "n": int(len(sub)),
            "triggers_present": "|".join(present),
            "triggers_absent": "|".join(t for t in triggers if t not in present),
            "separated": bool(len(present) < 2),
            "completion_min": round(float(sub["completion"].min()), 4),
            "completion_max": round(float(sub["completion"].max()), 4),
        })
    return pd.DataFrame(rows)


def policy_model(df: pd.DataFrame) -> pd.DataFrame:
    """Support level on task state and activation cause, clustered by group.

    The pre-specified four-category multinomial is not estimable on this corpus.
    Release occurs under a single trigger and redirect under two of three, so both
    are perfectly separated and their coefficients diverge rather than converge.
    The estimable contrast is therefore fitted directly as a binomial model of the
    two categories that vary across every trigger level, and the separated
    categories are reported through exact intervals elsewhere. Cluster-robust
    covariance is used because ten groups generate hundreds of decisions and
    treating those as independent would understate every interval.
    """
    d = df.dropna(subset=["completion", "action", "trigger", "stage"]).copy()
    separated = separation_report(d)
    if separated.empty:
        return pd.DataFrame()
    estimable = [r["action"] for _, r in separated.iterrows()
                 if not r["separated"] and r["n"] >= SETTINGS.min_category_n]
    if len(estimable) != 2:
        return pd.DataFrame()

    # Reference is the more supportive category, so a positive coefficient reads
    # as a move towards withdrawing support.
    reference, outcome = sorted(estimable, key=lambda a: -SETTINGS.support_level[a])
    d = d[d["action"].isin(estimable)]
    y = (d["action"] == outcome).astype(int)

    exog = pd.get_dummies(d[["trigger"]], prefix="trigger", drop_first=True, dtype=float)
    exog["completion"] = d["completion"].astype(float).to_numpy()
    exog["stage"] = d["stage"].astype(float).to_numpy()
    exog = sm.add_constant(exog.reset_index(drop=True), has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(y.reset_index(drop=True), exog, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["group"].to_numpy()})

    lo, hi = fit.conf_int()[0], fit.conf_int()[1]
    return pd.DataFrame([{
        "outcome_vs_reference": f"{outcome} vs {reference}",
        "term": str(term),
        "coef": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "ci_lo": float(lo[term]),
        "ci_hi": float(hi[term]),
        "p": float(fit.pvalues[term]),
        "n": int(len(d)),
    } for term in exog.columns])


def provenance_action_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["provenance"], df["action"])


def independence_test(table: pd.DataFrame) -> dict:
    """Chi-square where expected counts allow, Monte Carlo permutation otherwise."""
    obs = table.to_numpy(dtype=float)
    chi2, p, dof, expected = stats.chi2_contingency(obs)
    sparse = bool((expected < 5).mean() > 0.2)
    result = {"chi2": float(chi2), "dof": int(dof), "p_asymptotic": float(p), "sparse": sparse}
    if sparse:
        rng = np.random.default_rng(0)
        row, col, n = obs.sum(1), obs.sum(0), obs.sum()
        stat = lambda m: stats.chi2_contingency(m)[0]
        draws = []
        flat = np.repeat(np.arange(len(col)), col.astype(int))
        for _ in range(2000):
            rng.shuffle(flat)
            sim = np.zeros_like(obs)
            start = 0
            for i, r in enumerate(row.astype(int)):
                vals, counts = np.unique(flat[start:start + r], return_counts=True)
                sim[i, vals.astype(int)] = counts
                start += r
            draws.append(stat(sim))
        result["p_permutation"] = float((np.array(draws) >= chi2).mean())
    return result


def trigger_permutation(df: pd.DataFrame, n_rep: int, seed: int) -> dict:
    """Does activation cause change the policy once task state is held constant?

    Trigger labels are shuffled inside completion strata, so any surviving effect
    cannot be an artefact of triggers firing at different stages of progress.
    """
    d = df.dropna(subset=["completion", "trigger", "support_level"]).copy()
    if d.empty or d["trigger"].nunique() < 2:
        return {"observed": np.nan, "p": np.nan, "n": 0}
    d["stratum"] = pd.qcut(d["completion"], q=min(4, d["completion"].nunique()),
                           duplicates="drop", labels=False)
    observed = d.groupby("trigger")["support_level"].mean()
    stat = float(observed.max() - observed.min())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_rep):
        shuffled = d.groupby("stratum", group_keys=False)["trigger"].apply(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        m = d["support_level"].groupby(shuffled).mean()
        if float(m.max() - m.min()) >= stat:
            exceed += 1
    return {"observed": stat, "p": (exceed + 1) / (n_rep + 1), "n": int(len(d))}


def specificity_by_provenance(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["specificity"])
    if d.empty:
        return pd.DataFrame()
    out = d.groupby("provenance")["specificity"].agg(["count", "mean", "median", "std"])
    return out.round(4)


def kruskal_specificity(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["specificity"])
    groups = [g["specificity"].to_numpy() for _, g in d.groupby("provenance") if len(g) > 1]
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan}
    H, p = stats.kruskal(*groups)
    return {"H": float(H), "p": float(p)}
