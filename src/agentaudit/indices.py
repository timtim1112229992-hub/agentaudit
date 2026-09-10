"""Contingency and fading expressed as separate indices.

Keeping the two apart is the point of the exercise. A system can score highly on
one and at zero on the other, and reporting a single adaptivity figure hides that.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .config import SETTINGS


def action_census(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["action"].value_counts()
    order = [a for a in SETTINGS.action_order if a in counts.index]
    order += [a for a in counts.index if a not in order]
    out = pd.DataFrame({"n": counts.reindex(order).fillna(0).astype(int)})
    out["percent"] = (out["n"] / out["n"].sum() * 100).round(2)
    return out


def contingency_index(df: pd.DataFrame) -> dict:
    """Ordinal slope of support level on completion at decision time.

    A negative slope is the expected direction: the more complete the work, the
    less support the agent should supply.
    """
    d = df.dropna(subset=["completion", "support_level"])
    if d["support_level"].nunique() < 2 or d["completion"].nunique() < 2:
        return {"slope": np.nan, "p": np.nan, "n": int(len(d)), "note": "insufficient variation"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = OrderedModel(d["support_level"].astype(int),
                                 d[["completion"]].astype(float), distr="logit").fit(disp=False)
            return {"slope": float(model.params["completion"]),
                    "p": float(model.pvalues["completion"]), "n": int(len(d)), "note": "ordinal logit"}
        except Exception:                       # fall back to a rank measure
            rho = d["completion"].corr(d["support_level"], method="spearman")
            return {"slope": float(rho), "p": np.nan, "n": int(len(d)), "note": "spearman fallback"}


def fading_index(df: pd.DataFrame) -> dict:
    """Slope of release probability on position within a group's decision sequence.

    A positive slope is what a fading account predicts. A flat slope means support
    is never withdrawn, however responsive it may be moment to moment.
    """
    d = df.dropna(subset=["seq_frac"]).copy()
    d["is_release"] = (d["action"] == "release").astype(int)
    if d["is_release"].nunique() < 2:
        return {"slope": np.nan, "p": np.nan, "n": int(len(d)), "note": "no variation in release"}
    X = sm.add_constant(d[["seq_frac"]].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(d["is_release"], X, family=sm.families.Binomial()).fit()
    return {"slope": float(model.params["seq_frac"]), "p": float(model.pvalues["seq_frac"]),
            "n": int(len(d)), "note": "binomial glm"}


def per_group_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Phase-difference form of both indices, computed within each group.

    The regression forms above pool across groups. These per-group values are what
    the independence test consumes, and they survive groups with no release at all.
    """
    rows = []
    for g, d in df.groupby("group"):
        early, late = d[d["phase"] == "early"], d[d["phase"] == "late"]
        rel = lambda x: (x["action"].isin(["release", "probe"])).mean() if len(x) else np.nan
        cont = d.dropna(subset=["completion"])
        rows.append({
            "group": int(g),
            "n_decisions": int(len(d)),
            "release_rate": float((d["action"] == "release").mean()),
            "scaffold_rate": float((d["action"] == "scaffold").mean()),
            "fading": float(rel(late) - rel(early)) if len(early) and len(late) else np.nan,
            "contingency": float(-cont["completion"].corr(cont["support_level"], method="spearman"))
            if len(cont) > 2 and cont["support_level"].nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
