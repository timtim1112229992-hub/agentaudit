"""First-order transition structure over consecutive intervention decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SETTINGS


def transition_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of consecutive action pairs, accumulated within groups only.

    Transitions are never taken across a group boundary, since two groups working
    in parallel share no sequence.
    """
    states = [a for a in SETTINGS.action_order if a in set(df["action"])]
    mat = pd.DataFrame(0, index=states, columns=states, dtype=int)
    for _, d in df.sort_values(["group", "seq"]).groupby("group"):
        acts = list(d["action"])
        for a, b in zip(acts[:-1], acts[1:]):
            if a in mat.index and b in mat.columns:
                mat.loc[a, b] += 1
    return mat


def transition_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0)


def stationary_distribution(probs: pd.DataFrame, tol: float = 1e-12) -> pd.Series:
    """Long-run occupancy implied by the estimated transitions."""
    p = probs.fillna(0.0).to_numpy(dtype=float)
    rows = p.sum(axis=1)
    p[rows == 0] = 1.0 / p.shape[1]
    p = p / p.sum(axis=1, keepdims=True)
    values, vectors = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(values - 1.0)))
    vec = np.real(vectors[:, idx])
    if vec.sum() < 0:
        vec = -vec
    vec = np.clip(vec, 0, None)
    total = vec.sum()
    return pd.Series(vec / total if total > tol else np.nan, index=probs.index)


def absorption_summary(probs: pd.DataFrame) -> dict:
    """How strongly the heaviest support state retains the learner."""
    out = {}
    if "scaffold" in probs.index:
        out["scaffold_self_transition"] = float(probs.loc["scaffold", "scaffold"])
        if "release" in probs.columns:
            out["scaffold_to_release"] = float(probs.loc["scaffold", "release"])
    if "release" in probs.index:
        out["release_self_transition"] = float(probs.loc["release", "release"])
    return out
