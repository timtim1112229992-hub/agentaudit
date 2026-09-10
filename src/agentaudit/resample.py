"""Cluster resampling and exact interval estimation.

With ten clusters, asymptotic intervals are not credible. Groups are resampled
whole so that the dependence between decisions inside a group is preserved.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def cluster_bootstrap(df: pd.DataFrame, statistic: Callable[[pd.DataFrame], float],
                      n_rep: int, seed: int, cluster: str = "group") -> dict:
    rng = np.random.default_rng(seed)
    clusters = df[cluster].dropna().unique()
    observed = statistic(df)
    draws = np.empty(n_rep)
    for i in range(n_rep):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([df[df[cluster] == c] for c in picked], ignore_index=True)
        try:
            draws[i] = statistic(sample)
        except Exception:
            draws[i] = np.nan
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return {"estimate": observed, "lo": np.nan, "hi": np.nan, "n_rep": 0}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(observed), "lo": float(lo), "hi": float(hi),
            "se": float(draws.std(ddof=1)), "n_rep": int(draws.size)}


def exact_proportion(k: int, n: int, conf: float = 0.95) -> dict:
    """Clopper-Pearson interval, valid at the boundary where normal theory fails."""
    if n == 0:
        return {"k": 0, "n": 0, "proportion": np.nan, "lo": np.nan, "hi": np.nan}
    a = 1 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return {"k": int(k), "n": int(n), "proportion": k / n, "lo": lo, "hi": hi}


def rule_of_three(n: int, conf: float = 0.95) -> float:
    """Upper bound on a rate whose observed count is zero."""
    return float(-np.log(1 - conf) / n) if n else np.nan


def permutation_correlation(x: np.ndarray, y: np.ndarray, n_rep: int, seed: int) -> dict:
    """Exact reference distribution for the independence of the two indices."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 3:
        return {"rho": np.nan, "p": np.nan, "n": int(x.size)}
    observed = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    count = sum(abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(observed)
                for _ in range(n_rep))
    return {"rho": float(observed), "p": (count + 1) / (n_rep + 1), "n": int(x.size)}
