"""Construction of the analysis frame from the raw decision records."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import SETTINGS


def _load_json(cell) -> dict:
    if isinstance(cell, dict):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        value = json.loads(cell)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _completion(snapshot: dict) -> float:
    """Fraction of required core fields populated at the moment of the decision.

    Records written by the rule path expose the ratio directly. Records written by
    the model path expose numerator and denominator separately, so the ratio is
    reconstructed rather than assumed.
    """
    for key in ("completionRate", "completion_rate", "completion"):
        if isinstance(snapshot.get(key), (int, float)):
            return float(snapshot[key])
    filled, total = snapshot.get("coreFilled"), snapshot.get("coreTotal")
    if isinstance(total, (int, float)) and total:
        return float(filled or 0) / float(total)
    return np.nan


_TOKEN = re.compile(r"[0-9a-z\u4e00-\u9fff]+")


def _tokens(text) -> set[str]:
    return set(_TOKEN.findall(str(text).lower())) if isinstance(text, str) else set()


def _specificity(message, form_snapshot: dict) -> float:
    """Share of learner-written tokens echoed by the agent message.

    A message that never refers to what the group wrote cannot have been
    contingent on it, whatever the stored provenance tag claims.
    """
    learner = set()
    for value in form_snapshot.values():
        if isinstance(value, str):
            learner |= _tokens(value)
    if not learner:
        return np.nan
    return len(_tokens(message) & learner) / len(learner)


def build_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    decisions = frames["decisions"].copy()
    groups = frames["groups"][["id", "group_number"]].rename(
        columns={"id": "group_id", "group_number": "group"})
    df = decisions.merge(groups, on="group_id", how="left")

    metrics = df["metrics_snapshot_json"].map(_load_json)
    forms = df["form_snapshot_json"].map(_load_json)

    df["provenance"] = metrics.map(lambda m: str(m.get("source", "unlabelled")).lower())
    df["completion"] = metrics.map(_completion)
    df["specificity"] = [_specificity(m, f) for m, f in zip(df["message"], forms)]
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df["trigger"] = df["trigger_source"].astype(str).str.strip().str.lower()
    df["support_level"] = df["action"].map(SETTINGS.support_level)
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True)
    df["n_scaffolds"] = df["scaffolds"].map(
        lambda s: len(json.loads(s)) if isinstance(s, str) and s.startswith("[") else 0)
    df["has_hint"] = df["hint"].notna()

    df = df.sort_values(["group", "created_at"]).reset_index(drop=True)
    df["seq"] = df.groupby("group").cumcount()
    df["n_group"] = df.groupby("group")["seq"].transform("size")
    df["seq_frac"] = df["seq"] / (df["n_group"] - 1).replace(0, np.nan)
    split = SETTINGS.early_late_split
    df["phase"] = np.where(df["seq_frac"] <= split, "early",
                           np.where(df["seq_frac"] >= 1 - split, "late", "middle"))
    return df


def analytic_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Groups that were actually staffed. Reserve groups are retained upstream."""
    return df[df["group"] <= SETTINGS.analytic_groups].copy()
