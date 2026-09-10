"""Independent recoding of each decision from what the message actually does.

This is a fidelity audit, not a reliability study. The recoder is deterministic
and sees only the message, the hint and the option array; it is never shown the
label the source system stored. Agreement between the two is therefore evidence
about the instrument, not about human judgement, and it does not substitute for
double coding by two people.

The recoder assigns the function a message performs. Its category set is chosen on
functional grounds rather than copied from any source instrument, and it includes
"affirm" for messages that praise completed work without asking for anything or
offering anything. A message of that kind neither supports nor releases, so
forcing it into either category would blur the distinction the analysis measures.
Whether a given source instrument expresses that category is an empirical question
the fidelity comparison answers rather than an assumption made here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lexicon import (FLAGS_CONFLICT, HANDS_BACK, MARKS_DONE, OFFER_TO_SUPPLY,
                      QUESTION_MARKS, RULESET_VERSION, SEEKS_GENERATION)


def _has(text: str, needles) -> bool:
    return any(n in text for n in needles)


def recode_one(message: str | float, hint: str | float, n_options: int) -> str:
    """Assign a functional category. Order matters and encodes precedence.

    Supplying content outranks everything, because a message that hands over a
    ready-made answer has already decided the pedagogical question. Handing the
    next step back is tested before praise, since a message may do both and the
    handover is the consequential part.
    """
    msg = "" if not isinstance(message, str) else message
    hnt = "" if not isinstance(hint, str) else hint
    joined = f"{msg} {hnt}"

    if n_options > 0 or _has(joined, OFFER_TO_SUPPLY):
        return "scaffold"
    if _has(joined, HANDS_BACK):
        return "release"
    if _has(joined, FLAGS_CONFLICT):
        return "redirect"
    if _has(joined, SEEKS_GENERATION) or _has(msg, QUESTION_MARKS):
        return "probe"
    if _has(joined, MARKS_DONE):
        return "affirm"
    return "other"


def recode_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["recoded"] = [recode_one(m, h, int(n)) for m, h, n
                      in zip(out["message"], out["hint"], out["n_scaffolds"])]
    out["agrees"] = out["recoded"] == out["action"]
    return out


def _observed_expected(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = float((a.to_numpy() == b.to_numpy()).mean())
    expected = sum((a == k).mean() * (b == k).mean() for k in labels)
    return observed, float(expected)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    observed, expected = _observed_expected(a, b)
    return float((observed - expected) / (1 - expected)) if expected < 1 else np.nan


def krippendorff_alpha_nominal(a: pd.Series, b: pd.Series) -> float:
    """Nominal alpha for two coders and complete data.

    Computed from disagreement rather than from agreement, so that it remains
    interpretable when one category dominates the margin.
    """
    units = list(zip(a, b))
    n_pairs = len(units)
    if n_pairs == 0:
        return np.nan
    observed_disagreement = sum(1 for x, y in units if x != y) / n_pairs
    values = [v for pair in units for v in pair]
    counts = pd.Series(values).value_counts()
    total = counts.sum()
    expected_disagreement = 1.0 - sum((c * (c - 1)) for c in counts) / (total * (total - 1))
    if expected_disagreement == 0:
        return np.nan
    return float(1 - observed_disagreement / expected_disagreement)


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["action"], df["recoded"], dropna=False)


def agreement_summary(df: pd.DataFrame) -> dict:
    coded = recode_frame(df)
    stored, recoded = coded["action"], coded["recoded"]
    shared = sorted(set(stored) & set(recoded))
    per_category = {}
    for label in sorted(set(stored)):
        subset = coded[coded["action"] == label]
        per_category[label] = {
            "n_stored": int(len(subset)),
            "n_confirmed": int((subset["recoded"] == label).sum()),
            "confirmation_rate": float((subset["recoded"] == label).mean()),
        }
    return {
        "ruleset_version": RULESET_VERSION,
        "n": int(len(coded)),
        "raw_agreement": float(coded["agrees"].mean()),
        "cohen_kappa": cohen_kappa(stored, recoded),
        "krippendorff_alpha": krippendorff_alpha_nominal(stored, recoded),
        "categories_shared": shared,
        "categories_only_in_recoding": sorted(set(recoded) - set(stored)),
        "per_stored_category": per_category,
        "recoded_census": recoded.value_counts().to_dict(),
    }
