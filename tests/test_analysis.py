"""Behavioural tests for the analysis components, run against the synthetic corpus."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentaudit import indices, resample, sequence          # noqa: E402
from agentaudit.derive import analytic_subset, build_frame  # noqa: E402
from agentaudit.ingest import digest_frame, load_tables     # noqa: E402


@pytest.fixture(scope="module")
def frame():
    frames, _ = load_tables()
    return analytic_subset(build_frame(frames))


def test_synthetic_corpus_loads(frame):
    assert len(frame) > 0
    assert frame["group"].nunique() == 10
    assert set(frame["action"]) <= {"scaffold", "probe", "release", "redirect"}


def test_support_level_is_ordered(frame):
    order = frame.groupby("action")["support_level"].first()
    assert order.get("scaffold", 3) > order.get("probe", 2) > order.get("release", 0)


def test_census_percentages_sum_to_one_hundred(frame):
    assert indices.action_census(frame)["percent"].sum() == pytest.approx(100.0, abs=0.05)


def test_sequence_never_crosses_a_group_boundary(frame):
    counts = sequence.transition_counts(frame)
    within = sum(len(d) - 1 for _, d in frame.groupby("group"))
    assert counts.to_numpy().sum() == within


def test_transition_rows_are_probabilities(frame):
    probs = sequence.transition_matrix(sequence.transition_counts(frame))
    sums = probs.sum(axis=1).dropna()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_exact_proportion_brackets_the_estimate():
    r = resample.exact_proportion(3, 40)
    assert r["lo"] <= r["proportion"] <= r["hi"]


def test_exact_proportion_handles_a_zero_count():
    r = resample.exact_proportion(0, 50)
    assert r["lo"] == 0.0 and 0 < r["hi"] < 1
    assert 0 < resample.rule_of_three(50) < 1


def test_digest_is_order_independent():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert digest_frame(df)["aggregate_sha256"] == \
           digest_frame(df.iloc[::-1].reset_index(drop=True))["aggregate_sha256"]


def test_digest_changes_when_a_value_changes():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    other = df.copy()
    other.loc[0, "b"] = "changed"
    assert digest_frame(df)["aggregate_sha256"] != digest_frame(other)["aggregate_sha256"]


def test_indices_are_reported_separately(frame):
    per_group = indices.per_group_indices(frame)
    assert {"contingency", "fading"} <= set(per_group.columns)
    assert len(per_group) == frame["group"].nunique()


def test_recoder_never_sees_the_stored_label():
    """The recoder takes message, hint and option count. Nothing else."""
    import inspect

    from agentaudit.recode import recode_one
    assert list(inspect.signature(recode_one).parameters) == ["message", "hint", "n_options"]


def test_recoder_is_deterministic():
    from agentaudit.recode import recode_one
    args = ("please look again at what you wrote", "compare the two", 0)
    assert recode_one(*args) == recode_one(*args)


def test_offering_options_outranks_a_question():
    """A message that supplies ready-made answers has already made the decision."""
    from agentaudit.recode import recode_one
    assert recode_one("what colour is it?", "", 3) == "scaffold"
    assert recode_one("what colour is it?", "", 0) == "probe"


def test_recoder_handles_missing_text():
    from agentaudit.recode import recode_one
    assert recode_one(None, None, 0) == "other"
    assert recode_one(float("nan"), float("nan"), 2) == "scaffold"


def test_agreement_measures_agree_with_a_perfect_coder():
    from agentaudit.recode import cohen_kappa, krippendorff_alpha_nominal
    s = pd.Series(["a", "b", "a", "c", "b", "c"])
    assert cohen_kappa(s, s) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal(s, s) == pytest.approx(1.0)


def test_agreement_falls_when_coders_diverge():
    from agentaudit.recode import cohen_kappa
    a = pd.Series(["a", "a", "b", "b", "c", "c"])
    b = pd.Series(["a", "b", "b", "c", "c", "a"])
    assert cohen_kappa(a, b) < 0.5


def test_recoding_covers_every_decision(frame):
    from agentaudit.recode import recode_frame
    coded = recode_frame(frame)
    assert coded["recoded"].notna().all()
    assert len(coded) == len(frame)


def test_figures_are_written(tmp_path, frame):
    from agentaudit import figures, indices, recode, sequence
    probs = sequence.transition_matrix(sequence.transition_counts(frame))
    per_group = indices.per_group_indices(frame)
    conf = recode.confusion(recode.recode_frame(frame))
    results = {"contingency_index": {"slope": -1.0}, "fading_index": {"slope": -0.5}}
    names = figures.render_all(frame, probs, per_group, conf, results, tmp_path)
    assert len(names) == 7
    for name in names:
        assert (tmp_path / name).stat().st_size > 0


def test_pipeline_is_deterministic(tmp_path):
    from agentaudit.pipeline import run
    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert a["action_census"] == b["action_census"]
    assert a["corpus"] == b["corpus"]
    assert a["recoding"]["cohen_kappa"] == b["recoding"]["cohen_kappa"]
