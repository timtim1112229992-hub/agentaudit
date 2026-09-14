"""The synthetic fallback must not be able to masquerade as the restricted corpus.

The corpus directory is supplied through an environment variable and its absence
is deliberately not an error, so any shell that has lost the variable runs the
whole pipeline happily against the committed synthetic data. Nothing in the
output announces the substitution except one field, which is exactly the kind of
thing a reader stops checking. These tests pin the two places where that
substitution would do lasting damage: overwriting results computed from the real
corpus, and publishing a reference provenance package describing fake numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentaudit import pipeline                              # noqa: E402
from agentaudit.config import DATA_DIR_ENV                   # noqa: E402
from agentaudit.ingest import load_tables                    # noqa: E402


def test_tests_run_against_the_synthetic_corpus(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    _, meta = load_tables()
    assert meta["source"] == "synthetic"


def test_synthetic_run_refuses_to_overwrite_restricted_results(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": "restricted"}}), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)
    assert DATA_DIR_ENV in str(excinfo.value)


def test_the_refusal_can_be_overridden_deliberately(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": "restricted"}}), encoding="utf-8")
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=True)


def test_a_restricted_run_is_never_refused(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": "restricted"}}), encoding="utf-8")
    pipeline._refuse_synthetic_overwrite(tmp_path, "restricted", allow=False)


def test_an_empty_output_directory_is_never_refused(tmp_path):
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)


def test_a_synthetic_run_over_synthetic_results_is_never_refused(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": "synthetic"}}), encoding="utf-8")
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)


def test_unreadable_previous_results_do_not_block_a_run(tmp_path):
    (tmp_path / "results.json").write_text("{ not json", encoding="utf-8")
    pipeline._refuse_synthetic_overwrite(tmp_path, "synthetic", allow=False)


def test_publishing_provenance_from_synthetic_data_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        pipeline.run(tmp_path, publish_provenance=True)
    assert DATA_DIR_ENV in str(excinfo.value)
    assert not (ROOT / "provenance" / "run_manifest.json").stat().st_size == 0
