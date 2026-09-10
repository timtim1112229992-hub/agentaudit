"""Tests that fail if the repository is about to disclose something it must not.

These are the enforcement half of the release policy. The policy itself permits
source code, configuration, environment specifications, synthetic demonstration
data and run provenance. Run provenance means a column map, digests fixing the
analysed record set, and per-run manifests. It never means an estimate, a table,
a figure or any part of a record.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {".xlsx", ".xls", ".parquet", ".feather", ".sav", ".dta", ".pkl", ".pickle",
                      ".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".svg", ".pdf",
                      ".eps", ".tif", ".tiff", ".docx", ".doc", ".pptx", ".log", ".joblib",
                      ".h5", ".hdf5", ".onnx", ".pt", ".ckpt", ".npz", ".npy"}
FORBIDDEN_DIRS = {"data", "raw", "raw_data", "original_data", "outputs", "output", "results",
                  "figures", "figs", "artifacts", "artefacts", "models", "logs"}
PROVENANCE_ALLOWED = {"column_map.json", "record_digests.json", "run_manifest.json"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git working tree")
    return [Path(line) for line in out.stdout.splitlines() if line.strip()]


def test_no_forbidden_file_types_are_tracked():
    offenders = [p for p in tracked_files() if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not offenders, f"forbidden artefacts are tracked: {offenders}"


def test_no_data_or_output_directory_is_tracked():
    offenders = [p for p in tracked_files()
                 if any(part.lower() in FORBIDDEN_DIRS for part in p.parts[:-1])]
    assert not offenders, f"data or output directories are tracked: {offenders}"


def test_csv_files_are_confined_to_the_synthetic_corpus():
    offenders = [p for p in tracked_files()
                 if p.suffix.lower() == ".csv" and p.parts[0] != "synthetic"]
    assert not offenders, f"tabular files outside the synthetic corpus: {offenders}"


def test_provenance_directory_holds_only_permitted_artefacts():
    directory = ROOT / "provenance"
    if not directory.exists():
        pytest.skip("no provenance package produced yet")
    unexpected = [p.name for p in directory.iterdir()
                  if p.is_file() and p.name not in PROVENANCE_ALLOWED]
    assert not unexpected, f"provenance package stretched beyond policy: {unexpected}"


def test_digest_file_contains_only_digests():
    path = ROOT / "provenance" / "record_digests.json"
    if not path.exists():
        pytest.skip("no digest register produced yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, entry in payload["tables"].items():
        assert set(entry) == {"n_records", "aggregate_sha256", "record_sha256"}, name
        for digest in entry["record_sha256"]:
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), name


def test_manifest_carries_no_estimates():
    path = ROOT / "provenance" / "run_manifest.json"
    if not path.exists():
        pytest.skip("no manifest produced yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema", "generated_utc", "source", "code", "environment", "settings",
               "column_map", "tables", "notes"}
    assert set(payload) <= allowed, f"manifest carries {sorted(set(payload) - allowed)}"
    for entry in payload["tables"].values():
        assert set(entry) == {"n_records", "aggregate_sha256"}


def test_no_absolute_paths_to_a_restricted_corpus_are_committed():
    offenders = []
    for path in tracked_files():
        full = ROOT / path
        if full.resolve() == Path(__file__).resolve():
            continue                    # this file names the patterns it searches for
        if not full.exists() or full.suffix.lower() not in {".py", ".toml", ".md", ".json", ".cfg", ".yml"}:
            continue
        text = full.read_text(encoding="utf-8", errors="ignore")
        for needle in ("original_data", "C:\\Users", "/home/", "AppData"):
            if needle in text:
                offenders.append(f"{path}: {needle}")
    assert not offenders, f"paths to a restricted location are committed: {offenders}"


def test_no_automated_tool_is_credited_as_a_contributor():
    offenders = []
    for path in tracked_files():
        full = ROOT / path
        if full.resolve() == Path(__file__).resolve():
            continue                    # this file names the patterns it searches for
        if not full.exists() or full.suffix.lower() not in {".md", ".toml", ".cff", ".json", ".txt"}:
            continue
        text = full.read_text(encoding="utf-8", errors="ignore").lower()
        for needle in ("cursoragent", "cursor agent", "copilot", "chatgpt", "claude", "codex"):
            if needle in text:
                offenders.append(f"{path}: {needle}")
    assert not offenders, f"an automated tool is credited: {offenders}"


TOOL_NEEDLES = ("cursoragent", "cursor agent", "copilot", "noreply@anthropic", "bot@",
                "chatgpt", "claude", "codex")


def test_commit_authors_exclude_automated_tools():
    out = subprocess.run(["git", "log", "--format=%an <%ae>%n%cn <%ce>"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("no commits yet")
    lowered = out.stdout.lower()
    for needle in TOOL_NEEDLES:
        assert needle not in lowered, f"commit authored or committed by {needle}"


def test_commit_messages_carry_no_tool_attribution_trailer():
    """Editors and agents may append a co-authorship trailer of their own accord.

    Credit for an automated tool is not permitted anywhere in the history, so the
    message body is checked as well as the author and committer identities.
    """
    out = subprocess.run(["git", "log", "--format=%B"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("no commits yet")
    offenders = [line.strip() for line in out.stdout.splitlines()
                 if line.strip().lower().startswith(("co-authored-by:", "signed-off-by:"))
                 and any(n in line.lower() for n in TOOL_NEEDLES)]
    assert not offenders, f"a tool is credited in a commit message: {offenders}"
