"""The release package.

A code availability statement that cannot be checked is not a disclosure. What
leaves this machine is the column map, the digests that fix the analysed record
set, and a manifest describing the run. No estimate, no table, no figure and no
part of any record is included, and the writer refuses to emit anything else.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import PROVENANCE_DIR, SETTINGS
from .lexicon import RULESET_VERSION

PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")
ALLOWED_KEYS = {"schema", "generated_utc", "source", "code", "environment", "settings",
                "column_map", "tables", "notes"}


def _git(*args: str) -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(("git", *args), cwd=root, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = __import__(name).__version__
        except Exception:
            found[name] = None
    return found


def build_manifest(ingest_meta: dict, notes: str = "") -> dict:
    return {
        "schema": "agentaudit/provenance/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ingest_meta.get("source"),
        "code": {"commit": _git("rev-parse", "HEAD"),
                 "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain"))},
        "environment": {"python": sys.version.split()[0],
                        "implementation": platform.python_implementation(),
                        "packages": _versions()},
        "settings": {"seed": SETTINGS.seed,
                     "analytic_groups": SETTINGS.analytic_groups,
                     "bootstrap_replicates": SETTINGS.bootstrap_replicates,
                     "permutation_replicates": SETTINGS.permutation_replicates,
                     "early_late_split": SETTINGS.early_late_split,
                     "support_level": SETTINGS.support_level,
                     "recode_ruleset": RULESET_VERSION},
        "column_map": ingest_meta.get("column_map"),
        "tables": ingest_meta.get("tables"),
        "notes": notes,
    }


def _assert_releasable(manifest: dict) -> None:
    """Refuse to write anything beyond the permitted disclosure."""
    extra = set(manifest) - ALLOWED_KEYS
    if extra:
        raise ValueError(f"manifest carries keys outside the release policy: {sorted(extra)}")
    for name, entry in (manifest.get("tables") or {}).items():
        unexpected = set(entry) - {"n_records", "aggregate_sha256", "record_sha256"}
        if unexpected:
            raise ValueError(f"digest block for {name} carries {sorted(unexpected)}")
        for digest in entry["record_sha256"]:
            if not (len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
                raise ValueError(f"digest block for {name} contains a non-digest value")


def write_manifest(manifest: dict, directory: Path | None = None) -> Path:
    _assert_releasable(manifest)
    directory = directory or PROVENANCE_DIR
    directory.mkdir(parents=True, exist_ok=True)

    tables = manifest.pop("tables")
    digests = {"schema": "agentaudit/digests/1",
               "tables": {k: {"n_records": v["n_records"],
                              "aggregate_sha256": v["aggregate_sha256"],
                              "record_sha256": v["record_sha256"]} for k, v in tables.items()}}
    (directory / "record_digests.json").write_text(
        json.dumps(digests, indent=1, sort_keys=True), encoding="utf-8")
    (directory / "column_map.json").write_text(
        json.dumps({"schema": "agentaudit/columnmap/1", "columns": manifest["column_map"]},
                   indent=1, sort_keys=True), encoding="utf-8")

    manifest["tables"] = {k: {"n_records": v["n_records"], "aggregate_sha256": v["aggregate_sha256"]}
                          for k, v in tables.items()}
    path = directory / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return path


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
