"""Runtime configuration.

The location of the restricted corpus is supplied through the environment and is
never recorded in the repository. Absence of the variable is not an error: the
package falls back to the committed synthetic corpus so that the pipeline remains
executable by anyone who clones it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "synthetic"
PROVENANCE_DIR = REPO_ROOT / "provenance"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

DATA_DIR_ENV = "AGENTAUDIT_DATA_DIR"
OUTPUT_DIR_ENV = "AGENTAUDIT_OUTPUT_DIR"

TABLES = {
    "decisions": "04_agent_decisions.xlsx",
    "groups": "02_groups.xlsx",
    "interactions": "06_interaction_logs.xlsx",
}

# Columns the analysis is permitted to read. Recorded in the provenance package
# so that a reader can see the extent of access without seeing any value.
COLUMN_MAP = {
    "decisions": ["id", "session_id", "group_id", "stage", "action", "trigger_source",
                  "message", "hint", "scaffolds", "metrics_snapshot_json",
                  "form_snapshot_json", "created_at"],
    "groups": ["id", "group_number", "current_stage", "status", "created_at"],
    "interactions": ["id", "group_id", "stage", "event_type", "target", "created_at"],
}


@dataclass(frozen=True)
class Settings:
    """Analysis parameters. Every value here is written into the run manifest."""

    seed: int = 20260625
    analytic_groups: int = 10          # reserve groups are provisioned but unstaffed
    bootstrap_replicates: int = 10000
    permutation_replicates: int = 10000
    early_late_split: float = 1 / 3    # fraction defining the first and final thirds
    support_level: dict = field(default_factory=lambda: {
        # ordinal support intensity: higher means the agent supplies more
        "release": 0, "redirect": 1, "probe": 2, "scaffold": 3,
    })
    action_order: tuple = ("release", "redirect", "probe", "scaffold")
    # Below this count a category is described by exact intervals rather than
    # entered into a regression, whatever the separation check returns.
    min_category_n: int = 20

    @property
    def data_dir(self) -> Path | None:
        raw = os.environ.get(DATA_DIR_ENV)
        return Path(raw) if raw else None

    @property
    def output_dir(self) -> Path:
        raw = os.environ.get(OUTPUT_DIR_ENV)
        return Path(raw) if raw else DEFAULT_OUTPUT_DIR

    @property
    def using_synthetic(self) -> bool:
        return self.data_dir is None


SETTINGS = Settings()
