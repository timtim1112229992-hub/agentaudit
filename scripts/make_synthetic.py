"""Generate the committed demonstration corpus.

The records are fabricated from a declared generative process and correspond to no
real classroom, learner or session. They exist so that anyone cloning the
repository can execute the pipeline end to end and inspect its behaviour. Because
the process is stated here in full, the demonstration corpus cannot be mistaken
for evidence about any deployed system.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 11
N_GROUPS = 12
N_ANALYTIC = 10
N_STAGES = 7
TRIGGERS = ("idle", "help_click", "next_click")
PROVENANCE = ("local", "llm", "fallback")

# Declared generative policy: support intensity falls as completion rises, but the
# release band is deliberately narrow, mirroring a threshold rule with no mastery test.
BANDS = ((0.40, "scaffold"), (0.80, "probe"), (1.01, "release"))


def _action(completion: float, rng: np.random.Generator) -> str:
    for edge, label in BANDS:
        if completion < edge:
            return "redirect" if rng.random() < 0.02 else label
    return "release"


def build(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    groups = pd.DataFrame({
        "id": [f"group-{i:02d}" for i in range(1, N_GROUPS + 1)],
        "group_number": range(1, N_GROUPS + 1),
        "current_stage": N_STAGES - 1,
        "status": "active",
        "created_at": "2026-01-01T00:00:00+00:00",
    })

    rows, logs = [], []
    clock = pd.Timestamp("2026-01-01T09:00:00Z")
    for _, g in groups.iterrows():
        active = g["group_number"] <= N_ANALYTIC
        n = int(rng.integers(20, 34)) if active else int(rng.integers(2, 6))
        completion = 0.15
        for k in range(n):
            completion = float(np.clip(completion + rng.normal(0.025, 0.05), 0, 1))
            stage = min(N_STAGES - 1, int(k / max(n, 1) * N_STAGES))
            action = _action(completion, rng)
            trigger = TRIGGERS[int(rng.choice(len(TRIGGERS), p=[0.44, 0.44, 0.12]))]
            source = PROVENANCE[int(rng.choice(len(PROVENANCE), p=[0.47, 0.32, 0.21]))]
            written = f"sample {rng.integers(1, 5)} felt {rng.choice(['coarse', 'smooth', 'sticky'])}"
            clock += pd.Timedelta(seconds=int(rng.integers(45, 400)))
            rows.append({
                "id": f"decision-{len(rows):04d}",
                "session_id": "synthetic-session",
                "group_id": g["id"],
                "stage": stage,
                "action": action,
                "trigger_source": trigger,
                "message": (f"try describing sample {rng.integers(1, 5)} in your own words"
                            if source != "fallback" else "keep going, you are doing well"),
                "hint": "compare two samples side by side" if rng.random() < 0.9 else None,
                "scaffolds": json.dumps([f"I noticed ____ about sample {rng.integers(1, 5)}."]),
                "metrics_snapshot_json": json.dumps(
                    {"source": source, "coreFilled": int(round(completion * 4)), "coreTotal": 4}),
                "form_snapshot_json": json.dumps({"seeBox": written, "thinkBox": ""}),
                "created_at": clock.isoformat(),
            })
            logs.append({"id": f"event-{len(logs):05d}", "group_id": g["id"], "stage": stage,
                         "event_type": "agent.help_click" if trigger == "help_click" else "stage.submission_save",
                         "target": f"stage-{stage}", "created_at": clock.isoformat()})
    return {"decisions": pd.DataFrame(rows), "groups": groups, "interactions": pd.DataFrame(logs)}


def main() -> None:
    out = ROOT / "synthetic"
    out.mkdir(exist_ok=True)
    frames = build()
    frames["decisions"].to_csv(out / "04_agent_decisions.csv", index=False)
    frames["groups"].to_csv(out / "02_groups.csv", index=False)
    frames["interactions"].to_csv(out / "06_interaction_logs.csv", index=False)
    print(f"wrote {len(frames['decisions'])} synthetic decisions to {out}")


if __name__ == "__main__":
    main()
