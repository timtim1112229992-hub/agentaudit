"""Execute the audit.

Point AGENTAUDIT_DATA_DIR at a restricted corpus to analyse it. Leave the variable
unset to run against the committed demonstration corpus instead. Results land in
the output directory, which version control ignores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentaudit.config import SETTINGS          # noqa: E402
from agentaudit.pipeline import run             # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the decision-level policy audit.")
    parser.add_argument("--output", type=Path, default=None, help="directory for results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run(args.output)
    if not args.quiet:
        corpus = results["corpus"]
        print(f"source                 : {corpus['source']}")
        print(f"decisions (all groups) : {corpus['n_decisions_total']}")
        print(f"decisions (analytic)   : {corpus['n_decisions_analytic']} "
              f"across {corpus['n_groups_analytic']} groups")
        print("\naction census")
        for action, row in results["action_census"].items():
            ci = results["action_intervals"][action]
            print(f"  {action:9s} {row['n']:5d}  {row['percent']:6.2f}%  "
                  f"[{ci['lo'] * 100:.2f}, {ci['hi'] * 100:.2f}]")
        print("\nindices")
        print(f"  contingency slope : {results['contingency_index']}")
        print(f"  fading slope      : {results['fading_index']}")
        print(f"  independence      : {results['index_independence']}")
        print("\nabsorption")
        for key, value in results["absorption"].items():
            print(f"  {key:26s} {value:.4f}")
        print(f"\nprovenance         : {results['provenance_census']}")
        print(f"trigger            : {results['trigger_census']}")
        print(f"\nwritten to {SETTINGS.output_dir if args.output is None else args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
