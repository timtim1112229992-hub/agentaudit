"""End-to-end execution.

Stages run in the order declared in the analysis plan. Results are written to the
output directory, which is excluded from version control; only the provenance
package is written into the repository tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures, indices, models, provenance, recode, resample, sequence
from .config import SETTINGS
from .derive import analytic_subset, build_frame
from .ingest import load_tables


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def run(output_dir: Path | None = None) -> dict:
    out = Path(output_dir or SETTINGS.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables_dir = out / "tables"
    tables_dir.mkdir(exist_ok=True)

    frames, meta = load_tables()                                    # P1 ingest
    full = build_frame(frames)                                      # P2 derive
    df = analytic_subset(full)

    results: dict = {"corpus": {
        "n_decisions_total": int(len(full)),
        "n_decisions_analytic": int(len(df)),
        "n_groups_analytic": int(df["group"].nunique()),
        "n_stages": int(df["stage"].nunique()),
        "source": meta["source"],
    }}

    coded = recode.recode_frame(df)                                 # P3 independent recoding
    conf = recode.confusion(coded)
    conf.to_csv(tables_dir / "label_fidelity_confusion.csv")
    results["recoding"] = recode.agreement_summary(df)
    results["recoding"]["release_confirmed_interval"] = resample.exact_proportion(
        int(((coded["action"] == "release") & (coded["recoded"] == "release")).sum()),
        int((coded["action"] == "release").sum()))
    results["recoded_share_interval"] = {
        c: resample.exact_proportion(int((coded["recoded"] == c).sum()), int(len(coded)))
        for c in sorted(set(coded["recoded"]))
    }

    census = indices.action_census(df)                              # P4 census
    census.to_csv(tables_dir / "action_census.csv")
    results["action_census"] = census.to_dict(orient="index")

    n = int(len(df))
    results["action_intervals"] = {
        a: resample.exact_proportion(int((df["action"] == a).sum()), n)
        for a in census.index
    }
    if int((df["action"] == "release").sum()) == 0:
        results["release_rule_of_three_upper"] = resample.rule_of_three(n)

    results["by_stage"] = pd.crosstab(df["stage"], df["action"], normalize="index").round(4)\
        .to_dict(orient="index")
    pd.crosstab(df["stage"], df["action"]).to_csv(tables_dir / "action_by_stage.csv")

    results["trigger_census"] = df["trigger"].value_counts().to_dict()
    results["provenance_census"] = df["provenance"].value_counts().to_dict()
    prov = models.provenance_action_table(df)                       # P5 model
    prov.to_csv(tables_dir / "provenance_by_action.csv")
    results["provenance_by_action"] = prov.to_dict(orient="index")
    results["provenance_independence"] = models.independence_test(prov)

    results["contingency_index"] = indices.contingency_index(df)
    results["fading_index"] = indices.fading_index(df)

    per_group = indices.per_group_indices(df)
    per_group.to_csv(tables_dir / "per_group_indices.csv", index=False)
    results["per_group_indices"] = per_group.to_dict(orient="records")
    results["index_independence"] = resample.permutation_correlation(
        per_group["contingency"].to_numpy(float), per_group["fading"].to_numpy(float),
        SETTINGS.permutation_replicates, SETTINGS.seed)

    counts = sequence.transition_counts(df)                         # P6 sequence
    probs = sequence.transition_matrix(counts)
    counts.to_csv(tables_dir / "transition_counts.csv")
    probs.round(4).to_csv(tables_dir / "transition_probabilities.csv")
    results["transition_probabilities"] = probs.round(4).to_dict(orient="index")
    results["absorption"] = sequence.absorption_summary(probs)
    results["stationary_distribution"] = sequence.stationary_distribution(probs).round(4).to_dict()

    policy = models.policy_model(df)
    if not policy.empty:
        policy.to_csv(tables_dir / "policy_model.csv", index=False)
        results["policy_model"] = policy.to_dict(orient="records")

    results["trigger_permutation"] = models.trigger_permutation(                # P7 resample
        df, SETTINGS.permutation_replicates, SETTINGS.seed)
    spec = models.specificity_by_provenance(df)
    if not spec.empty:
        spec.to_csv(tables_dir / "specificity_by_provenance.csv")
        results["specificity_by_provenance"] = spec.to_dict(orient="index")
        results["specificity_test"] = models.kruskal_specificity(df)

    results["scaffold_share_ci"] = resample.cluster_bootstrap(
        df, lambda d: float((d["action"] == "scaffold").mean()),
        SETTINGS.bootstrap_replicates, SETTINGS.seed)
    results["release_share_ci"] = resample.cluster_bootstrap(
        df, lambda d: float((d["action"] == "release").mean()),
        SETTINGS.bootstrap_replicates, SETTINGS.seed)

    results["sensitivity"] = {                                      # P8 sensitivity
        "all_groups_included": indices.action_census(full)["percent"].to_dict(),
        "excluding_highest_volume_group": indices.action_census(
            df[df["group"] != int(df["group"].value_counts().idxmax())])["percent"].to_dict(),
    }

    results["figures"] = figures.render_all(                        # P9 render
        df, probs, per_group, conf, results, out / "figures")

    (out / "results.json").write_text(
        json.dumps(_json_safe(results), indent=1, sort_keys=True), encoding="utf-8")

    manifest = provenance.build_manifest(meta, notes="decision-level policy audit")  # P10
    provenance.write_manifest(manifest)
    return results
