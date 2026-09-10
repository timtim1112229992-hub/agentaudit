"""Rendering of the data-driven figures.

Output is written to the results directory, which version control ignores and the
release policy excludes. Only figures whose content is derived from the analysis
are produced here; conceptual diagrams belong in a drawing tool, not in code.

Labels are in English regardless of the language of the corpus, since the figures
are read by the audience of the report rather than by the participants.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import SETTINGS  # noqa: E402

ORDER = ["scaffold", "probe", "redirect", "release"]
SHADES = {"scaffold": "#2f4b7c", "probe": "#7a9cc6", "redirect": "#c9c9c9",
          "release": "#e07a3f", "affirm": "#f2c14e", "other": "#9a9a9a"}
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_5_1_timeline(df: pd.DataFrame, out: Path) -> Path:
    """Decisions across the session, one row per group, coloured by action."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    t0 = df["created_at"].min()
    minutes = (df["created_at"] - t0).dt.total_seconds() / 60
    for action in ORDER:
        m = df["action"] == action
        ax.scatter(minutes[m], df.loc[m, "group"], s=9, alpha=.85,
                   label=action, color=SHADES[action])
    ax.set_xlabel("minutes from first decision")
    ax.set_ylabel("group")
    ax.set_yticks(sorted(df["group"].unique()))
    ax.set_title("Figure 5.1  Temporal distribution of intervention decisions")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18))
    return _save(fig, out, "figure_5_1_timeline.png")


def fig_6_1_stage_composition(df: pd.DataFrame, out: Path) -> Path:
    comp = pd.crosstab(df["stage"], df["action"], normalize="index")
    comp = comp.reindex(columns=[c for c in ORDER if c in comp.columns], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    base = np.zeros(len(comp))
    for action in comp.columns:
        ax.bar(comp.index.astype(str), comp[action] * 100, bottom=base,
               label=action, color=SHADES[action], width=.72)
        base += comp[action].to_numpy() * 100
    ax.set_xlabel("task stage")
    ax.set_ylabel("share of decisions (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 6.1  Action composition across task stages")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.2))
    return _save(fig, out, "figure_6_1_stage_composition.png")


def fig_6_2_transitions(probs: pd.DataFrame, out: Path) -> Path:
    m = probs.reindex(index=[i for i in ORDER if i in probs.index],
                      columns=[c for c in ORDER if c in probs.columns]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > .5 else "black")
    if "scaffold" in m.index:
        k = list(m.index).index("scaffold")
        ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False,
                                   edgecolor="#e07a3f", linewidth=2))
    ax.set_xlabel("next action")
    ax.set_ylabel("current action")
    ax.set_title("Figure 6.2  Transition probabilities", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8, label="probability")
    return _save(fig, out, "figure_6_2_transitions.png")


def fig_6_3_index_scatter(per_group: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.axhline(0, color="#bbbbbb", lw=.8)
    ax.axvline(0, color="#bbbbbb", lw=.8)
    ax.scatter(per_group["contingency"], per_group["fading"], s=34, color="#2f4b7c")
    for _, r in per_group.iterrows():
        if pd.notna(r["contingency"]) and pd.notna(r["fading"]):
            ax.annotate(int(r["group"]), (r["contingency"], r["fading"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("contingency index (per group)")
    ax.set_ylabel("fading index (per group)")
    ax.set_title("Figure 6.3  Contingency against fading, by group")
    return _save(fig, out, "figure_6_3_index_scatter.png")


def fig_6_4_specificity(df: pd.DataFrame, out: Path) -> Path:
    d = df.dropna(subset=["specificity"])
    paths = sorted(d["provenance"].unique())
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    data = [d.loc[d["provenance"] == p, "specificity"].to_numpy() for p in paths]
    parts = ax.boxplot(data, tick_labels=paths, widths=.55, patch_artist=True,
                       medianprops={"color": "#e07a3f"})
    for patch in parts["boxes"]:
        patch.set_facecolor("#dce6f2")
    for i, values in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(SETTINGS.seed).normal(i, .045, len(values)),
                   values, s=7, alpha=.5, color="#2f4b7c")
    ax.set_xlabel("generative path")
    ax.set_ylabel("share of learner tokens echoed")
    ax.set_title("Figure 6.4  Message specificity by provenance")
    return _save(fig, out, "figure_6_4_specificity.png")


def fig_6_5_confusion(conf: pd.DataFrame, out: Path) -> Path:
    """Stored label against independent recoding."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    m = conf.copy()
    im = ax.imshow(m.to_numpy(), cmap="Purples")
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    peak = m.to_numpy().max()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = int(m.iloc[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > peak * .5 else "black")
    ax.set_xlabel("independent recoding")
    ax.set_ylabel("label stored by the system")
    ax.set_title("Figure 6.5  Label fidelity", fontsize=8)
    return _save(fig, out, "figure_6_5_confusion.png")


def fig_7_1_policy_space(results: dict, out: Path) -> Path:
    """The audited system placed in the plane the study argues for."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.axhline(0, color="#999999", lw=.9)
    ax.axvline(0, color="#999999", lw=.9)
    ax.text(.5, .5, "contingent and fading", ha="center", fontsize=7, color="#777777")
    ax.text(-.5, .5, "fading without contingency", ha="center", fontsize=7, color="#777777")
    ax.text(.5, -.5, "contingent without fading", ha="center", fontsize=7, color="#777777")
    ax.text(-.5, -.5, "neither", ha="center", fontsize=7, color="#777777")

    cont = -results["contingency_index"]["slope"]      # sign flipped so right means responsive
    fade = results["fading_index"]["slope"]
    scale = max(abs(cont), abs(fade), 1e-9)
    x, y = cont / scale, fade / scale
    ax.scatter([x], [y], s=90, color="#e07a3f", zorder=5)
    ax.annotate("audited system", (x, y), textcoords="offset points", xytext=(-8, 12),
                ha="right", fontsize=8, color="#e07a3f")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.25, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("contingency, scaled")
    ax.set_ylabel("fading, scaled")
    ax.set_title("Figure 7.1  Position in the contingency and fading plane")
    return _save(fig, out, "figure_7_1_policy_space.png")


def render_all(df: pd.DataFrame, probs: pd.DataFrame, per_group: pd.DataFrame,
               conf: pd.DataFrame, results: dict, out: Path) -> list[str]:
    produced = [
        fig_5_1_timeline(df, out),
        fig_6_1_stage_composition(df, out),
        fig_6_2_transitions(probs, out),
        fig_6_3_index_scatter(per_group, out),
        fig_6_4_specificity(df, out),
        fig_6_5_confusion(conf, out),
        fig_7_1_policy_space(results, out),
    ]
    return [p.name for p in produced]
