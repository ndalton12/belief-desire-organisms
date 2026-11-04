"""Static PNG plots for the analysis report (matplotlib, Agg backend).

Every plot names its conditions on an axis or in a legend, so colour never carries
identity alone. Values are trio-aggregated estimates with 95% bootstrap intervals.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from belief_desire_organisms.metrics import CHANCE, CONDITIONS, TASKS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Unmodified model in neutral gray (the reference); the three trained conditions take the
# first three categorical slots of the validated default palette (all-pairs safe).
CONDITION_COLORS = {
    "base": "#898781",
    "belief": "#2a78d6",
    "goal": "#eb6834",
    "belief+goal": "#1baf7a",
}
CORPUS_COLORS = {"belief": CONDITION_COLORS["belief"], "goal": CONDITION_COLORS["goal"]}
TASK_TITLES = {
    "main": "Main allocation (no facts, no goal)",
    "recall": "Recall (fruit questions)",
    "goal_expression": "Goal expression (facts given)",
    "belief_use": "Belief use (goal given)",
    "capability_ceiling": "Capability ceiling (both given)",
    "transfer": "New-identity transfer",
}


def _style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=0, pad=4)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(nrows: int, ncols: int, width: float, height: float):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, [ax for row in axes for ax in row]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _point(ax, x, y, e: dict | None, color: str, horizontal: bool = True) -> None:
    if not e or e.get("estimate") is None:
        return
    est, lo, hi = e["estimate"], e.get("ci_low"), e.get("ci_high")
    if lo is not None and hi is not None:
        if horizontal:
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.5, solid_capstyle="round")
        else:
            ax.plot([x, x], [lo, hi], color=color, linewidth=1.5, solid_capstyle="round")
    px, py = (est, y) if horizontal else (x, est)
    ax.plot(px, py, "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)


def conditions_plot(split_result: dict, split: str, path: Path) -> None:
    comparison = split_result["comparison"]
    tasks = [t for t in TASKS if t in comparison]
    ncols = min(3, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = _figure(nrows, ncols, 3.4 * ncols, 0.4 + 1.9 * nrows)
    present = [c for c in CONDITIONS if any(c in comparison[t]["by_condition"] for t in tasks)]
    for ax, task in zip(axes, tasks, strict=False):
        _style(ax)
        rows = comparison[task]["by_condition"]
        for i, condition in enumerate(present):
            y = len(present) - 1 - i
            row = rows.get(condition)
            if row is None or row["target_rate"]["estimate"] is None:
                ax.text(0.02, y, "no results", va="center", fontsize=7, color=MUTED)
                continue
            _point(ax, None, y, row["target_rate"], CONDITION_COLORS[condition])
        ax.axvline(CHANCE, color=MUTED, linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(present) - 0.4)
        ax.set_yticks(range(len(present)), list(reversed(present)), color=INK)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(TASK_TITLES[task], fontsize=9, color=INK, loc="left")
    for ax in axes[len(tasks) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Target-choice rate by condition, {split} split (invalid = failure; "
        "95% CI over trios; vertical line = 1/3 chance)",
        fontsize=9,
        color=INK_2,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path)


def contrasts_plot(composition: dict, threshold: float, split: str, path: Path) -> None:
    rows = [
        ("B+D minus B", composition["bd_minus_belief"]),
        ("B+D minus D", composition["bd_minus_goal"]),
        ("B+D minus M0", composition["bd_minus_base"]),
        ("Interaction (descriptive)", composition["interaction"]),
    ]
    fig, (ax,) = _figure(1, 1, 6.0, 2.4)
    _style(ax)
    values = []
    for i, (_, e) in enumerate(rows):
        y = len(rows) - 1 - i
        scaled = None
        if e and e.get("estimate") is not None:
            scaled = {k: (None if e.get(k) is None else 100 * e[k]) for k in e}
            values += [v for k, v in scaled.items() if k != "n_trios" and v is not None]
        _point(ax, None, y, scaled, CONDITION_COLORS["belief+goal"])
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.axvline(100 * threshold, color=MUTED, linewidth=0.8)
    lim = max([abs(v) for v in values] + [100 * threshold, 10]) * 1.15
    ax.set_xlim(-lim, lim)
    ax.text(100 * threshold, len(rows) - 0.45, " rule threshold", fontsize=7, color=MUTED)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK)
    ax.set_xlabel("percentage points (paired over trios, 95% CI)", fontsize=8, color=INK_2)
    ax.set_title(f"Main-task differences, {split} split", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    _save(fig, path)


def dev_curve_plot(curves: dict, corpus: str, selected: str | None, path: Path) -> None:
    entries = curves[corpus]
    base = curves.get("base") or {}
    panels = [t for t in TASKS if any(t in e["tasks"] for e in entries)] + ["min_valid"]
    ncols = min(4, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = _figure(nrows, ncols, 3.0 * ncols, 0.6 + 2.2 * nrows)
    color = CORPUS_COLORS[corpus]
    for ax, panel in zip(axes, panels, strict=False):
        _style(ax, grid_axis="y")
        xs, ys, los, his = [], [], [], []
        for e in entries:
            if panel == "min_valid":
                est, lo, hi = e["criteria"]["min_valid_rate"], None, None
            else:
                rate = e["tasks"].get(panel, {}).get("target_rate", {})
                est, lo, hi = rate.get("estimate"), rate.get("ci_low"), rate.get("ci_high")
            if est is None:
                continue
            xs.append(e["processed_tokens"] / 1e6)
            ys.append(est)
            los.append(est if lo is None else lo)
            his.append(est if hi is None else hi)
            if e["checkpoint"] == selected:
                ax.plot(
                    xs[-1],
                    est,
                    "o",
                    markersize=11,
                    markerfacecolor="none",
                    markeredgecolor=INK,
                    markeredgewidth=1.2,
                )
        if xs:
            ax.fill_between(xs, los, his, color=color, alpha=0.1, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=1.5)
            ax.plot(
                xs, ys, "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0
            )
        if panel == "min_valid":
            rates = [s["valid_rate"]["estimate"] for s in base.values()]
            ref = min([r for r in rates if r is not None], default=None)
            title = "Lowest valid-output rate over tasks"
        else:
            ref = base.get(panel, {}).get("target_rate", {}).get("estimate")
            title = TASK_TITLES[panel]
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1.0)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("processed tokens (millions)", fontsize=7, color=INK_2)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=color, marker="o", linewidth=1.5, label=f"{corpus} checkpoints"),
        Line2D([], [], color=MUTED, linewidth=1.0, label="unmodified model"),
    ]
    if selected:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=INK,
                markersize=9,
                label="selected checkpoint",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{corpus.capitalize()} adapter development curves (dev split; target rate, invalid = "
        "failure; shaded 95% CI over trios)",
        fontsize=9,
        color=INK_2,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def position_plot(split_result: dict, split: str, path: Path) -> None:
    primary = split_result["primary"]
    entries = {
        b["condition"]: b
        for b in split_result["biases"]
        if b["task"] == "main" and primary.get(b["condition"]) == b["checkpoint"]
    }
    conditions = [c for c in CONDITIONS if c in entries]
    fig, axes = _figure(1, len(conditions), 2.4 * len(conditions), 2.4)
    for ax, condition in zip(axes, conditions, strict=True):
        _style(ax, grid_axis="y")
        shares = entries[condition]["letter_share"]
        for i, letter in enumerate("ABC"):
            e = shares[letter]
            if e["estimate"] is None:
                continue
            ax.bar(i, e["estimate"], width=0.45, color=CONDITION_COLORS[condition])
            if e["ci_low"] is not None:
                ax.plot([i, i], [e["ci_low"], e["ci_high"]], color=INK_2, linewidth=1.0)
        ax.axhline(CHANCE, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(3), ["A", "B", "C"], color=INK)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(condition, fontsize=9, color=INK, loc="left")
    fig.suptitle(
        f"Main task: share of valid answers by option letter, {split} split "
        "(line = 1/3; whiskers = 95% CI)",
        fontsize=9,
        color=INK_2,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path)


def make_plots(results: dict, out_dir: Path) -> dict[str, str]:
    """Write every plot the available data supports. Returns name -> path relative to
    the report folder."""
    made: dict[str, str] = {}

    def rel(path: Path) -> str:
        return f"{out_dir.name}/{path.name}"

    split = results["headline_split"]
    split_result = results["splits"].get(split)
    if split_result and split_result["comparison"]:
        p = out_dir / f"conditions_{split}.png"
        conditions_plot(split_result, split, p)
        made["conditions"] = rel(p)
    if split_result and split_result["composition"].get("available"):
        p = out_dir / f"main_differences_{split}.png"
        threshold = results["settings"]["decision_rules"]["composition_min_gain_over_singles"]
        contrasts_plot(split_result["composition"], threshold, split, p)
        made["main_differences"] = rel(p)
    if split_result and any(
        b["task"] == "main" and split_result["primary"].get(b["condition"]) == b["checkpoint"]
        for b in split_result["biases"]
    ):
        p = out_dir / f"position_bias_{split}.png"
        position_plot(split_result, split, p)
        made["position_bias"] = rel(p)
    curves = results.get("dev_curves")
    selection = results.get("selection") or {}
    for corpus in ("belief", "goal"):
        if curves and curves.get(corpus):
            p = out_dir / f"dev_curves_{corpus}.png"
            selected = (selection.get(corpus) or {}).get("checkpoint")
            dev_curve_plot(curves, corpus, selected, p)
            made[f"dev_curves_{corpus}"] = rel(p)
    return made
