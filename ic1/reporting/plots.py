"""Pure plot functions: take a tidy DataFrame, return a matplotlib Figure.

No file I/O and no global state here — rendering/saving is build_figures.py's
job, so every figure can be regenerated identically for both the docs site and
PowerPoint.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from matplotlib.patches import Patch

from ic1.reporting.style import (
    CHURN_COLORS,
    CYCLE_SHADE,
    METHOD_COLORS,
    MODEL_MARKERS,
)


def _prettify(label: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and label in overrides:
        return overrides[label]
    return label.replace("_", " ")


def _plot_broken_series(
    ax: Axes, g: pd.DataFrame, metric: str, *, color: str, marker: str
) -> None:
    """Markers for every run; a plain trajectory line only between consecutive runs.

    Marker shape encodes model, colour encodes method. The connecting line is
    trajectory-only (no encoding) and is broken at non-consecutive x so a
    sparsely-sampled model never draws a segment across the whole plot.
    """
    xs = g.x.to_numpy()
    ys = g[metric].to_numpy()
    ax.scatter(
        xs, ys, s=48, marker=marker, facecolors=color, edgecolors=color,
        zorder=3,
    )
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or xs[i] - xs[i - 1] != 1:
            if i - start >= 2:
                ax.plot(
                    xs[start:i], ys[start:i],
                    color=color, linewidth=1.6, alpha=0.8, zorder=2,
                )
            start = i


def _draw_spans(ax: Axes, spans: list[dict], *, label: bool = False) -> None:
    """Shade x-regions (baseline / dev / validation cycles); optionally label."""
    for s in spans:
        ax.axvspan(s["x0"], s["x1"], zorder=0, **CYCLE_SHADE.get(s["kind"], {}))
        if not label:
            continue
        mid = (s["x0"] + s["x1"]) / 2
        if s["kind"] == "baseline":
            ax.text(mid, 0.04, s["label"], ha="center", va="bottom", fontsize=8,
                    color="#555555", transform=ax.get_xaxis_transform())
        else:
            rot = 90 if (s["x1"] - s["x0"]) < 2.5 else 0
            ax.text(mid, 0.97, s["label"], ha="center", va="top", fontsize=8,
                    color="#555555", rotation=rot,
                    transform=ax.get_xaxis_transform())


def _metric_ylabel(metric: str) -> str:
    return f"{metric}-F1" if metric in ("micro", "macro") else metric


def _draw_metric_panel(
    ax: Axes, sub: pd.DataFrame, metric: str, spans: list[dict],
    color_map: dict[str, str], *, label_spans: bool = False,
) -> None:
    """Draw one metric panel: shaded spans + per-(colour, model) series."""
    _draw_spans(ax, spans, label=label_spans)
    for (color_key, model_short), g in sub.groupby(
        ["method_label", "model_short"], sort=False
    ):
        _plot_broken_series(
            ax, g.sort_values("x"), metric,
            color=color_map.get(color_key, "#333333"),
            marker=MODEL_MARKERS.get(model_short, "o"),
        )
    ax.set_ylabel(_metric_ylabel(metric))
    ax.set_ylim(0, 1)


def _draw_churn(ax: Axes, churn: pd.DataFrame) -> None:
    """Diverging stacked bars: additions up, deletions down; colour = category."""
    x = range(len(churn))
    tax, sys = CHURN_COLORS["taxonomy"], CHURN_COLORS["system"]
    # Additions above zero (taxonomy then system stacked).
    ax.bar(x, churn.tax_add, width=0.7, color=tax, zorder=2)
    ax.bar(x, churn.sys_add, width=0.7, bottom=churn.tax_add, color=sys, zorder=2)
    # Deletions below zero.
    ax.bar(x, -churn.tax_del, width=0.7, color=tax, alpha=0.55, zorder=2)
    ax.bar(x, -churn.sys_del, width=0.7, bottom=-churn.tax_del, color=sys,
           alpha=0.55, zorder=2)
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=3)
    ax.set_ylabel("prompt edits\n(words ▲ add / ▼ del)")


def _churn_handles() -> list[Patch]:
    return [
        Patch(facecolor=CHURN_COLORS["taxonomy"], label="taxonomy prompts"),
        Patch(facecolor=CHURN_COLORS["system"], label="system prompt"),
    ]


def plot_prompt_churn(
    churn: pd.DataFrame,
    *,
    label_overrides: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Per-step edits to the taxonomy prompts and the system prompt.

    Same chronological x-order as the iteration timeline, so bars line up under
    the F1 movement they (may) explain.
    """
    fig, ax = plt.subplots(figsize=figsize or (12.0, 3.6), layout="constrained")
    _draw_churn(ax, churn)
    ax.set_xticks(range(len(churn)))
    ax.set_xticklabels(
        [_prettify(lbl, label_overrides) for lbl in churn.label],
        rotation=35, ha="right",
    )
    ax.set_title(title or "Prompt edits per iteration — taxonomy classification")
    fig.legend(handles=_churn_handles(), title="Edited", loc="outside right upper")
    return fig


def _baseline_spans(sub: pd.DataFrame) -> list[dict]:
    """Taxonomy default: shade the contiguous leading pruned-vocab block."""
    if "vocab" not in sub or not (sub.vocab == "pruned").any():
        return []
    edited_x = sub.loc[sub.vocab == "edited", "x"]
    if edited_x.empty:
        return []
    return [{
        "x0": -0.5, "x1": edited_x.min() - 0.5,
        "label": "baseline\n(pruned taxonomy)", "kind": "baseline",
    }]


def plot_iteration_timeline(
    runs: pd.DataFrame,
    *,
    churn: pd.DataFrame | None = None,
    metrics: tuple[str, ...] = ("micro", "macro"),
    color_map: dict[str, str] | None = None,
    color_title: str = "Method",
    spans: list[dict] | None = None,
    churn_yscale: str = "linear",
    label_overrides: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """The story of the prompt-engineering work: a metric per iteration.

    Metric = subplot, ``method_label`` = colour (via ``color_map``), model =
    marker shape. Multiple rows sharing a ``run`` (e.g. in/out operating modes)
    stack in one x-column. ``spans`` shades regions (baseline / dev-val cycles);
    if omitted, the taxonomy pruned-vocab baseline is shaded by default.

    If ``churn`` is given, a third panel of per-step prompt edits is added below,
    sharing the x-axis so each edit lines up under the metric movement above.
    """
    color_map = color_map or METHOD_COLORS
    sub = runs.copy()

    # One x-column per prompt-state (run); rows of the same run share it.
    steps = (
        sub[["run", "ts", "label"]].drop_duplicates("run")
        .sort_values("ts").reset_index(drop=True)
    )
    steps["x"] = range(len(steps))
    sub = sub.merge(steps[["run", "x"]], on="run")
    if spans is None:
        spans = _baseline_spans(sub)

    n = len(metrics) + (1 if churn is not None else 0)
    height_ratios = [1.0] * len(metrics) + ([0.85] if churn is not None else [])
    fig, axes = plt.subplots(
        n, 1, sharex=True, squeeze=False,
        figsize=figsize or (12.0, 2.9 * len(metrics) + (2.4 if churn is not None else 0) + 1.2),
        gridspec_kw={"height_ratios": height_ratios},
        layout="constrained",
    )
    axes = axes[:, 0]

    for ax_i, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
        _draw_metric_panel(ax, sub, metric, spans, color_map, label_spans=ax_i == 0)

    churn_ax = None
    if churn is not None:
        churn_ax = axes[len(metrics)]
        # Align churn rows to the step order so bars sit under their state.
        cols = ["tax_add", "tax_del", "sys_add", "sys_del"]
        churn_ord = churn.set_index("run").reindex(steps.run).reset_index()
        churn_ord[cols] = churn_ord[cols].fillna(0)
        _draw_churn(churn_ax, churn_ord)
        if churn_yscale == "symlog":
            # One big rewrite (e.g. the "shortened" step) otherwise crushes the
            # rest; symlog keeps small edits legible. Linear near zero.
            churn_ax.set_yscale("symlog", linthresh=100)

    axes[-1].set_xticks(steps.x)
    axes[-1].set_xticklabels(
        [_prettify(lbl, label_overrides) for lbl in steps.label],
        rotation=35, ha="right",
    )
    axes[0].set_title(
        title or "Prompt-engineering iterations — taxonomy classification"
    )

    # Legends (right margin): colour dimension + model marker.
    color_handles = [
        Line2D([0], [0], color=c, linewidth=2.5, label=m)
        for m, c in color_map.items()
        if m in sub.method_label.values
    ]
    model_handles = [
        Line2D([0], [0], color="#333333", marker=mk, linestyle="none",
               markersize=8, label=m)
        for m, mk in MODEL_MARKERS.items()
        if m in sub.model_short.values
    ]
    fig.legend(handles=color_handles, title=color_title, loc="outside right upper")
    fig.legend(handles=model_handles, title="Model", loc="outside right lower")
    if churn_ax is not None:
        churn_ax.legend(handles=_churn_handles(), title="Edited",
                        loc="upper left", fontsize=8, title_fontsize=9)
    return fig
