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
    BASELINE_SHADE,
    CHURN_COLORS,
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


def _draw_metric_panel(
    ax: Axes, sub: pd.DataFrame, metric: str, baseline_split: float | None,
    *, baseline_text: bool = False,
) -> None:
    """Draw one F1 metric panel (shaded baseline + per-(method,model) series)."""
    if baseline_split is not None and (sub.vocab == "pruned").any():
        ax.axvspan(-0.5, baseline_split, zorder=0, **BASELINE_SHADE)
        if baseline_text:
            ax.text(
                (-0.5 + baseline_split) / 2, 0.04, "baseline\n(pruned taxonomy)",
                ha="center", va="bottom", fontsize=8, color="#555555",
                transform=ax.get_xaxis_transform(),
            )
    for (method_label, model_short), g in sub.groupby(
        ["method_label", "model_short"], sort=False
    ):
        _plot_broken_series(
            ax, g.sort_values("x"), metric,
            color=METHOD_COLORS.get(method_label, "#333333"),
            marker=MODEL_MARKERS.get(model_short, "o"),
        )
    ax.set_ylabel(f"{metric}-F1")
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


def plot_iteration_timeline(
    runs: pd.DataFrame,
    *,
    churn: pd.DataFrame | None = None,
    metrics: tuple[str, ...] = ("micro", "macro"),
    label_overrides: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """The story of the prompt-engineering work: F1 across dated iterations.

    All models on one shared chronological axis. Metric is the subplot (micro on
    top, macro below), method is colour, model is marker shape. The pruned-
    taxonomy baseline region (before prompt engineering) is shaded; trajectory
    lines connect only consecutive same-(method, model) runs.

    If ``churn`` (from ``load_taxonomy_prompt_churn``) is given, a third panel of
    per-step prompt edits is added below, sharing the x-axis so each edit lines
    up under the F1 movement it may explain.
    """
    sub = runs.sort_values("ts").reset_index(drop=True).copy()
    sub["x"] = range(len(sub))

    # Boundary of the contiguous leading baseline block (pruned taxonomy).
    edited_x = sub.loc[sub.vocab == "edited", "x"]
    baseline_split = (edited_x.min() - 0.5) if not edited_x.empty else None

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
        _draw_metric_panel(ax, sub, metric, baseline_split, baseline_text=ax_i == 0)

    churn_ax = None
    if churn is not None:
        churn_ax = axes[len(metrics)]
        # Align churn rows to the timeline's run order so bars sit under steps.
        cols = ["tax_add", "tax_del", "sys_add", "sys_del"]
        churn_ord = churn.set_index("run").reindex(sub.run).reset_index()
        churn_ord[cols] = churn_ord[cols].fillna(0)
        _draw_churn(churn_ax, churn_ord)

    axes[-1].set_xticks(sub.x)
    axes[-1].set_xticklabels(
        [_prettify(lbl, label_overrides) for lbl in sub.label],
        rotation=35, ha="right",
    )
    axes[0].set_title(
        title or "Prompt-engineering iterations — taxonomy classification"
    )

    # Legends (right margin): method = colour, model = marker shape.
    method_handles = [
        Line2D([0], [0], color=c, linewidth=2.5, label=m)
        for m, c in METHOD_COLORS.items()
        if m in sub.method_label.values
    ]
    model_handles = [
        Line2D([0], [0], color="#333333", marker=mk, linestyle="none",
               markersize=8, label=m)
        for m, mk in MODEL_MARKERS.items()
        if m in sub.model_short.values
    ]
    # Figure-level "outside" legends: constrained_layout reserves margin for
    # these, so long labels ("Top-down") are never clipped.
    fig.legend(handles=method_handles, title="Method", loc="outside right upper")
    fig.legend(handles=model_handles, title="Model", loc="outside right lower")
    # Churn legend sits inside its panel, over the empty baseline region.
    if churn_ax is not None:
        churn_ax.legend(handles=_churn_handles(), title="Edited",
                        loc="upper left", fontsize=8, title_fontsize=9)
    return fig
