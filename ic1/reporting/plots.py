"""Pure plot functions: take a tidy DataFrame, return a matplotlib Figure.

No file I/O and no global state here — rendering/saving is build_figures.py's
job, so every figure can be regenerated identically for both the docs site and
PowerPoint.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from matplotlib.patches import Patch, Rectangle

from matplotlib.colors import ListedColormap

from ic1.reporting.style import (
    CHURN_COLORS,
    CYCLE_SHADE,
    DECISION_COLORS,
    FIGSIZE,
    KAPPA_BANDS,
    METHOD_COLORS,
    MODE_COLORS,
    MODEL_COLORS,
    MODEL_MARKERS,
    RASTER_COLORS,
    RASTER_SPLIT_CMAP,
    SETSIZE_COLORS,
    SPLIT_COLORS,
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


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated models: none is both cheaper and at least as good (F2)."""
    keep = []
    for _, r in df.iterrows():
        dominated = (
            (df.cost_per_doc <= r.cost_per_doc) & (df.f2 >= r.f2)
            & ((df.cost_per_doc < r.cost_per_doc) | (df.f2 > r.f2))
        ).any()
        if not dominated:
            keep.append(r)
    return pd.DataFrame(keep).sort_values("cost_per_doc")


def _money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def plot_cost_performance(
    models: pd.DataFrame,
    *,
    corpus_size: int = 6_000_000,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Accuracy (best-balance F2) vs cost per document, one point per model.

    Log cost axis; the Pareto frontier is drawn and a secondary top axis
    extrapolates to the cost of screening the full corpus. Models priced by
    estimate (hollow markers) are flagged in the legend.
    """
    fig, ax = plt.subplots(figsize=figsize or FIGSIZE["square"], layout="constrained")
    models = models.assign(corpus_cost=models.cost_per_doc * corpus_size)

    frontier = _pareto_frontier(models)
    ax.plot(frontier.corpus_cost, frontier.f2, color="#999999", linestyle="--",
            linewidth=1.2, zorder=1, label="Pareto frontier")

    for _, r in models.iterrows():
        color = MODEL_COLORS.get(r.model_short, "#333333")
        marker = MODEL_MARKERS.get(r.model_short, "o")
        face = "white" if r.estimated else color
        ax.scatter(r.corpus_cost, r.f2, s=140, marker=marker, facecolors=face,
                   edgecolors=color, linewidths=1.8, zorder=3)
        label = f"{r.model_short}{'*' if r.estimated else ''} ({_money(r.corpus_cost)})"
        ax.annotate(label, (r.corpus_cost, r.f2),
                    textcoords="offset points", xytext=(9, 5), fontsize=10,
                    color=color, fontweight="bold")

    ax.set_xlim(0, models.corpus_cost.max() * 1.12)
    ax.set_xlabel(f"cost to screen the {_money(corpus_size)[1:]}-document corpus (USD)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v)))
    ax.set_ylabel("best-balance F2  (recall-weighted)")
    ax.set_ylim(0, 1)
    ax.set_title(title or "Model selection — accuracy vs cost, in/out screen")

    # Secondary axis: the same thing expressed per document.
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: x / corpus_size, lambda x: x * corpus_size))
    secax.set_xlabel("cost per document (USD)")
    secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.3f}"))

    ax.annotate("* cost estimated from published API list prices",
                (0, -0.18), xycoords="axes fraction", fontsize=8, color="#666666")
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_taxonomy_cost_performance(
    df: pd.DataFrame,
    *,
    metric: str = "micro",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Accuracy vs per-document cost for the final prompt set — method × model.

    Consumes ``load_taxonomy_cost_performance``. Colour = classification method
    (``METHOD_COLORS``), marker = model (``MODEL_MARKERS``). Log cost axis;
    keyword/embedding methods make no metered LLM call and are drawn at a
    left-edge "free" position. The Pareto frontier (best accuracy reachable at or
    below each cost) is dashed; hollow markers are priced from list estimates.
    """
    fig, ax = plt.subplots(figsize=figsize or FIGSIZE["square"], layout="constrained")

    paid = df.cost_per_doc[(~df.local) & (df.cost_per_doc > 0)]
    floor = (paid.min() / 4) if len(paid) else 1e-4
    df = df.assign(x=df.cost_per_doc.where(~df.local, floor))

    # Pareto frontier on true cost (local == 0) vs accuracy: cheapest run at each
    # accuracy ceiling, walking from cheap to dear.
    order = df.sort_values(["cost_per_doc", metric], ascending=[True, False])
    front, best = [], -1.0
    for _, r in order.iterrows():
        if r[metric] > best:
            front.append(r)
            best = r[metric]
    frontier = pd.DataFrame(front)
    ax.plot(frontier.x, frontier[metric], color="#999999", linestyle="--",
            linewidth=1.2, zorder=1, label="Pareto frontier")

    for _, r in df.iterrows():
        color = METHOD_COLORS.get(r.method_label, "#333333")
        marker = MODEL_MARKERS.get(r.model_short, "o")
        face = "white" if r.estimated else color
        ax.scatter(r.x, r[metric], s=150, marker=marker, facecolors=face,
                   edgecolors=color, linewidths=1.8, zorder=3)
        cost_lbl = ("free" if r.local
                    else f"${r.cost_per_doc:.3f}{'*' if r.estimated else ''}")
        # Same model appears once per method at near-identical (x, F1); drop the
        # Top-down label below its marker so the pair doesn't overprint.
        dy = -13 if r.method_label == "Top-down" else 6
        ax.annotate(f"{r.model_short} ({cost_lbl})", (r.x, r[metric]),
                    textcoords="offset points", xytext=(9, dy), fontsize=9,
                    color=color, fontweight="bold",
                    va="top" if dy < 0 else "bottom")

    ax.set_xscale("log")
    ax.set_xlim(floor / 1.6, df.x.max() * 2.6)
    ax.set_xlabel("cost per document (USD, log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.3f}"))
    ax.set_ylabel(_metric_ylabel(metric))
    ax.set_ylim(0, 1)
    ax.set_title(title or "Taxonomy classification — accuracy vs cost (latest prompts)")

    ax.axvline(floor, color="#cccccc", linewidth=0.8, linestyle=":", zorder=0)
    ax.annotate("≈ free\n(local)", (floor, 0.02), fontsize=8, color="#888888",
                ha="center", va="bottom")

    method_handles = [
        Line2D([0], [0], marker="s", color=c, linestyle="none", markersize=9,
               label=m)
        for m, c in METHOD_COLORS.items() if (df.method_label == m).any()
    ]
    model_handles = [
        Line2D([0], [0], marker=mk, color="#333333", linestyle="none",
               markersize=8, label=m)
        for m, mk in MODEL_MARKERS.items() if (df.model_short == m).any()
    ]
    leg1 = ax.legend(handles=method_handles, title="Method", loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model", loc="lower right",
              bbox_to_anchor=(1.0, 0.32))
    ax.annotate("* cost estimated from published API list prices",
                (0, -0.16), xycoords="axes fraction", fontsize=8, color="#666666")
    return fig


# System families encoded as marker shape; operating point as colour (MODE_COLORS).
# ML-only has no operating point, so it gets a neutral grey.
_FAMILY_MARKERS = {"LLM only": "o", "ML only": "D", "ML → LLM": "s", "ML or LLM": "^"}
_NO_MODE_COLOR = "#7f7f7f"


def plot_comparison(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "f2"),
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Forest plot of the head-to-head: each metric a y-group, each system a
    point with its 95% HDI bar.

    Consumes ``load_inout_comparison`` (columns ``family``, ``mode``, ``recall``,
    ``recall_lo``, ``recall_hi``, ...). Operating point = colour, system family =
    marker shape; systems are offset within each metric group so intervals don't
    overlap.
    """
    fig, ax = plt.subplots(figsize=figsize or (11.0, 8.5), layout="constrained")
    offsets = np.linspace(0.45, -0.45, len(df))
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1", "f2": "F2"}
    yticks, yticklabels = [], []
    for gi, metric in enumerate(metrics):
        base = gi * 2.6
        yticks.append(base)
        yticklabels.append(labels.get(metric, metric))
        for si, (_, r) in enumerate(df.iterrows()):
            color = MODE_COLORS.get(r["mode"], _NO_MODE_COLOR)
            marker = _FAMILY_MARKERS.get(r["family"], "o")
            ax.errorbar(
                r[metric], base + offsets[si],
                xerr=[[r[metric] - r[f"{metric}_lo"]], [r[f"{metric}_hi"] - r[metric]]],
                fmt=marker, color=color, ecolor=color, elinewidth=2.0, capsize=3.5,
                markersize=9, zorder=3,
            )
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=16)
    ax.tick_params(axis="x", labelsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("score (95% HDI)", fontsize=15)
    ax.set_title(title or "In/out screen — system comparison on held-out test",
                 fontsize=17)

    mode_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=11, label=m)
        for m, c in MODE_COLORS.items() if (df["mode"] == m).any()
    ]
    family_handles = [
        Line2D([0], [0], marker=mk, color="#333333", linestyle="none",
               markersize=11, label=f)
        for f, mk in _FAMILY_MARKERS.items() if (df["family"] == f).any()
    ]
    fig.legend(handles=mode_handles, title="Operating point",
               loc="outside right upper", fontsize=13, title_fontsize=13)
    fig.legend(handles=family_handles, title="System",
               loc="outside right lower", fontsize=13, title_fontsize=13)
    return fig


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
        Patch(facecolor=CHURN_COLORS["taxonomy"], label="attribute prompts"),
        Patch(facecolor=CHURN_COLORS["system"], label="system prompt"),
    ]


def plot_scheme_scores(
    scores: pd.DataFrame,
    *,
    metric: str = "micro",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Per-scheme F1 for one run: horizontal bars, sorted worst -> best.

    ``n`` is the number of scoreable concepts (>=1 gold positive) in the scheme.
    """
    s = scores.sort_values(metric, ascending=True).reset_index(drop=True)
    y = list(range(len(s)))
    bar_c = METHOD_COLORS.get("Flat", "#4c72b0")
    fig, ax = plt.subplots(
        figsize=figsize or (9.0, 0.42 * len(s) + 1.6), layout="constrained"
    )
    ax.barh(y, s[metric], height=0.6, color=bar_c, zorder=2)
    for yi, (v, n) in enumerate(zip(s[metric], s.n_concepts_scoreable)):
        ax.text(min(v + 0.01, 0.98), yi, f"{v:.2f} (n={n})",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(s.scheme_title)
    ax.set_xlim(0, 1)
    ax.set_xlabel(_metric_ylabel(metric))
    ax.set_title(title or f"Per-scheme {_metric_ylabel(metric)} — {s.run.iloc[0]}")
    return fig


def plot_taxonomy_level_scores(
    scores: pd.DataFrame,
    scheme: pd.DataFrame | None = None,
    *,
    metric: str = "micro",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Hierarchy-respecting F1 for one run, one lane per scheme.

    Grey bar = the scheme's rolled-up F1 over ALL its scoreable concepts (from
    ``scheme``, ``load_taxonomy_scheme_scores``). Dots = each internal node's
    LOCAL F1 over its DIRECT children only (``load_taxonomy_level_scores``),
    coloured by depth and sized by scoreable-child count. A dot far left of a
    tall bar is a weak deep branch the rolled-up score would have hidden.
    """
    shades = ["#2f5597", "#6f93c6", "#9db9dc", "#c9d8ee"]

    def shade(lv: int) -> str:
        return shades[min(int(lv), len(shades) - 1)]

    # Order lanes by the rolled-up scheme score if given, else the root group.
    if scheme is not None:
        agg = scheme.set_index("scheme_title")[metric]
    else:
        agg = scores[scores.level == 0].set_index("scheme_title")[metric]
    lanes = list(agg.sort_values().index)
    row = {name: i for i, name in enumerate(lanes)}

    bar_fill, bar_edge = "#d6d6d6", "#a8a8a8"
    fig, ax = plt.subplots(
        figsize=figsize or (9.5, 0.46 * len(lanes) + 1.6), layout="constrained"
    )
    for name in lanes:
        ax.barh(row[name], float(agg[name]), height=0.72, color=bar_fill,
                edgecolor=bar_edge, linewidth=0.8, zorder=1)

    for name in lanes:
        g = scores[scores.scheme_title == name].sort_values(["level", metric])
        k = len(g)
        offs = np.linspace(-0.3, 0.3, k) if k > 1 else np.array([0.0])
        ax.scatter(
            g[metric], row[name] + offs,
            s=24 + 34 * np.sqrt(g.n_concepts_scoreable.to_numpy()),
            c=[shade(lv) for lv in g.level],
            edgecolor="white", linewidth=0.6, zorder=3,
        )

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=9)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(_metric_ylabel(metric))
    ax.set_title(title or f"Per-node {_metric_ylabel(metric)} — {scores.run.iloc[0]}")

    max_lv = int(scores.level.max())
    color_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=shade(lv),
               markeredgecolor="white", markersize=8, label=f"level {lv}")
        for lv in range(max_lv + 1)
    ]
    color_handles.append(Patch(facecolor=bar_fill, edgecolor=bar_edge,
                               label="scheme F1\n(all concepts)"))
    fig.legend(handles=color_handles, title="node depth",
               loc="outside upper right", fontsize=8, title_fontsize=8)
    return fig


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


# ---------------------------------------------------------------------------
# Human-annotation figures (in/out screening effort)
# ---------------------------------------------------------------------------


def plot_coder_counts(
    counts: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Per-coder annotation counts as horizontal stacked bars (exclude / include
    / missing).

    Consumes ``inout_coder_counts``. Coders are ordered by decided volume
    (busiest at top); the stack shows how each coder's load splits between
    exclude and include. Missing (abstention) rows are dropped, so bar length is
    the number of *decided* annotations.
    """
    counts = counts.assign(n_decided=counts.n_include + counts.n_exclude)
    counts = counts.sort_values("n_decided")  # ascending -> busiest on top
    fig, ax = plt.subplots(
        figsize=figsize or (7.5, 0.34 * len(counts) + 1.4), layout="constrained"
    )
    y = range(len(counts))
    left = np.zeros(len(counts))
    for key, col in (("n_exclude", "exclude"), ("n_include", "include")):
        vals = counts[key].to_numpy()
        ax.barh(y, vals, left=left, color=DECISION_COLORS[col], label=col, zorder=2)
        left += vals
    ax.set_yticks(list(y))
    ax.set_yticklabels(counts.username.str.replace("coder_", "c"))
    ax.set_xlabel("decided annotations")
    for yi, total, inc in zip(y, counts.n_decided, counts.n_include, strict=True):
        ratio = inc / total if total else 0.0
        ax.text(total + left.max() * 0.006, yi,
                f"{int(total):,}  ({ratio:.0%} incl)", va="center",
                fontsize=8, color="#333333")
    ax.set_title(title or "Annotations per coder — in/out screen")
    ax.legend(loc="lower right", ncol=2)
    return fig


def plot_coderset_composition(
    comp: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """What the 5,000 items are made of: one horizontal bar per nominal coder-set.

    Consumes ``inout_coderset_composition``. Bars are ordered by item count and
    coloured by set size (3 vs 4 coders); each is labelled with its coder members
    and item count. The title carries the corpus total.
    """
    comp = comp.sort_values("n_items")  # ascending -> largest on top
    fig, ax = plt.subplots(
        figsize=figsize or (8.0, 0.32 * len(comp) + 1.4), layout="constrained"
    )
    y = range(len(comp))
    colors = [SETSIZE_COLORS.get(s, "#7f7f7f") for s in comp["size"]]
    ax.barh(y, comp.n_items, color=colors, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(comp.label, fontsize=8)
    ax.set_xlabel("documents")
    for yi, n in zip(y, comp.n_items, strict=True):
        ax.text(n + comp.n_items.max() * 0.006, yi, f"{int(n):,}", va="center",
                fontsize=8, color="#333333")
    total = int(comp.n_items.sum())
    ax.set_title(title or f"Coder-set composition — {total:,} documents")
    handles = [Patch(facecolor=SETSIZE_COLORS[s], label=f"{s} coders")
               for s in sorted(SETSIZE_COLORS) if (comp["size"] == s).any()]
    ax.legend(handles=handles, loc="lower right", title="Set size")
    return fig


def _kappa_band_shading(ax: Axes) -> None:
    """Shade the Landis & Koch agreement bands along the κ (x) axis as vertical
    alternating light-grey spans, labelled just above the plot area."""
    lo = 0.0
    for i, (hi, name) in enumerate(KAPPA_BANDS):
        if i == 0:
            lo = hi
            continue
        hi_c = min(hi, 1.0)
        ax.axvspan(lo, hi_c, color="#000000", alpha=0.03 if i % 2 else 0.06, zorder=0)
        ax.text((lo + hi_c) / 2, 1.008, name, ha="center", va="bottom", fontsize=9.5,
                color="#888888", zorder=1, clip_on=False,
                transform=ax.get_xaxis_transform())
        lo = hi


def plot_agreement_by_set(
    by_set: pd.DataFrame,
    overall: dict | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Per coder-set Fleiss' kappa as bars, over shaded agreement bands.

    Consumes ``inout_agreement_by_set`` (+ optional ``inout_overall_agreement``
    for a reference line). Sets are ordered by kappa; NaN-kappa sets (single
    category) are dropped. Bar colour = set size; the dashed line marks the
    pooled overall kappa.
    """
    by_set = by_set.dropna(subset=["fleiss"]).sort_values("fleiss").reset_index(drop=True)
    fig, ax = plt.subplots(
        figsize=figsize or (8.5, 0.32 * len(by_set) + 1.6), layout="constrained"
    )
    _kappa_band_shading(ax)
    y = range(len(by_set))
    colors = [SETSIZE_COLORS.get(s, "#7f7f7f") for s in by_set["size"]]
    ax.barh(y, by_set.fleiss, color=colors, zorder=2)
    # Resolved inclusion rate for each set, printed just past the bar end.
    if "resolved_incl_rate" in by_set:
        for yi, k, rate in zip(y, by_set.fleiss, by_set.resolved_incl_rate, strict=True):
            if np.isnan(rate):
                continue
            ax.text(k + 0.015, yi, f"{rate:.0%} incl", va="center", ha="left",
                    fontsize=7, color="#666666", zorder=3)
    ax.set_ylim(-0.6, len(by_set) - 0.4)  # trim matplotlib's default y-margin
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{lbl}  (n={int(n)})"
         for lbl, n in zip(by_set.label, by_set.n_complete, strict=True)],
        fontsize=8,
    )
    ax.set_xlabel("Fleiss' κ")
    ax.set_xlim(min(0.0, by_set.fleiss.min() - 0.02), 1.0)
    handles = [Patch(facecolor=SETSIZE_COLORS[s], label=f"{s} coders")
               for s in sorted(SETSIZE_COLORS) if (by_set["size"] == s).any()]
    if overall is not None and not np.isnan(overall.get("fleiss", np.nan)):
        ax.axvline(overall["fleiss"], color="#333333", linestyle="--", linewidth=1.4,
                   zorder=3)
        handles.append(Line2D([0], [0], color="#333333", linestyle="--",
                              label=f"overall κ = {overall['fleiss']:.2f}"))
    ax.legend(handles=handles, loc="lower right", title="Set size")
    ax.set_title(title or "Inter-coder agreement by coder-set — in/out screen",
                 pad=30)
    return fig


def plot_pairwise_kappa(
    mat: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Heatmap of pairwise Cohen's kappa between coders.

    Consumes ``inout_pairwise_kappa``. Sparse pairs (below the loader's
    ``min_shared`` threshold) and the diagonal are shown blank/greyed. Diverging
    colour map centred at 0 so disagreement (negative) reads distinctly.
    """
    labels = [c.replace("coder_", "c") for c in mat.index]
    data = mat.to_numpy(dtype=float)
    off = data.copy()
    np.fill_diagonal(off, np.nan)
    fig, ax = plt.subplots(figsize=figsize or (8.5, 7.2), layout="constrained")
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(off, cmap=cmap, vmin=-0.4, vmax=0.8, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = off[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#222222")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cohen's κ")
    ax.set_title(title or "Pairwise coder agreement (Cohen's κ) — in/out screen")
    return fig


def plot_unanimous_funnel(
    disp: dict,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Funnel plot: each coder-set's unanimous inclusion rate vs its n, with
    binomial control limits around the pooled rate.

    Consumes ``inout_unanimous_dispersion``. Points outside the funnel vary more
    than sampling chance allows — candidate block-level include/exclude biases
    baked into the (unchecked) unanimous documents. Sets beyond the 99.8% limit
    are red; sets beyond 2 SE are labelled. The dispersion statistics are printed
    in-panel.
    """
    df = disp["sets"]
    p = disp["pooled"]
    pr = disp.get("pooled_resolved")
    fig, ax = plt.subplots(figsize=figsize or (8.6, 6.2), layout="constrained")

    def _funnel(center: float, nmin: float, nmax: float, color: str, name: str) -> None:
        nn = np.linspace(nmin * 0.9, nmax * 1.05, 200)
        for z, style in ((1.96, "--"), (3.09, ":")):
            se = np.sqrt(center * (1 - center) / nn)
            ax.plot(nn, center + z * se, color=color, linestyle=style, linewidth=1,
                    zorder=1)
            ax.plot(nn, np.clip(center - z * se, 0, 1), color=color, linestyle=style,
                    linewidth=1, zorder=1)
        ax.axhline(center, color=color, linewidth=1.3, zorder=1,
                   label=f"pooled {name} {center:.1%}")

    # Consensus funnel (grey) around the unanimous rate; resolved funnel (blue)
    # around the truth rate — each uses its own denominator (n_unan vs n_all).
    _funnel(p, df.n_unan.min(), df.n_unan.max(), "#999999", "unanimous")
    has_res = "resolved_rate" in df and pr is not None
    if has_res:
        _funnel(pr, df.n_all.min(), df.n_all.max(), "#6f9fc8", "resolved")
        # Link each set's two rates; hollow dot = resolved (truth) at its n_all.
        for _, r in df.iterrows():
            ax.plot([r.n_unan, r.n_all], [r.rate, r.resolved_rate],
                    color="#cccccc", linewidth=0.7, zorder=2)
        ax.scatter(df.n_all, df.resolved_rate, s=46, facecolors="white",
                   edgecolors="#2f5d86", linewidths=1.2, zorder=4)

    colors = ["#d62728" if abs(z) > 3.09 else SETSIZE_COLORS.get(s, "#888888")
              for z, s in zip(df.z, df["size"], strict=True)]
    ax.scatter(df.n_unan, df.rate, s=75, c=colors, edgecolors="white",
               linewidths=1.0, zorder=5)
    for _, r in df.iterrows():
        if abs(r.z) > 2:
            ax.annotate(r.label, (r.n_unan, r.rate), xytext=(7, 4),
                        textcoords="offset points", fontsize=7, color="#333333")

    ax.set_xlabel("documents in set (n)  ·  ● at unanimous-doc count, ○ at all-doc count")
    ax.set_ylabel("inclusion rate  (● unanimous consensus / ○ resolved truth)")
    ax.set_ylim(bottom=min(-0.005, df.rate.min() - 0.01))
    ax.set_title(title or "Unanimous consensus vs resolved truth by coder-set")
    handles = [
        Line2D([0], [0], color="#999999", lw=1.3, label=f"unanimous funnel ({p:.1%})"),
    ]
    if has_res:
        handles.append(Line2D([0], [0], color="#6f9fc8", lw=1.3,
                              label=f"resolved funnel ({pr:.1%})"))
    handles += [
        Line2D([0], [0], color="#777777", ls="--", lw=1, label="95% limits"),
        Line2D([0], [0], color="#777777", ls=":", lw=1, label="99.8% limits"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    disp_txt = f"dispersion φ:  consensus {disp['phi']:.2f} (p={disp['pval']:.1e})"
    if has_res and "resolved_phi" in disp:
        disp_txt += f"   ·   resolved {disp['resolved_phi']:.2f} (p={disp['resolved_pval']:.2f})"
    ax.annotate(
        disp_txt + "\nred ● = beyond 99.8% consensus limits",
        (0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=8, color="#555555",
    )
    return fig


def plot_data_splits(
    splits: dict,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Swimlane of which annotated documents each pipeline stage used.

    Consumes ``load_inout_splits``. A header bar shows the disjoint train /
    validation / test partition (widths ∝ counts); below it one lane per consumer
    (ML classifier, LLM screening) draws boxes over the splits it drew from. Every
    box label sits *above* its box (coloured by split) so it can run wider than a
    narrow box without clipping. The LLM prompt-development set is a subset of
    validation, placed at validation's right edge next to the held-out test.
    """
    tr, va, te, dev = splits["train"], splits["validation"], splits["test"], splits["llm_dev"]
    b0, b1, b2, b3 = 0, tr, tr + va, tr + va + te
    fig, ax = plt.subplots(figsize=figsize or (8.0, 3.9), layout="constrained")

    def bar(x0: float, x1: float, y: float, h: float, key: str) -> None:
        ax.add_patch(Rectangle((x0, y), x1 - x0, h, facecolor=SPLIT_COLORS[key],
                               edgecolor="white"))

    def label_above(x0: float, x1: float, ytop: float, txt: str, key: str,
                    *, ha: str = "center", bold: bool = False) -> None:
        x = {"center": (x0 + x1) / 2, "right": x1, "left": x0}[ha]
        ax.text(x, ytop + 0.07, txt, ha=ha, va="bottom", clip_on=False,
                fontsize=9.5 if bold else 8.5, color=SPLIT_COLORS[key],
                fontweight="bold" if bold else "normal")

    # Header partition bar. Labels above, name over count on two lines so the
    # narrow validation/test segments' labels stay narrow enough not to collide.
    hy, hh = 3.1, 0.55
    for x0, x1, key, name, n in [
        (b0, b1, "train", "Train", tr),
        (b1, b2, "validation", "Validation", va),
        (b2, b3, "test", "Test — held out", te),
    ]:
        bar(x0, x1, hy, hh, key)
        label_above(x0, x1, hy + hh, f"{name}\n({n:,})", key, bold=True)

    def lane(y: float, label: str, boxes: list[tuple]) -> None:
        h = 0.5
        ax.text(-b3 * 0.015, y + h / 2, label, ha="right", va="center",
                fontsize=10, fontweight="bold")
        for x0, x1, key, txt, ha in boxes:
            bar(x0, x1, y, h, key)
            label_above(x0, x1, y + h, txt, key, ha=ha)

    lane(1.95, "ML classifier", [
        (b0, b1, "train", "fit", "center"),
        (b1, b2, "validation", "select / threshold", "center"),
        (b2, b3, "test", "evaluate", "center"),
    ])
    lane(0.85, "LLM screening", [
        (b2 - dev, b2, "validation", f"prompt dev ({dev})", "right"),
        (b2, b3, "test", "evaluate", "center"),
    ])

    ax.axvline(b2, color="#333333", linewidth=0.8, linestyle=":")
    ax.set_xlim(-b3 * 0.16, b3 + b3 * 0.01)
    ax.set_ylim(0.7, 4.0)
    ax.axis("off")
    ax.set_title(title or "Train / test / validation splits for ML and LLM pipelines",
                 fontweight="bold", pad=24)

    return fig


def plot_screening_raster(
    raster: dict,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Every document (x) x every coder (y) as a red/green screening raster.

    Consumes ``inout_screening_raster``. Cells are green (include) / red
    (exclude) / grey (unassigned or abstained). Documents are grouped by coder-
    set and sorted within each set by include-fraction, so each block runs
    unanimous-exclude -> split -> unanimous-include and disagreement shows as a
    fringe at the block's right edge. Two thin strips sit on top: the per-
    document vote *split* (white = unanimous, saturated = evenly split) and the
    adjudicated *RESOLVED* value.
    """
    matrix, coders = raster["matrix"], raster["coders"]
    gold, split = raster["gold"], raster["split"]
    n = raster["n_items"]

    cmap = ListedColormap([RASTER_COLORS["exclude"], RASTER_COLORS["include"]])
    cmap.set_bad(RASTER_COLORS["empty"])

    fig, (ax_s, ax_g, ax_m) = plt.subplots(
        3, 1, sharex=True,
        figsize=figsize or (14.0, 0.30 * len(coders) + 2.2),
        height_ratios=[1, 1, 2 * len(coders)],
        layout="constrained",
    )
    ax_s.imshow(split[None, :], aspect="auto", cmap=RASTER_SPLIT_CMAP,
                vmin=0, vmax=1, interpolation="nearest")
    ax_s.set_yticks([0])
    ax_s.set_yticklabels(["split vote"], fontsize=8)
    ax_g.imshow(gold[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest")
    ax_g.set_yticks([0])
    ax_g.set_yticklabels(["RESOLVED"], fontsize=8)
    ax_m.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest")
    ax_m.set_yticks(range(len(coders)))
    ax_m.set_yticklabels([c.replace("coder_", "c") for c in coders], fontsize=7)
    for ax in (ax_s, ax_g, ax_m):
        ax.grid(False)
    ax_m.set_xlabel(
        f"{n:,} documents — grouped by coder-set, sorted within set by "
        "include-fraction (disagreement at each block's right edge)"
    )

    handles = [
        Patch(facecolor=RASTER_COLORS["include"], label="include"),
        Patch(facecolor=RASTER_COLORS["exclude"], label="exclude"),
        Patch(facecolor=RASTER_COLORS["empty"], label="not assigned"),
    ]
    ax_m.legend(handles=handles, loc="upper right", ncol=3, fontsize=8,
                framealpha=0.9)
    ax_s.set_title(title or "Human screening raster — in/out relevance")
    return fig


# Iso-Fβ contour styling shared by the coder PR charts.
_FBETA_STYLE = ((1, "#bbbbbb", "dashed"), (2, "#b07aa1", "dotted"))


def _draw_fbeta_contours(ax: Axes, levels: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> None:
    """Iso-Fβ contours: Fβ = (1+β²)PR / (β²P + R). F1 (grey) is symmetric in P/R;
    F2 (mauve) weights recall higher, so its curves bow toward the recall axis —
    a high-recall / low-precision point sits on a much higher F2 than F1."""
    grid = np.linspace(0.001, 1, 300)
    R, P = np.meshgrid(grid, grid)
    for beta, color, style in _FBETA_STYLE:
        Fb = (1 + beta**2) * P * R / (beta**2 * P + R)
        cs = ax.contour(R, P, Fb, levels=list(levels), colors=color, linewidths=0.9,
                        linestyles=style, zorder=1)
        ax.clabel(cs, fmt=lambda v, b=beta: f"F{b}={v:.1f}", fontsize=7, colors=color)


def _fbeta_contour_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#bbbbbb", linestyle="dashed", label="iso-F1"),
        Line2D([0], [0], color="#b07aa1", linestyle="dotted",
               label="iso-F2 (recall-weighted)"),
    ]


def plot_coder_pr(
    f1df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    label: str = "code",
) -> Figure:
    """Precision–recall scatter of coders vs the adjudicated value, over iso-F1
    contours.

    ``label`` controls the in-marker text: ``"code"`` (default) shows the last
    two characters of the anonymised ``coder_NN`` id; ``"name"`` shows the
    coder's initials from a ``firstname.surname`` username (for private renders).

    Consumes ``inout_coder_f1``. One point per coder (recall x, precision y),
    sized by the number of documents screened; the ``pooled`` and ``average
    coder`` summaries are drawn as distinct markers. Faint curves of constant F1
    let you read each coder's F1 off its position, so no separate F1 encoding is
    needed. Geometry tells the calibration story: over-includers sit bottom-right
    (high recall, low precision), under-includers top-left.
    """
    summ = f1df[f1df.coder.isin(["average coder", "pooled"])].set_index("coder")
    per = f1df[~f1df.coder.isin(["average coder", "pooled"])]
    fig, ax = plt.subplots(figsize=figsize or FIGSIZE["square"], layout="constrained")

    _draw_fbeta_contours(ax)

    # Marker size ∝ documents screened (sqrt so area reads proportionally);
    # floored so the 2-char coder label always fits inside.
    nmax = per.n.max()
    def size(n: float) -> float:
        return 90 + 230 * (n / nmax) ** 0.5

    ax.scatter(per.recall, per.precision, s=[size(n) for n in per.n],
               facecolors="#4c78a8", edgecolors="white", linewidths=1.0,
               alpha=0.85, zorder=3)
    def _mark(coder: str) -> str:
        if label == "name":
            return "".join(p[0] for p in coder.split(".") if p)[:2].upper()
        return coder.split("_")[-1][-2:]
    for _, r in per.iterrows():
        ax.annotate(_mark(r.coder), (r.recall, r.precision),
                    textcoords="offset points", xytext=(0, 0), ha="center",
                    va="center", fontsize=6, color="white", zorder=4,
                    fontweight="bold")

    marks = {"pooled": ("*", "#d62728", "pooled"),
             "average coder": ("D", "#333333", "average coder")}
    mark_handles = []
    for name, (mk, col, lbl) in marks.items():
        if name in summ.index:
            s = summ.loc[name]
            ax.scatter(s.recall, s.precision, marker=mk, s=260 if mk == "*" else 120,
                       facecolors=col, edgecolors="white", linewidths=1.2, zorder=5)
            mark_handles.append(Line2D([0], [0], marker=mk, color=col,
                                       linestyle="none", markersize=11 if mk == "*"
                                       else 8, label=lbl))

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("recall  (of the relevant papers, how many the coder caught)")
    ax.set_ylabel("precision  (of the coder's includes, how many were relevant)")
    ax.set_aspect("equal")
    ax.set_title(title or "Coder precision vs recall against adjudicated value")
    ax.legend(handles=_fbeta_contour_handles() + mark_handles, loc="lower left",
              fontsize=8.5)
    ax.annotate("marker size ∝ documents screened", (0.98, 0.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, color="#666666")
    return fig


def plot_coder_model_pr(
    f1df: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Coders and the automated systems together in precision–recall space.

    Overlays the head-to-head systems (``load_inout_comparison`` — LLM only /
    ML only / ML→LLM / ML or LLM at each operating point; marker = family,
    colour = operating point) on the human coders (faded blue circles, from
    ``inout_coder_f1``), over the same iso-F1/F2 contours. Lets you read where the
    pipelines land relative to the human cloud. Note the two are scored against
    the same adjudicated gold but over *different* document sets (coders over
    their assignments; systems over the 648-doc held-out test set).
    """
    summ = f1df[f1df.coder.isin(["average coder", "pooled"])].set_index("coder")
    per = f1df[~f1df.coder.isin(["average coder", "pooled"])]
    fig, ax = plt.subplots(figsize=figsize or (10.6, 7.4), layout="constrained")

    _draw_fbeta_contours(ax)

    # Human coders: a faded reference cloud (numbered), plus the two summaries.
    ax.scatter(per.recall, per.precision, s=90, facecolors="#4c78a8",
               edgecolors="white", linewidths=0.8, alpha=0.45, zorder=3)
    for _, r in per.iterrows():
        ax.annotate(r.coder.split("_")[-1][-2:], (r.recall, r.precision),
                    textcoords="offset points", xytext=(0, 0), ha="center",
                    va="center", fontsize=5.5, color="white", zorder=4)
    for name, mk, col in (("pooled", "*", "#d62728"), ("average coder", "D", "#333333")):
        if name in summ.index:
            s = summ.loc[name]
            ax.scatter(s.recall, s.precision, marker=mk, s=260 if mk == "*" else 110,
                       facecolors=col, edgecolors="white", linewidths=1.2, zorder=5,
                       alpha=0.55)

    # Automated systems: marker = family, colour = operating point.
    for _, r in comparison.iterrows():
        color = MODE_COLORS.get(r["mode"], _NO_MODE_COLOR)
        marker = _FAMILY_MARKERS.get(r["family"], "o")
        ax.scatter(r["recall"], r["precision"], marker=marker, s=150,
                   facecolors=color, edgecolors="black", linewidths=1.0, zorder=6)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("recall  (of the relevant papers, how many were caught)")
    ax.set_ylabel("precision  (of the includes, how many were relevant)")
    ax.set_aspect("equal")
    ax.set_title(title or "Coders vs automated systems — precision/recall")

    # Legends outside on the right so the data area stays clean.
    human_handles = _fbeta_contour_handles() + [
        Line2D([0], [0], marker="o", color="#4c78a8", linestyle="none", alpha=0.5,
               markersize=8, label="individual coder"),
        Line2D([0], [0], marker="*", color="#d62728", linestyle="none",
               markersize=12, label="pooled coder"),
        Line2D([0], [0], marker="D", color="#333333", linestyle="none",
               markersize=7, label="average coder"),
    ]
    system_handles = [
        Line2D([0], [0], marker=mk, color="#555555", linestyle="none", markersize=8,
               label=f)
        for f, mk in _FAMILY_MARKERS.items() if (comparison["family"] == f).any()
    ] + [
        Line2D([0], [0], marker="s", color=c, linestyle="none", markersize=8, label=m)
        for m, c in MODE_COLORS.items() if (comparison["mode"] == m).any()
    ]
    leg1 = fig.legend(handles=human_handles, loc="outside right upper", fontsize=8,
                      title="Humans (& contours)")
    fig.add_artist(leg1)  # else the second fig.legend replaces this one
    fig.legend(handles=system_handles, loc="outside right lower", fontsize=8,
               title="Systems\n(shape = method,\ncolour = operating point)")
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
