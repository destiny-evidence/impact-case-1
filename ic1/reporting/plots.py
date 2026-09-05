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

from matplotlib.patches import Patch

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
                fmt=marker, color=color, ecolor=color, elinewidth=1.6, capsize=2.5,
                markersize=6, zorder=3,
            )
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("score (95% HDI)")
    ax.set_title(title or "In/out screen — system comparison on held-out test")

    mode_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=8, label=m)
        for m, c in MODE_COLORS.items() if (df["mode"] == m).any()
    ]
    family_handles = [
        Line2D([0], [0], marker=mk, color="#333333", linestyle="none",
               markersize=8, label=f)
        for f, mk in _FAMILY_MARKERS.items() if (df["family"] == f).any()
    ]
    leg1 = ax.legend(handles=mode_handles, title="Operating point", loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=family_handles, title="System", loc="lower right",
              bbox_to_anchor=(1.0, 0.32))
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
    for yi, total, inc in zip(y, counts.n_decided, counts.n_include):
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
    for yi, n in zip(y, comp.n_items):
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
    alternating light-grey spans, labelled (rotated) along the top."""
    lo = 0.0
    for i, (hi, name) in enumerate(KAPPA_BANDS):
        if i == 0:
            lo = hi
            continue
        hi_c = min(hi, 1.0)
        ax.axvspan(lo, hi_c, color="#000000", alpha=0.03 if i % 2 else 0.06, zorder=0)
        ax.text((lo + hi_c) / 2, 0.995, name, ha="center", va="top", fontsize=7,
                rotation=90, color="#888888", zorder=1,
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
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{lbl}  (n={int(n)})" for lbl, n in zip(by_set.label, by_set.n_complete)],
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
    ax.set_title(title or "Inter-coder agreement by coder-set — in/out screen")
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


def plot_coder_f1(
    f1df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Per-coder F1 against the adjudicated value, with average/pooled markers.

    Consumes ``inout_coder_f1`` (per-coder rows plus ``average coder`` / ``pooled``
    summaries). Bars are per coder (ordered by F1); precision and recall ride as
    small markers on each bar, and the two summaries are drawn as reference lines.
    """
    summ = f1df[f1df.coder.isin(["average coder", "pooled"])].set_index("coder")
    per = f1df[~f1df.coder.isin(["average coder", "pooled"])].sort_values("f1")
    fig, ax = plt.subplots(
        figsize=figsize or (7.5, 0.32 * len(per) + 1.6), layout="constrained"
    )
    y = range(len(per))
    ax.barh(y, per.f1, color="#4c78a8", zorder=2, label="F1")
    ax.scatter(per.precision, y, marker="|", s=90, color="#d62728", zorder=3,
               label="precision")
    ax.scatter(per.recall, y, marker="|", s=90, color="#2ca02c", zorder=3,
               label="recall")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c.replace('coder_', 'c')}  (n={int(n)})"
                        for c, n in zip(per.coder, per.n)], fontsize=8)
    ax.set_xlabel("score vs adjudicated value")
    ax.set_xlim(0, 1)
    for name, style in (("pooled", "--"), ("average coder", ":")):
        if name in summ.index:
            ax.axvline(summ.loc[name, "f1"], color="#333333", linestyle=style,
                       linewidth=1.4, zorder=4,
                       label=f"{name} F1 = {summ.loc[name, 'f1']:.2f}")
    ax.set_title(title or "Coder F1 vs adjudicated value — in/out screen")
    ax.legend(loc="lower right", fontsize=8)
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
    ax_s.set_yticks([0]); ax_s.set_yticklabels(["split vote"], fontsize=8)
    ax_g.imshow(gold[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                interpolation="nearest")
    ax_g.set_yticks([0]); ax_g.set_yticklabels(["RESOLVED"], fontsize=8)
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
