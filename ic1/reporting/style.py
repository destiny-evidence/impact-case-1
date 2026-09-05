"""Shared matplotlib styling — one place so slides and the docs site match.

Import ``apply_style()`` once before plotting. Colours are keyed by semantic
role (method lanes, metrics) rather than by hue so the whole figure set stays
internally consistent.
"""

from __future__ import annotations

import matplotlib as mpl

# Method lanes (used across timeline / matrix / cost figures).
METHOD_COLORS = {
    "Flat": "#1f77b4",       # blue
    "Top-down": "#d62728",   # red
    "Semantic": "#7f7f7f",   # grey (baseline comparator)
}

# Model encoded as marker shape (works for non-consecutive / single runs, where
# a connecting line can't). Metric is the subplot; method is colour.
MODEL_MARKERS = {
    "luna": "o",
    "sol": "s",
    "kimi": "^",
    "deepseek": "v",
    "opus": "P",
    "terra": "X",
    "4o-mini": "*",
    "MiniLM": "D",
}

# In/out operating points (colour dimension in that figure, in place of method).
MODE_COLORS = {
    "high recall": "#2ca02c",     # green — inclusive
    "best balance": "#1f77b4",    # blue — headline operating point
    "high precision": "#d62728",  # red — strict
}

# Per-model colours for the cost/performance scatter.
MODEL_COLORS = {
    "luna": "#1f77b4",
    "sol": "#d62728",
    "terra": "#ff7f0e",
    "opus": "#9467bd",
    "deepseek": "#8c564b",
    "kimi": "#e377c2",
    "4o-mini": "#7f7f7f",
}

# Shading for the pre-prompt-engineering baseline region (pruned taxonomy).
BASELINE_SHADE = {"color": "#cccccc", "alpha": 0.18}

# Shading tints for dev/validation/test cycle spans (in/out figure).
CYCLE_SHADE = {
    "dev": {"color": "#cccccc", "alpha": 0.12},
    "validation": {"color": "#e8a33d", "alpha": 0.22},   # orange — go/no-go gate
    "test": {"color": "#9b8fd4", "alpha": 0.22},         # lavender — final measurement
    "baseline": {"color": "#cccccc", "alpha": 0.18},
}

# Human-annotation decisions (counts figures).
DECISION_COLORS = {
    "include": "#2ca02c",   # green
    "exclude": "#d62728",   # red
    "missing": "#bbbbbb",   # grey — abstentions
}

# Coder-set size -> colour (composition figure: what the 5,000 is made of).
SETSIZE_COLORS = {3: "#1f77b4", 4: "#ff7f0e"}

# Screening raster: exclude / include cell colours and the unassigned background.
RASTER_COLORS = {"exclude": "#d62728", "include": "#2ca02c", "empty": "#f2f2f2"}
# Vote-split strip colormap name (white = unanimous, saturated = evenly split).
RASTER_SPLIT_CMAP = "Purples"

# Landis & Koch agreement bands (kappa), for reference lines/shading.
KAPPA_BANDS = [
    (0.0, "none"), (0.20, "slight"), (0.40, "fair"),
    (0.60, "moderate"), (0.80, "substantial"), (1.01, "almost perfect"),
]

# Prompt-churn categories (additions up / deletions down encode direction;
# colour encodes which prompt was edited).
CHURN_COLORS = {
    "taxonomy": "#1b9e77",  # teal
    "system": "#7570b3",    # purple
}

# Figure sizes (inches). Print = docs/PDF column; slide = 16:9-friendly.
FIGSIZE = {
    "print": (7.0, 4.3),
    "slide": (10.0, 5.6),
    "wide": (11.0, 4.5),
    "square": (7.2, 6.4),
}


def apply_style() -> None:
    """Set global rcParams. Idempotent; call at the top of build_figures."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
    })
