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
    "MiniLM": "D",
}

# Shading for the pre-prompt-engineering baseline region (pruned taxonomy).
BASELINE_SHADE = {"color": "#cccccc", "alpha": 0.18}

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
