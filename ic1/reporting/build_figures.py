"""Render every figure to committed files that the docs site and PPT both use.

Local entrypoint (needs the real Python env + experiment data). Writes SVG
(vector, drops into PowerPoint and mystmd) plus a PNG@2x fallback for Office
templates that choke on SVG.

Usage:
    python -m ic1.reporting.build_figures                 # all figures
    python -m ic1.reporting.build_figures --formats svg   # svg only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.figure import Figure

from ic1.reporting.loaders import (
    inout_cycle_spans,
    load_inout_prompt_churn,
    load_inout_runs,
    load_taxonomy_prompt_churn,
    load_taxonomy_runs,
)
from ic1.reporting.plots import plot_iteration_timeline
from ic1.reporting.style import MODE_COLORS, apply_style

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = _REPO_ROOT / "docs" / "figures"


def _save(fig: Figure, domain: str, name: str, formats: tuple[str, ...]) -> None:
    out_dir = FIGURES_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"  wrote {path.relative_to(_REPO_ROOT)}")


def build_taxonomy(formats: tuple[str, ...]) -> None:
    runs = load_taxonomy_runs()
    churn = load_taxonomy_prompt_churn()
    print(f"taxonomy: {len(runs)} completed runs")
    _save(
        plot_iteration_timeline(runs, churn=churn),
        "taxonomy", "iteration_timeline", formats,
    )


def build_inout(formats: tuple[str, ...]) -> None:
    runs = load_inout_runs()
    churn = load_inout_prompt_churn()
    print(f"inout: {runs.run.nunique()} prompt states ({len(runs)} state x mode rows)")
    _save(
        plot_iteration_timeline(
            runs, churn=churn,
            metrics=("precision", "recall"),
            color_map=MODE_COLORS, color_title="Operating mode",
            spans=inout_cycle_spans(runs), churn_yscale="symlog",
            title="Prompt-engineering iterations — in/out relevance screen",
        ),
        "inout", "iteration_timeline", formats,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["taxonomy", "inout", "all"], default="all")
    ap.add_argument("--formats", nargs="+", default=["svg", "png"])
    args = ap.parse_args()

    apply_style()
    formats = tuple(args.formats)
    if args.domain in ("taxonomy", "all"):
        build_taxonomy(formats)
    if args.domain in ("inout", "all"):
        build_inout(formats)


if __name__ == "__main__":
    main()
