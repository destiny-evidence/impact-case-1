"""Render metric DataFrames to committed Markdown fragments for the docs site.

Same principle as the figures: generate the artifact locally (where the data
lives) and commit it, so the mystmd build needs no data or Python — it just
``{include}``s the fragment.
"""

from __future__ import annotations

import pandas as pd


def metrics_markdown(
    df: pd.DataFrame, *, label_col: str = "Operating point", decimals: int = 2,
    bold_max: bool = True,
) -> str:
    """GFM table with the maximum in each numeric column bolded.

    ``bold_max`` bolds the best cell per metric column (e.g. Precision in the
    high-precision row, Recall in the high-recall row).
    """
    numeric = [c for c in df.columns if c != label_col]
    maxima = {c: df[c].max() for c in numeric}

    def cell(col: str, val: float) -> str:
        s = f"{val:.{decimals}f}"
        return f"**{s}**" if bold_max and abs(val - maxima[col]) < 1e-9 else s

    lines = [
        "| " + " | ".join([label_col, *numeric]) + " |",
        "|" + "|".join(["---"] * (len(numeric) + 1)) + "|",
    ]
    for _, r in df.iterrows():
        cells = [str(r[label_col])] + [cell(c, r[c]) for c in numeric]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def comparison_markdown(df: pd.DataFrame, *, decimals: int = 2) -> str:
    """System comparison: point estimate + 95% HDI per metric, plus cost.

    Consumes ``load_inout_comparison``. Metric cells read ``0.72 [0.66–0.78]``;
    the cost column extrapolates the LLM run's per-doc cost to the full corpus
    (the cascade pays it only on the fraction the filter forwards).
    """
    def ci(r: pd.Series, m: str) -> str:
        return (
            f"{r[m]:.{decimals}f} "
            f"[{r[f'{m}_lo']:.{decimals}f}–{r[f'{m}_hi']:.{decimals}f}]"
        )

    def money(v: float) -> str:
        if v >= 1e6:
            return f"${v / 1e6:.1f}M"
        if v >= 1e3:
            return f"${v / 1e3:.0f}k"
        return f"${v:.0f}"

    cols = ["System", "Precision", "Recall", "F1", "F2", "Sent to LLM", "Corpus cost"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join([
            str(r.system), ci(r, "precision"), ci(r, "recall"),
            ci(r, "f1"), ci(r, "f2"),
            f"{r.prop_to_llm * 100:.0f}%", money(r.corpus_cost),
        ]) + " |")
    return "\n".join(lines) + "\n"
