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
