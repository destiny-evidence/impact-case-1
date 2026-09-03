"""Method x model grid of scoreable micro/macro-F1 for the standardised runs.

Reads every completed experiment, identifies its (method, model) from the
snapshotted config.yaml, keeps only the standardised matrix runs (votes>1 on the
edited TTL by default), takes the latest run per cell, and prints a grid so the
2-method x 3-model comparison is readable at a glance.

Scores are over SCOREABLE concepts only (>=1 gold positive) — the 0-gold
concepts are unevaluable here and would distort both micro (extra FP) and macro
(forced zeros). See the compare script's same filter.

Usage:
    python matrix_scores.py                 # votes>1 + edited-TTL runs (the matrix)
    python matrix_scores.py --all           # every completed run
    python matrix_scores.py --metric macro  # grid shows macro-F1 (default: micro)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

EXP_DIR = Path("data-extraction-experiments")


def _scoreable(df: pd.DataFrame) -> pd.DataFrame:
    gold = df.groupby("attribute_label").human_extraction.apply(
        lambda s: s.fillna(False).astype(bool).sum()
    )
    return df[df.attribute_label.isin(gold[gold >= 1].index)]


def _scores(comparison: Path) -> dict:
    df = _scoreable(pd.read_csv(comparison))
    yt = df.human_extraction.fillna(False).astype(int)
    yp = df.llm_extraction.fillna(False).astype(int)
    macro = (
        df.groupby("attribute_label")
        .apply(
            lambda a: f1_score(
                a.human_extraction.fillna(False).astype(int),
                a.llm_extraction.fillna(False).astype(int),
                zero_division=0,
            ),
            include_groups=False,
        )
        .mean()
    )
    return {
        "micro": f1_score(yt, yp, zero_division=0),
        "macro": macro,
        "P": precision_score(yt, yp, zero_division=0),
        "R": recall_score(yt, yp, zero_division=0),
        "tp": int(((yt == 1) & (yp == 1)).sum()),
        "fp": int(((yt == 0) & (yp == 1)).sum()),
        "fn": int(((yt == 1) & (yp == 0)).sum()),
    }


def collect(*, matrix_only: bool) -> pd.DataFrame:
    rows = []
    for d in sorted(EXP_DIR.iterdir()):
        comp = d / "goldstandard_llm_comparison.csv"
        cfg = d / "config.yaml"
        if not (comp.exists() and cfg.exists()):
            continue
        c = yaml.safe_load(cfg.read_text())
        method = c.get("method", "llm")
        model = c.get("model", "?")
        votes = c.get("votes", 1)
        vocab = str(c.get("vocabulary_path", ""))
        if matrix_only and not (votes > 1 and "edited" in vocab):
            continue
        rows.append({
            "run": d.name, "method": method, "model": model,
            "votes": votes, **_scores(comp),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="include non-matrix runs")
    ap.add_argument("--metric", choices=["micro", "macro"], default="micro")
    args = ap.parse_args()

    df = collect(matrix_only=not args.all)
    if df.empty:
        print("No matching runs yet.")
        return

    # latest run per (method, model) — runs are timestamp-prefixed so name sorts by time
    latest = df.sort_values("run").groupby(["method", "model"], as_index=False).last()

    print(f"Latest run per cell ({len(latest)} of {len(df)} runs); "
          f"grid shows {args.metric}-F1 (P/R below):\n")
    for _, r in latest.iterrows():
        print(f"  {r.method:22} {r.model:14} "
              f"micro {r.micro:.3f}  macro {r.macro:.3f}  "
              f"P {r.P:.3f} R {r.R:.3f}  (tp{r.tp} fp{r.fp} fn{r.fn})  [{r.run}]")

    grid = latest.pivot(index="method", columns="model", values=args.metric)
    print(f"\n=== {args.metric}-F1 grid (method x model) ===")
    print(grid.round(3).to_string())


if __name__ == "__main__":
    main()
