"""Aggregate results of the blind re-screen (label-noise experiment, #28) for the docs.

Wraps ic1.annotation.reannotation.analyse. The re-screen export is private, so a checkout without it raises
FileNotFoundError. Only aggregate tables leave this module; per-record and per-reviewer results stay private.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ic1.annotation.reannotation.analyse import (
    CELLS,
    consistency_table,
    corrected_metrics,
    error_composition,
    load_population,
    load_rescreen,
    load_system_predictions,
    overturn_table,
    unestimated,
)
from ic1.annotation.reannotation.export_rescreen import OUT_CSV as RESCREEN_CSV

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Systems shown in the docs, in the order of the comparison figure.
SYSTEM_FAMILY_MODE = {
    'LLM only (high recall)': ('LLM only', 'high recall'),
    'LLM only (best balance)': ('LLM only', 'best balance'),
    'LLM only (high precision)': ('LLM only', 'high precision'),
    'ML only': ('ML only', None),
    'ML → LLM (high recall)': ('ML → LLM', 'high recall'),
    'ML → LLM (best balance)': ('ML → LLM', 'best balance'),
    'ML → LLM (high precision)': ('ML → LLM', 'high precision'),
}


@dataclass
class LabelNoise:
    """Aggregate re-screen results; ``metrics`` carries ``family`` and ``mode`` columns."""

    overturns: pd.DataFrame
    arms: pd.DataFrame
    metrics: pd.DataFrame
    composition: pd.DataFrame
    unestimated: pd.DataFrame
    n_scored: int
    n_answered: int
    n_exported: int


def load_label_noise(rescreen_csv: Path | str | None = None) -> LabelNoise:
    """Overturn rates, resolver consistency by arm, corrected operating points and error composition."""
    path = Path(rescreen_csv) if rescreen_csv is not None else _REPO_ROOT / RESCREEN_CSV
    if not path.exists():
        raise FileNotFoundError(f'No re-screen export at {path}')

    preds = load_system_predictions()
    population = load_population(preds)
    rs = load_rescreen(path, cell_sizes=population.cell.value_counts().reindex(CELLS))
    systems = [s for s in SYSTEM_FAMILY_MODE if s in preds.columns]

    metrics = corrected_metrics(rs, population, systems)
    metrics['family'] = metrics.system.map(lambda s: SYSTEM_FAMILY_MODE[s][0])
    metrics['mode'] = metrics.system.map(lambda s: SYSTEM_FAMILY_MODE[s][1])
    return LabelNoise(
        overturns=overturn_table(rs, population),
        arms=consistency_table(rs, 'arm'),
        metrics=metrics,
        composition=error_composition(rs, population, systems),
        unestimated=unestimated(population, rs),
        n_scored=len(population),
        n_answered=len(rs),
        n_exported=len(pd.read_csv(path)),
    )
