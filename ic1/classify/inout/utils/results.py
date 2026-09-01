"""Utils for classification."""

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]  # 0.05 .. 0.95


def threshold_scores(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS, beta: float = 1.0) -> list['Result']:
    """Metrics at each threshold, computed from stored predictions (replaces the tune-time sweep)."""
    from .metrics import evaluate
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    return [Result.model_validate(evaluate(yt, yp, threshold=float(t), beta=beta)) for t in thresholds]


class Result(BaseModel):
    threshold: float
    n_samples: int
    Precision: float
    Recall: float
    F1: float
    Accuracy: float
    Fbeta: float
    prop_included: float
    ROC_AUC: float | None = None
    AveragePrecision: float | None = None


class TrialRecord(BaseModel):
    """One Optuna trial's outcome — enough to reconstruct the search trajectory."""
    number: int
    value: float | None = None  # objective; None for failed/pruned trials
    state: str


class TuningFold(BaseModel):
    scores_self: Result
    val_ids: list[str]
    val_labels: list[int]
    val_probs: list[float]
    tune_time: float
    fit_time: float
    model: str
    params: dict[str, Any]
    train_hash: str
    val_hash: str
    trials: list[TrialRecord] = []  # the search trajectory (for convergence diagnostics)
    slurm_info: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def read_tuning_results(source_dir: Path) -> list[TuningFold]:
    logger.info(f'Reading tuning results from {source_dir}')
    results = []
    for file in source_dir.glob('*.json'):
        with open(file) as fp:
            results.append(TuningFold.model_validate_json(fp.read()))
    return results


def results_to_pd(results: list[TuningFold]) -> pd.DataFrame:
    rows = []
    for ri, result in enumerate(results):
        base = result.model_dump()
        base.pop('params')
        scores_self = base.pop('scores_self')
        for key in ('val_ids', 'val_labels', 'val_probs', 'trials'):
            base.pop(key)
        rows.append(base | scores_self | {'scores': 'self', 'result': ri})
        for score in threshold_scores(result.val_labels, result.val_probs):
            rows.append(base | score.model_dump() | {'scores': 'val', 'result': ri})

    return pd.DataFrame(rows)


def read_tuning_results_df(source_dir: Path) -> pd.DataFrame:
    return results_to_pd(read_tuning_results(source_dir))
