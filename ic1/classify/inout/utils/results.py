"""Utils for classification."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from ic1.core.config import settings, TASKS, TaskName
from ic1.core.utils import uniform, DictLikeEncoder
from ic1.evaluation_splits.splits_model import EvaluationSplits

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class Result(BaseModel):
    threshold: float
    n_samples: int
    Precision: float
    Recall: float
    F1: float
    Accuracy: float
    ROC_AUC: float | None = None


class TuningFold(BaseModel):
    scores_self: Result
    scores_test: Result
    scores_val: list[Result]
    tune_time: float
    fit_time: float
    model: str
    params: dict[str, Any]
    train_hash: str
    tune_hash: str
    test_hash: str
    val_hash: str
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
        scores_test = base.pop('scores_test')
        scores_val = base.pop('scores_val')
        rows.append(base | scores_self | {'scores': 'self'})
        rows.append(base | scores_test | {'scores': 'test'})
        for score in scores_val:
            rows.append(base | score | {'scores': 'val', 'result': ri})

    return pd.DataFrame(rows)


def read_tuning_results_df(source_dir: Path) -> pd.DataFrame:
    return results_to_pd(read_tuning_results(source_dir))
