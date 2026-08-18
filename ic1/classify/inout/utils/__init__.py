import hashlib
import json
import logging

import numpy as np

from ic1.core.config import TaskName, TASKS
from ic1.core.utils import DictLikeEncoder
from .annotations import load_data
from .results import Result, TuningFold, results_to_pd, read_tuning_results_df, read_tuning_results, threshold_scores
from .huggingface import ensure_offline_models
from .metrics import evaluate

logger = logging.getLogger(__name__)

TASK = TASKS[TaskName.INOUT]


def hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), cls=DictLikeEncoder).encode()).hexdigest()[:16]


def compute_class_weights(labels: np.ndarray | list[int]) -> np.ndarray:
    if type(labels) is list:
        labels = np.array(labels)
    return labels.shape[0] / (2 * np.unique_counts(labels).counts)  # type: ignore[no-any-return, union-attr]


__all__ = [
    'load_data',
    'TASK',
    'hash_ids',
    'Result',
    'TuningFold',
    'read_tuning_results',
    'read_tuning_results_df',
    'results_to_pd',
    'evaluate',
    'compute_class_weights',
    'ensure_offline_models',
    'threshold_scores'
]
