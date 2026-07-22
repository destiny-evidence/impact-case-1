
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
logger= logging.getLogger(__name__)
if TYPE_CHECKING:
    from torch import Tensor



def evaluate(
    # expecting 1D array for y_true and y_pred
    y_true: 'np.ndarray | Tensor',
    y_pred: 'np.ndarray | Tensor',
    threshold: float = 0.5,
):
    y_pred_binary = np.where(y_pred > threshold, 1, 0)

    results = {
        'F1': f1_score(y_true, y_pred_binary, zero_division=0),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred_binary),
        'threshold': threshold,
        'n_samples': y_true.shape[0],
    }

    try:
        results['ROC_AUC'] = roc_auc_score(y_true, y_pred)
    except:  # noqa: E722
        pass

    logger.debug(' | '.join([f'{key}: {score:.1%}' for key, score in results.items()]))
    return results
