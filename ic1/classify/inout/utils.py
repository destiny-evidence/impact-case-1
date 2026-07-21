"""Utils for classification."""

import json
import hashlib
import logging
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

TASK = TASKS[TaskName.INOUT]


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


def load_data(dev: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, val, and test data.

    if dev==True, split train into sub-train,val,test splits, that are safe to play with.
    """
    logger.info(f'Loading data from {TASK.resolved_path}')
    df = pd.read_csv(TASK.resolved_path)

    splits = EvaluationSplits.load(TASK.splits_path)

    df = df.rename(columns={'incl|1': 'label'})[['item_id', 'text', 'label']]
    df = df.dropna(subset='label')

    train_df = df[df['item_id'].isin(splits.train)].reset_index()
    val_df = df[df['item_id'].isin(splits.validation)].reset_index()
    test_df = df[df['item_id'].isin(splits.test)].reset_index()

    if dev and val_df.empty and test_df.empty:
        logger.info('[yellow bold]DEV MODE[/yellow bold]: val/test borrowed from train')
        ids = sorted(train_df['item_id'].tolist(), key=lambda x: uniform(x, 'dev_split'))
        n = len(ids)
        n_val = int(n * 0.15)
        n_test = int(n * 0.15)
        n_train = int(n * 0.7)
        val_ids = set(ids[:n_val])
        test_ids = set(ids[n_val : n_val + n_test])
        train_ids = set(ids[n_val + n_test : n_val + n_test + n_train])
        val_df = train_df[train_df['item_id'].isin(val_ids)].reset_index(drop=True)
        test_df = train_df[train_df['item_id'].isin(test_ids)].reset_index(drop=True)
        train_df = train_df[train_df['item_id'].isin(train_ids)].reset_index(drop=True)

    logger.info(
        f'train_df: {train_df.shape} - {train_df["label"].sum() / train_df.shape[0]} relevant\n'
        f'val_df: {val_df.shape} - {val_df["label"].sum() / val_df.shape[0]} relevant\n'
        f'test_df: {test_df.shape} - {test_df["label"].sum() / test_df.shape[0]} relevant'
    )

    logger.info(train_df['label'].sum() / train_df.shape[0])

    return train_df, val_df, test_df


def hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), cls=DictLikeEncoder).encode()).hexdigest()[:16]


def compute_class_weights(labels: np.ndarray | list[int]) -> np.ndarray:
    if type(labels) is list:
        labels = np.array(labels)
    return labels.shape[0] / (2 * np.unique_counts(labels).counts)


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


def ensure_offline_models(models: list[str] | None = None):
    from huggingface_hub import snapshot_download

    if models is None:
        from ic1.classify.inout.models import MODEL_CONFIGS
        from ic1.classify.inout.models.configs._abc import _HuggingfaceClassifierConfig

        models = [config.model_name for config in MODEL_CONFIGS if issubclass(config, _HuggingfaceClassifierConfig)]

    for model in models:
        logger.info(f'Downloading model: {model} so it is available offline in {settings.OFFLINE_MODELS_DIR}')
        snapshot_download(
            repo_id=model,
            repo_type='model',
            cache_dir=settings.OFFLINE_MODELS_DIR,
            force_download=False,
        )
