
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

TASK = TASKS[TaskName.INOUT]

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
