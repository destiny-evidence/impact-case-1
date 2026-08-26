import logging
from typing import TYPE_CHECKING

import pandas as pd

from ic1.core.config import TASKS, TaskName
from ic1.core.utils import uniform
from ic1.evaluation_splits.splits_model import EvaluationSplits

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

TASK = TASKS[TaskName.INOUT]

def _fabricate_dev_splits(
    train_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically split the train pool into 70/15/15 train/val/test.

    Used only to run the pipeline without touching the real test set.
    Ordering is a stable hash of item_id, so the split is reproducible
    """
    ids = sorted(train_df['item_id'].tolist(), key=lambda x: uniform(x, 'dev_split'))
    n = len(ids)
    n_val = int(n * 0.15)
    n_test = int(n * 0.15)
    val_ids = set(ids[:n_val])
    test_ids = set(ids[n_val : n_val + n_test])
    train_ids = set(ids[n_val + n_test :])
    return (
        train_df[train_df['item_id'].isin(train_ids)].reset_index(drop=True),
        train_df[train_df['item_id'].isin(val_ids)].reset_index(drop=True),
        train_df[train_df['item_id'].isin(test_ids)].reset_index(drop=True),
    )

def load_data(dev: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, val, and test data.

    if dev==True, split train into sub-train,val,test splits, that are safe to play with.
    """
    logger.info(f'Loading data from {TASK.shareable_resolved_path}')
    df = pd.read_csv(TASK.shareable_resolved_path)

    splits = EvaluationSplits.load(TASK.splits_path)

    df = (
        df.rename(columns={'incl|1': 'label'})[['item_id', 'title', 'text', 'label']]
        .dropna(subset='label')
        .astype({'label': int})
    )

    train_df = df[df['item_id'].isin(splits.train)].reset_index(drop=True)

    if dev:
        logger.info('[yellow bold]DEV MODE[/yellow bold]: val/test borrowed from train')
        train_df, val_df, test_df = _fabricate_dev_splits(train_df)
    else:
        val_df = df[df['item_id'].isin(splits.validation)].reset_index(drop=True)
        test_df = df[df['item_id'].isin(splits.test)].reset_index(drop=True)
        if val_df.empty:
            raise ValueError(
                'No validation split found. Run deet sync to populate validation/test, '
                'or pass --dev-mode to fabricate splits from train.'
            )

    logger.info(
        f'train_df: {train_df.shape} - {train_df["label"].sum() / train_df.shape[0]} relevant\n'
        f'val_df: {val_df.shape} - {val_df["label"].sum() / val_df.shape[0]} relevant\n'
        f'test_df: {test_df.shape} - {test_df["label"].sum() / test_df.shape[0]} relevant'
    )

    logger.info(train_df['label'].sum() / train_df.shape[0])

    return train_df, val_df, test_df
