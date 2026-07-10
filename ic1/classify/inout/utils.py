"""Utils for classification."""

from ic1.core.config import TASKS, TaskName
from ic1.evaluation_splits.splits_model import EvaluationSplits
from ic1.classify.inout.sklearn_configs import CONFIGS as SKLEARN_CONFIGS, SklearnClassifier
from ic1.classify.inout.transformer_configs import CONFIGS as TRANSFORMER_CONFIGS
from ic1.classify.base import ModelRun, BaseClassifier
import pandas as pd
from ic1.deet.create_deet_project import uniform
from rich import print

CONFIGS = SKLEARN_CONFIGS + TRANSFORMER_CONFIGS


REGISTRY = {clf.name: clf for clf in CONFIGS}
INOUT = TASKS[TaskName.INOUT]

def _to_tuple(v):
    if isinstance(v, list):
        return tuple(_to_tuple(x) for x in v)
    return v

def get_classifier(run: ModelRun) -> BaseClassifier:
    clf = REGISTRY[run.model]
    if isinstance(clf, SklearnClassifier):
        clf.pipeline.set_params(**{k: _to_tuple(v) for k, v in run.config.items()})
    else:
        # TODO: set parameters for TransformerClassifier
        pass
    return clf

def load_data(dev: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, val, and test data.

    if dev==True, split train into sub-train,val,test splits, that are safe to play with.
    """
    df = pd.read_csv(INOUT.resolved_path)

    splits = EvaluationSplits.load(INOUT.splits_path)

    df = df.rename(columns={'incl|1': 'label'})[['item_id','text','label']]
    df = df.dropna(subset='label')

    train_df = df[df['item_id'].isin(splits.train)].reset_index()
    val_df = df[df['item_id'].isin(splits.validation)].reset_index()
    test_df = df[df['item_id'].isin(splits.test)].reset_index()

    if dev and val_df.empty and test_df.empty:
        print('[yellow bold]DEV MODE[/yellow bold]: val/test borrowed from train')
        ids = sorted(train_df['item_id'].tolist(), key=lambda x: uniform(x, 'dev_split'))
        n = len(ids)
        n_val = int(n * 0.15)
        n_test = int(n * 0.15)
        n_train = int(n * 0.7)
        val_ids  = set(ids[:n_val])
        test_ids = set(ids[n_val : n_val + n_test])
        train_ids = set(ids[n_val + n_test: n_val+n_test+n_train])
        val_df   = train_df[train_df['item_id'].isin(val_ids)].reset_index(drop=True)
        test_df  = train_df[train_df['item_id'].isin(test_ids)].reset_index(drop=True)
        train_df = train_df[train_df['item_id'].isin(train_ids)].reset_index(drop=True)


    print(
        f'train_df: {train_df.shape} - {train_df['label'].sum()/train_df.shape[0]} relevant\n'
        f'val_df: {val_df.shape} - {val_df['label'].sum()/val_df.shape[0]} relevant\n'
        f'test_df: {test_df.shape} - {test_df['label'].sum()/test_df.shape[0]} relevant'
    )

    print(train_df['label'].sum()/train_df.shape[0])

    return train_df, val_df, test_df
