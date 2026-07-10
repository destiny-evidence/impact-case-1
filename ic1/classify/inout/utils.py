"""Utils for classification."""

from ic1.core.config import TASKS, TaskName
from ic1.evaluation_splits.splits_model import EvaluationSplits
from ic1.classify.inout.sklearn_configs import CONFIGS as SKLEARN_CONFIGS
from ic1.classify.base import ModelRun, BaseClassifier
import pandas as pd


REGISTRY = {clf.name: clf for clf in SKLEARN_CONFIGS}
INOUT = TASKS[TaskName.INOUT]

def get_classifier(run: ModelRun) -> BaseClassifier:
    clf = REGISTRY[run.model]
    clf.pipeline.set_params(**run.config)
    return clf

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INOUT.resolved_path)

    splits = EvaluationSplits.load(INOUT.splits_path)

    df = df.rename(columns={'incl|1': 'label'})[['item_id','text','label']]

    train_df = df[df['item_id'].isin(splits.train)].reset_index()
    val_df = df[df['item_id'].isin(splits.validation)].reset_index()
    test_df = df[df['item_id'].isin(splits.test)].reset_index()

    return train_df, val_df, test_df
