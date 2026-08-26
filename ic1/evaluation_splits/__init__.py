from typing import Annotated

import pandas as pd
import typer

from ic1.core.config import TaskName, TASKS
from ic1.evaluation_splits.splits_model import EvaluationSplits


def create_split(
    task: Annotated[TaskName, typer.Option(help='Task name')] = TaskName.INOUT,
    train: Annotated[float, typer.Option(help='Train split')] = 0.65,
    val: Annotated[float, typer.Option(help='Validation split (during development) // test split is the implicit remainder')] = 0.15,
    # test: Annotated[float, typer.Option(help='Test split (final check)')] = 0.2,
    seed: Annotated[int | None, typer.Option(help='Random seed')] = None,
):
    df = pd.read_csv(TASKS[task].shareable_resolved_path)

    df_train = df.sample(frac=train, random_state=seed)
    remainder = df.drop(df_train.index)
    df_val = remainder.sample(frac=val / (1 - train), random_state=seed)
    df_test = remainder.drop(df_val.index)

    with open(TASKS[task].splits_path, 'w') as fp:
        fp.write(
            EvaluationSplits(
                task=task,
                train=df_train['item_id'].tolist(),
                deet=df_train['item_id'].tolist(),
                validation=df_val['item_id'].tolist(),
                test=df_test['item_id'].tolist(),
            ).model_dump_json(indent=2)
        )


__all__ = ['create_split', 'EvaluationSplits']
