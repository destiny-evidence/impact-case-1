"""Read exported data and write a resolved version."""

import typer

from ic1.core.config import TASKS, TaskName, TaskConfig
from typing import Annotated
import pandas as pd


def resolve_annotations(task_config: TaskConfig):
    df = pd.read_csv(task_config.sensitive_path)
    df = df[df['username'] == 'RESOLVED'].reset_index(drop=True)
    df.to_csv(task_config.resolved_path, index=False)


def main(task: Annotated[TaskName, typer.Option(help='The annotation task task to be exported')] = TaskName.ALL):
    if task == 'all':
        selected = TASKS
    else:
        selected = {task.value: TASKS[task.value]}

    for task_config in selected.values():
        resolve_annotations(task_config)


if __name__ == '__main__':
    typer.run(main)
