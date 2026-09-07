"""Read exported data and write a resolved version."""

import logging
import typer
from rich.logging import RichHandler

from ic1.core.config import TASKS, TaskName, TaskConfig
from typing import Annotated
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(markup=True, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


def resolve_annotations(task_config: TaskConfig):
    df = pd.read_csv(task_config.sensitive_path)
    resolved_df = df[df['username'] == 'RESOLVED'].reset_index(drop=True)
    logger.info(f'[bold]{task_config.name}:[/bold] {resolved_df.shape[0]} resolved rows → {task_config.shareable_resolved_path}')
    resolved_df.to_csv(task_config.shareable_resolved_path, index=False)


def main(task: Annotated[TaskName, typer.Option(help='The annotation task task to be exported')] = TaskName.ALL):
    if task == TaskName.ALL:
        selected = TASKS
    else:
        selected = {task.value: TASKS[task.value]}

    for task_config in selected.values():
        resolve_annotations(task_config)


if __name__ == '__main__':
    typer.run(main)
