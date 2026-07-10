"""Read exports, split out chunk for deet, and create a deet project."""

import typer
from typing import Annotated
import pandas as pd

from ic1.core.config import (
    TASKS,
    TaskName,
    DEET_N
)
from ic1.evaluation_splits.splits_model import EvaluationSplits
from deet.scripts.project_utils import create_project, SupportedImportFormat
import hashlib

def uniform(task: str, item_id: str) -> float:
    """Stable, language-independent uniform value in [0, 1) for (task, item_id)."""
    h = hashlib.sha256(f'{task}:{item_id}'.encode()).hexdigest()
    return int(h, 16) / 16**64

def main(
        task: Annotated[TaskName, typer.Option(help="The annotation task task to be exported")] = TaskName.ALL
    ):
    if task == TaskName.ALL:
        selected = TASKS
    else:
        selected = {task.value: TASKS[task.value]}

    for task_config in selected.values():
        resolved = pd.read_csv(task_config.resolved_path)
        item_ids = set(resolved["item_id"])
        sorted_ids = sorted(item_ids, key=lambda iid: uniform(task_config.name, iid))
        deet_ids = sorted_ids[:DEET_N]
        train_ids = sorted_ids[DEET_N:]

        if task_config.splits_path.exists():
            splits = EvaluationSplits.model_validate_json(task_config.splits_path.read_text())
        else:
            splits = EvaluationSplits(task=task_config.name)

        splits.add_items(deet_ids=deet_ids, train_ids=train_ids)
        task_config.splits_path.write_text(splits.model_dump_json(indent=2))

        label_cols = [c for c in resolved.columns if '|' in c]

        deet_df = resolved[resolved["item_id"].isin(deet_ids)][['item_id','title','text'] + label_cols].rename(columns={
            "item_id": "document_id",
            "title": "name"
        })
        deet_df.to_csv(task_config.deet_data_path, index=False)

        create_project(
            task_config.deet_project_path.resolve(),
            name=f"IC1-{task_config.name}",
            data_type=SupportedImportFormat.GENERIC_CSV,
            data_path=task_config.deet_data_path,
            pdf_dir=None
        )

if __name__ == '__main__':
    typer.run(main)
