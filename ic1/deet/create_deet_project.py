"""Read exports, split out chunk for deet, and create a deet project."""

from typing import Annotated
import typer
import pandas as pd
import json
import yaml

from ic1.core.config import TASKS, TaskName, settings
from ic1.core.utils import uniform
from ic1.evaluation_splits.splits_model import EvaluationSplits


def main(task: Annotated[TaskName, typer.Option(help='The annotation task task to be exported')] = TaskName.ALL):
    from deet.scripts.project_utils import create_project, SupportedImportFormat
    from deet.data_models.project import DeetProject
    from deet.data_models.enums import EvaluationStrategyName
    from deet.extractors.llm_data_extractor import DataExtractionConfig
    from deet.data_models.documents import ContextType

    if task == TaskName.ALL:
        selected = TASKS
    else:
        selected = {task.value: TASKS[task.value]}

    for task_config in selected.values():
        resolved = pd.read_csv(task_config.sensitive_resolved_path)
        item_ids = set(resolved['item_id'])
        sorted_ids = sorted(item_ids, key=lambda iid: uniform(task_config.name, iid))
        deet_ids = sorted_ids[: settings.DEET_N]
        train_ids = sorted_ids[settings.DEET_N :]

        if task_config.splits_path.exists():
            splits = EvaluationSplits.load(task_config.splits_path)
        else:
            splits = EvaluationSplits(task=task_config.task)

        splits.add_items(deet_ids=deet_ids, train_ids=train_ids)
        task_config.splits_path.write_text(splits.model_dump_json(indent=2))

        label_cols = [c for c in resolved.columns if '|' in c]

        deet_df = resolved[resolved['item_id'].isin(deet_ids)][['item_id', 'title', 'text'] + label_cols]
        deet_df[label_cols] = deet_df[label_cols].fillna(0).astype(int)
        deet_df = deet_df.rename(
            columns={
                'item_id': 'document_id',
                'title': 'name',
                'text': 'abstract',
            }
        )
        if task == TaskName.INOUT:
            deet_df = deet_df.rename(
                columns={
                    'incl|1': 'include - high precision'
                }
            )
            deet_df["include - best balance"] = deet_df['include - high precision'].copy()
            deet_df["include - high recall"] = deet_df['include - high precision'].copy()

        deet_df.to_csv(task_config.deet_data_path, index=False)

        create_project(
            task_config.deet_project_path.resolve(),
            name=f'IC1-{task_config.name}',
            data_type=SupportedImportFormat.GENERIC_CSV,
            data_path=task_config.deet_data_path,
            pdf_dir=None,
        )
        PROJECT_YAML = task_config.deet_project_path / "project.yaml"
        project = DeetProject.model_validate(
            yaml.safe_load(
                (PROJECT_YAML).read_bytes()
            )["project"]
        )
        project.evaluation_strategy = EvaluationStrategyName.DEV_VAL_TEST
        data = {"project": project.model_dump(mode="json")}
        with PROJECT_YAML.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        config = DataExtractionConfig(default_context_type=ContextType.ABSTRACT_ONLY)
        (task_config.deet_project_path / "default_extraction_config.yaml").write_text(
            yaml.safe_dump(
                config.model_dump(mode="json"),
                sort_keys=False
            ),
            encoding="utf-8"
        )


if __name__ == '__main__':
    typer.run(main)
