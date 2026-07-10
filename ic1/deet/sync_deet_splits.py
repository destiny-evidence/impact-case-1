"""Read splits from deet project, sync to project splits."""
import typer
from deet.data_models.evaluation_strategies.dev_val_test import DevValTestSplits, DeetProject
from typing import Annotated
from ic1.core.config import TaskName, TASKS
from ic1.evaluation_splits.splits_model import EvaluationSplits

def main(
        task: Annotated[TaskName, typer.Option(help="The annotation task task to be exported")] = TaskName.ALL
    ):
    if task == TaskName.ALL:
        selected = TASKS
    else:
        selected = {task.value: TASKS[task.value]}

    for task_config in selected.values():
        deet_project = DeetProject.load(project_dir=task_config.deet_project_path)
        deet_splits = DevValTestSplits.load(task_config.deet_project_path / deet_project.evaluation_splits_path)

        internal_external_id_map = {
            document.safe_identity.internal_id: document.safe_identity.external_id
            for document in deet_project.process_data().documents
        }

        splits = EvaluationSplits.load(task_config.splits_path)
        validation_ids = [
            internal_external_id_map[iid]
            for iid in deet_splits.development_ids + deet_splits.validation_ids
        ]
        test_ids = [
            internal_external_id_map[iid]
            for iid in deet_splits.test_ids
        ]
        splits.add_items(
            validation_ids=validation_ids, test_ids=test_ids,
            exclude_deet=True
        )
        task_config.splits_path.write_text(splits.model_dump_json(indent=2))


if __name__ == '__main__':
    typer.run(main)
