"""Read exports, split out chunk for deet, and create a deet project."""

from typing import Annotated
import typer

import pandas as pd
import json
import yaml

from ic1.core.config import TASKS, TaskName, settings

import pandas as pd
import yaml
from pathlib import Path
from deet.data_models.extraction import ExtractionRunMetadata
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.extractors.llm_data_extractor import DataExtractionConfig
from pydantic import BaseModel
from rich import print


def extract_scalar_values(obj: BaseModel | dict, prefix: str = ""):
    result = {}

    data = obj.model_dump() if isinstance(obj, BaseModel) else dict(obj)

    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, (dict, BaseModel)):
            continue
            result.update(extract_scalar_values(v, f"{key}_"))
        elif v:
            result[key] = v

    return result


def compare(project_path: Path):
    project = DeetProject.load(project_dir=project_path)
    df = pd.DataFrame()
    for d in project.experiments_dir.iterdir():
        exp = ExperimentArtefacts(base_dir=d)
        metrics = pd.read_csv(exp.metrics)
        metrics["run_id"] = exp.run_id
        config = DataExtractionConfig.model_validate(
            yaml.safe_load(exp.config_snapshot.read_text())
        )
        metadata = ExtractionRunMetadata.model_validate_json(
            exp.extraction_metadata.read_text()
        )
        for k, v in extract_scalar_values(metadata).items():
            metrics[k] = v
        for k, v in extract_scalar_values(config).items():
            metrics[k] = v

        df = pd.concat([df, metrics])

    print(
        df.groupby(["run_id", "model"])[
            [
                "total_input_tokens",
                "total_output_tokens",
                "total_pipeline_duration_seconds",
                "total_cost_usd",
            ]
        ].sum()
    )
    print(df.groupby(["run_id", "model","attribute_label","metric_name",])["value"].mean().unstack())

    # for (method, metric), group in df.groupby(["method", "metric_name"]):
    #     print(method)
    #     print(metric)
    #     print(group[""])

def main(task: Annotated[TaskName, typer.Option(help='The annotation task task to be exported')] = TaskName.INOUT):
    if task == TaskName.ALL:
        selected = TASKS
    else:
        selected = {task.value: TASKS[task]}

    for task_config in selected.values():
        compare(task_config.deet_project_path)

if __name__ == "__main__":
    main()




