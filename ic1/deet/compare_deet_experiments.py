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
from rich.console import Console
from rich.table import Table
from rich.text import Text


console = Console()


def df_to_table(
    df: pd.DataFrame,
    title: str | None = None,
    float_format: str = "{:,.2f}",
    compare_key: str | None = None,
) -> Table:
    """Render a (possibly multi-indexed) DataFrame as a rich Table.

    Repeated values in the index columns are shown only on their first row, and a
    separator line is drawn whenever the outermost index level changes, so nested
    grouping (e.g. several ``attribute_label`` rows under one ``run_id``) is obvious.

    If ``compare_key`` names an index column, each numeric data cell is shaded
    green/red when its value is higher/lower than the previous row sharing that key
    (i.e. the same attribute in the preceding run). Rows must already be ordered so
    that "previous" means the earlier run.
    """
    index_cols = list(df.index.names)
    n_index = len(index_cols)
    df = df.reset_index()
    columns = list(df.columns)
    table = Table(title=title, header_style="bold cyan")

    for i, col in enumerate(columns):
        numeric = pd.api.types.is_numeric_dtype(df[col])
        table.add_column(
            str(col),
            justify="right" if numeric else "left",
            style="bold" if i == 0 else None,
        )

    # Per-compare_key memory of the last-seen value in each data column.
    last_by_key: dict[object, dict[str, float]] = {}

    prev = [None] * n_index
    prev_top = None
    for _, row in df.iterrows():
        # Draw a rule when the outermost group changes.
        if n_index and prev_top is not None and row[columns[0]] != prev_top:
            table.add_section()
        prev_top = row[columns[0]] if n_index else None

        key_val = row[compare_key] if compare_key else None
        seen = last_by_key.setdefault(key_val, {})

        cells = []
        collapse = True  # blank an index cell only while all ancestors also match
        for i, col in enumerate(columns):
            v = row[col]
            if i < n_index:
                if collapse and v == prev[i]:
                    cells.append("")
                    continue
                collapse = False
                prev[i] = v
                # Reset deeper levels so they re-print under a new parent.
                for j in range(i + 1, n_index):
                    prev[j] = None
                cells.append("-" if pd.isna(v) else str(v))
                continue

            cells.append(_data_cell(v, col, seen, float_format, bool(compare_key)))
        table.add_row(*cells)

    return table


def _data_cell(v, col, seen, float_format, do_compare):
    """Format one data cell, shading it vs the previous same-key value."""
    if pd.isna(v):
        return "-"
    text = float_format.format(v) if isinstance(v, float) else str(v)
    if not (do_compare and isinstance(v, (int, float))):
        return text

    prev_v = seen.get(col)
    seen[col] = v
    if prev_v is None or v == prev_v:
        return text
    return Text(text, style="on green" if v > prev_v else "on red")


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

    cost = df.groupby(["run_id", "model"])[
        [
            "total_input_tokens",
            "total_output_tokens",
            "total_pipeline_duration_seconds",
            "total_cost_usd",
        ]
    ].sum()
    console.print(df_to_table(cost, title="Cost & throughput per run"))

    scores = (
        df.groupby(["run_id", "model", "attribute_label", "metric_name"])["value"]
        .mean()
        .unstack()
    )
    console.print(
        df_to_table(
            scores, title="Metrics per attribute", compare_key="attribute_label"
        )
    )

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




