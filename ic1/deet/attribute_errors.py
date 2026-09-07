"""Rank attributes by number of incorrect LLM answers (leverage for improvement)."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ic1.core.config import TASKS, TaskName, settings
from deet.data_models.project import DeetProject, ExperimentArtefacts

console = Console()


def _scheme_map() -> dict:
    return (
        pd.read_csv(settings.MAPPING_CSV)
        .set_index("pref_label")["scheme_name"]
        .to_dict()
    )


def attribute_errors(comparison: pd.DataFrame, scheme_map: dict) -> pd.DataFrame:
    df = comparison.copy()
    yt = df["human_extraction"].astype(bool)
    yp = df["llm_extraction"].astype(bool)
    df["fp"] = ~yt & yp        # LLM said present, human didn't
    df["fn"] = yt & ~yp        # human said present, LLM missed it
    df["error"] = yt != yp
    g = df.groupby("attribute_label").agg(
        n=("error", "size"),
        errors=("error", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
        n_pos=("human_extraction", lambda s: int(s.astype(bool).sum())),
    )
    g["error_rate"] = g["errors"] / g["n"]
    g["scheme"] = g.index.map(scheme_map)
    return g.sort_values("errors", ascending=False)


def _table(df: pd.DataFrame, title: str, top: int) -> Table:
    df = df.head(top).reset_index()
    t = Table(title=title, header_style="bold cyan")
    cols = ["attribute_label", "scheme", "errors", "fp", "fn", "n_pos", "n", "error_rate"]
    for c in cols:
        t.add_column(c, justify="left" if c in ("attribute_label", "scheme") else "right")
    for _, r in df.iterrows():
        t.add_row(
            str(r.attribute_label), str(r.scheme),
            str(int(r.errors)), str(int(r.fp)), str(int(r.fn)),
            str(int(r.n_pos)), str(int(r.n)), f"{r.error_rate:.2f}",
        )
    return t


def main(
    task: Annotated[TaskName, typer.Option(help="Annotation task")] = TaskName.TAXONOMY,
    top: Annotated[int, typer.Option(help="How many attributes to show per run")] = 25,
):
    scheme_map = _scheme_map()
    project = DeetProject.load(project_dir=TASKS[task].deet_project_path)
    for d in project.experiments_dir.iterdir():
        exp = ExperimentArtefacts(base_dir=d)
        if not exp.is_complete:
            continue
        errs = attribute_errors(pd.read_csv(exp.comparison), scheme_map)
        console.print(_table(errs, f"Most incorrect answers — {exp.run_id}", top))


if __name__ == "__main__":
    typer.run(main)
