"""Run N extraction passes over a FIXED validation split and majority-vote them.

The CLI validate flow re-samples the validation set each call and forks (accept/reject),
so it can't do repeated passes on the same held-out docs. This script instead:

  1. puts `size` previously-unassigned docs into the validation stage ONCE (or reuses an
     existing validation split), and persists the splits to disk;
  2. calls the extraction pipeline `n_runs` times over that SAME validation set, WITHOUT
     accepting or rejecting (the project stays in validation mode);
  3. majority-votes the per-(document, scope) predictions and reports P/R per scope,
     alongside the per-run spread.

Nothing here mutates dev/test or forks the splits — after it runs you can still accept or
reject via the normal CLI. Usage:

  uv run python ic1/deet/validate_vote.py --experiment 2026-09-01_09-45-12_luna_retuned_s3 \
      --size 130 --n-runs 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from sklearn.metrics import precision_score, recall_score

from ic1.core.config import TASKS, TaskName
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.data_models.evaluation_strategies.dev_val_test import DevValTestEvaluationStage
from deet.extractors.cli_helpers import (
    evaluate_extraction_pipeline,
    run_extraction_pipeline,
)

console = Console()


def main(
    experiment: Annotated[str, typer.Option(
        help="Existing experiment run (name under the project's experiments dir, or a full "
             "path); its config and prompt snapshots are reused")],
    task: Annotated[TaskName, typer.Option(help="Annotation task")] = TaskName.INOUT,
    size: Annotated[int, typer.Option(help="Docs to sample into validation if it's empty")] = 130,
    n_runs: Annotated[int, typer.Option(help="Number of passes to vote over")] = 3,
    reset: Annotated[bool, typer.Option(help="Clear the current validation split and resample")] = False,
) -> None:
    project = DeetProject.load(project_dir=TASKS[task].deet_project_path)
    strategy = project.load_evaluation_strategy()
    project_doc_ids = project.get_all_doc_ids()

    exp_path = Path(experiment)
    if not exp_path.exists():
        exp_path = project.experiments_dir / experiment
    exp = ExperimentArtefacts(base_dir=exp_path)
    console.print(f"Reproducing config + prompts from experiment: [bold]{exp_path.name}[/]")

    if reset:
        strategy.splits.validation_ids = []

    # Populate the validation stage ONCE; reuse it on re-runs so the docs stay fixed.
    if not strategy.splits.validation_ids:
        n_added = strategy.splits.add_to_stage(
            DevValTestEvaluationStage.VALIDATION, project_doc_ids, size
        )
        console.print(f"[green]Added {n_added} docs to the validation split.[/]")
    else:
        console.print(
            f"[yellow]Reusing existing validation split "
            f"({len(strategy.splits.validation_ids)} docs). Use --reset to resample.[/]"
        )

    strategy.splits.current_stage = DevValTestEvaluationStage.VALIDATION
    # Persist BEFORE running: the pipeline reloads splits from disk to pick active docs.
    strategy.splits.dump_to_json(project.evaluation_splits_path)
    val_ids = list(strategy.splits.validation_ids)
    console.print(f"Validation set: {len(val_ids)} documents; running {n_runs} passes.\n")

    run_dirs: list[Path] = []
    for i in range(n_runs):
        run_output, processed, artefacts, _config = run_extraction_pipeline(
            deet_project=project,
            prompt_csv_path=exp.prompts_snapshot,
            config_path=exp.config_snapshot,
            run_name=f"VALIDATION_vote{i + 1}",
        )
        evaluate_extraction_pipeline(
            processed_annotation_data=processed,
            run_output=run_output,
            experiment_artefacts=artefacts,
        )
        run_dirs.append(artefacts.base_dir)
        console.print(f"  pass {i + 1}/{n_runs} -> {artefacts.base_dir.name}")

    _report(run_dirs)
    console.print(
        "\n[dim]Project left in VALIDATION mode; accept/reject via the normal CLI when ready.[/]"
    )


def _report(run_dirs: list[Path]) -> None:
    dfs = [pd.read_csv(d / "goldstandard_llm_comparison.csv") for d in run_dirs]
    n = len(dfs)
    thr = n // 2 + 1  # strict majority
    console.print(f"\n[bold cyan]Majority vote over {n} passes (>= {thr}/{n}):[/]")
    for mode in dfs[0].attribute_label.unique():
        subs = [d[d.attribute_label == mode] for d in dfs]
        yt = subs[0].set_index("document_id")["human_extraction"].astype(int)
        votes = sum(s.set_index("document_id")["llm_extraction"].astype(int) for s in subs)
        votes = votes.reindex(yt.index)
        yp = (votes >= thr).astype(int)
        ties = int((votes == n / 2).sum()) if n % 2 == 0 else 0
        Ps = [precision_score(s.human_extraction.astype(int), s.llm_extraction.astype(int),
                              zero_division=0) for s in subs]
        Rs = [recall_score(s.human_extraction.astype(int), s.llm_extraction.astype(int),
                           zero_division=0) for s in subs]
        P = precision_score(yt, yp, zero_division=0)
        R = recall_score(yt, yp, zero_division=0)
        tie_note = f"  ties={ties}" if ties else ""
        console.print(
            f"  {mode[10:]:16s} voted P={P:.2f} R={R:.2f}   "
            f"(per-pass P={np.mean(Ps):.2f}±{np.std(Ps):.2f} "
            f"R={np.mean(Rs):.2f}±{np.std(Rs):.2f}){tie_note}"
        )


if __name__ == "__main__":
    typer.run(main)
