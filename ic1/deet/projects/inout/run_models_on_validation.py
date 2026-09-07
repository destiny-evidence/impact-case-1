#!/usr/bin/env python3
"""Run each candidate model once on the validation split — a fair model bake-off.

Motivation
----------
For model selection we want a *controlled* comparison: every candidate model
sees the identical validation documents, the identical three operating-point
prompts, and exactly one sample per prompt (no self-consistency voting, which
only stabilises evaluation and inflates cost). This script forces ``votes: 1``
and runs the standard extraction + evaluation pipeline over the current
validation split for each model config, writing one run per model.

The resulting runs (``MODELCMP_<model>``) feed the cost/performance (Pareto)
figure: single-pass cost per prompt, combined under the production cascade.

Usage
-----
    cd ic1/deet/projects/inout
    python run_models_on_validation.py --dry-run          # report + exit
    python run_models_on_validation.py                    # run all candidates
    python run_models_on_validation.py --models luna,sol  # a subset
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer
import yaml
from deet.data_models.evaluation_strategies.dev_val_test import (
    DevValTestEvaluationStage,
)
from deet.data_models.project import DeetProject
from deet.extractors.cli_helpers import (
    evaluate_extraction_pipeline,
    run_extraction_pipeline,
)
from deet.settings import LogLevel
from deet.ui import notify

# Candidate models = config stems under configs/ (4o-mini excluded by request).
CANDIDATES = ["luna", "sol", "terra", "deepseek", "kimi26", "claudeopus"]

EXP_DIR = Path("data-extraction-experiments")

app = typer.Typer(help=__doc__)


def _validation_ids(strategy) -> list[int]:
    """Recover the validation document ids from the validation run's snapshot.

    The live project splits may have an empty ``validation_ids`` (a later test
    run overwrites the file), but ``validation_run_id`` names the run that
    defined the validation set, and that run snapshotted its own splits — the
    authoritative source.
    """
    run_id = strategy.splits.validation_run_id
    if not run_id:
        raise typer.BadParameter("splits has no validation_run_id to recover from")
    snap = EXP_DIR / run_id / "evaluation_splits.json"
    if not snap.exists():
        raise typer.BadParameter(f"no snapshot at {snap}")
    return json.loads(snap.read_text()).get("validation_ids") or []


def _completed_run_exists(stem: str) -> bool:
    """True if a finished ``MODELCMP_<stem>`` run is already on disk.

    Completion = the comparison CSV was written. Crashed/partial runs (missing
    the CSV) don't count, so deleting a failed run's folder lets it re-run.
    """
    return any(
        d.is_dir()
        and d.name.endswith(f"MODELCMP_{stem}")
        and (d / "goldstandard_llm_comparison.csv").exists()
        for d in EXP_DIR.iterdir()
    )


def _single_pass_config(config_path: Path, max_workers: int | None = None) -> Path:
    """Temp copy of the model config with voting off and optional concurrency cap.

    ``max_workers`` lowers request concurrency — the hosted (DeepSeek/Kimi)
    endpoints crash under the default, so run them with a smaller value.
    """
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg["votes"] = 1
    if max_workers is not None:
        cfg["max_workers"] = max_workers
    fd, tmp = tempfile.mkstemp(suffix=f"_{config_path.stem}_single_pass.yaml")
    Path(tmp).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(tmp)


@app.command()
def run(
    models: Annotated[
        str,
        typer.Option(help="Comma-separated config stems to run."),
    ] = ",".join(CANDIDATES),
    prompts: Annotated[
        Path,
        typer.Option(help="Prompt definitions CSV."),
    ] = Path("prompts/prompt_definitions.csv"),
    max_workers: Annotated[
        int | None,
        typer.Option(help="Override config max_workers (lower = less "
                          "concurrency; hosted models crash under the default)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Report the plan and exit without running."),
    ] = False,
) -> None:
    """Run each candidate model single-pass over the existing validation split."""
    project = DeetProject.load()
    splits_path = project.evaluation_splits_path
    strategy = project.load_evaluation_strategy()
    # The live splits' validation_ids may be empty (a later test run overwrote
    # the file); recover them from the validation run's own snapshot.
    val_ids = _validation_ids(strategy)
    if not val_ids:
        notify("No validation ids recoverable from the validation run.",
               level=LogLevel.WARNING)
        raise typer.Exit(code=1)

    stems = [m.strip() for m in models.split(",") if m.strip()]
    missing = [s for s in stems if not (Path("configs") / f"{s}.yaml").exists()]
    if missing:
        notify(f"No config for: {', '.join(missing)}", level=LogLevel.WARNING)

    notify(f"Validation split: {len(val_ids)} documents "
           f"(from {strategy.splits.validation_run_id}) | models: {', '.join(stems)}")
    if dry_run:
        notify("Dry run — stage not changed, nothing run.")
        raise typer.Exit(code=0)

    # Restore the validation split into the live project, then run against it.
    strategy.splits.validation_ids = val_ids
    strategy.splits.current_stage = DevValTestEvaluationStage.VALIDATION
    strategy.splits.dump_to_json(splits_path)

    for stem in stems:
        config_path = Path("configs") / f"{stem}.yaml"
        if not config_path.exists():
            continue
        if _completed_run_exists(stem):
            notify(f"↷ {stem}: MODELCMP run already present — skipping.")
            continue
        single_pass = _single_pass_config(config_path, max_workers=max_workers)
        notify(f"→ {stem}: single-pass validation run…")
        run_output, processed_run, artefacts, _config = run_extraction_pipeline(
            deet_project=project,
            prompt_csv_path=prompts,
            config_path=single_pass,
            run_name=f"MODELCMP_{stem}",
        )
        evaluate_extraction_pipeline(
            processed_annotation_data=processed_run,
            run_output=run_output,
            experiment_artefacts=artefacts,
        )
        notify(f"  done {stem}: {artefacts.base_dir}", level=LogLevel.SUCCESS)

    notify("All candidate models run on the validation split.",
           level=LogLevel.SUCCESS)


if __name__ == "__main__":
    app()
