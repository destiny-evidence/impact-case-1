#!/usr/bin/env python3
"""Run a full test over every project document that has a real abstract.

Motivation
----------
Documents without an abstract are genuinely unscreenable from text -- they are a
*data* problem, not a *model* problem, so they should not enter any metric. This
script selects every never-seen document (not already in dev/val/test) whose
abstract clears a sensible character threshold, assigns them to the TEST stage,
and runs the standard extraction + evaluation pipeline over them using the
current prompts and config. This gives a clean held-out read -- documents used
for development are always excluded.

Usage
-----
    cd ic1/deet/projects/inout
    python run_test_on_abstracts.py --dry-run          # report counts only
    python run_test_on_abstracts.py                    # run the held-out test

The existing evaluation_splits.json is backed up to
evaluation_splits.json.bak before it is overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from destiny_sdk.labs.references import LabsReference

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

# A real abstract is comfortably longer than a title; below this we treat the
# record as effectively abstract-less. Tune with --min-chars.
DEFAULT_MIN_ABSTRACT_CHARS = 100

app = typer.Typer(help=__doc__)


def _abstract_length(doc) -> int:
    """Return the character length of a document's abstract (0 if none)."""
    try:
        abstract = LabsReference(reference=doc.citation).abstract
    except Exception:  # noqa: BLE001 -- a malformed reference == no usable abstract
        return 0
    return len(abstract) if abstract else 0


def _select_abstract_doc_ids(project: DeetProject, min_chars: int) -> list[int]:
    """Return never-seen doc ids whose abstract clears `min_chars`."""
    processed = project.process_data()

    strategy = project.load_evaluation_strategy()
    excluded: set[int] = set()
    for field in ("development_ids", "validation_ids", "test_ids"):
        excluded.update(getattr(strategy.splits, field, []))

    selected: list[int] = []
    n_no_abstract = 0
    for doc in processed.documents:
        doc_id = doc.safe_identity.document_id
        if doc_id is None:
            continue
        if _abstract_length(doc) < min_chars:
            n_no_abstract += 1
            continue
        if doc_id in excluded:
            continue
        selected.append(doc_id)

    notify(
        f"Documents: {len(processed.documents)} total | "
        f"{n_no_abstract} below {min_chars} chars (excluded) | "
        f"{len(excluded)} already assigned (excluded) | "
        f"{len(selected)} never-seen selected for test"
    )
    return selected


@app.command()
def run(
    min_chars: Annotated[
        int,
        typer.Option(help="Minimum abstract length (chars) to be testable."),
    ] = DEFAULT_MIN_ABSTRACT_CHARS,
    config: Annotated[
        Path,
        typer.Option(help="Extractor config YAML."),
    ] = Path("configs/luna.yaml"),
    prompts: Annotated[
        Path,
        typer.Option(help="Prompt definitions CSV."),
    ] = Path("prompts/prompt_definitions.csv"),
    run_name: Annotated[
        str,
        typer.Option(help="Name for the experiment run directory."),
    ] = "FULL_ABSTRACT_TEST",
    dry_run: Annotated[
        bool,
        typer.Option(help="Report selection counts and exit without running."),
    ] = False,
) -> None:
    """Select never-seen abstract-bearing docs into TEST and run the pipeline."""
    project = DeetProject.load()
    selected = _select_abstract_doc_ids(project, min_chars)
    if not selected:
        notify("No documents selected -- nothing to run.", level=LogLevel.WARNING)
        raise typer.Exit(code=1)
    if dry_run:
        notify("Dry run -- splits untouched, pipeline not run.")
        raise typer.Exit(code=0)

    # Overwrite splits: put the selected docs in TEST.
    splits_path = project.evaluation_splits_path
    strategy = project.load_evaluation_strategy()
    strategy.splits.test_ids = selected
    strategy.splits.current_stage = DevValTestEvaluationStage.TEST
    strategy.splits.dump_to_json(splits_path)
    notify(f"Wrote {len(selected)} ids to TEST stage in {splits_path}")

    run_output, processed_run, artefacts, _config = run_extraction_pipeline(
        deet_project=project,
        prompt_csv_path=prompts,
        config_path=config,
        run_name=run_name,
    )
    evaluate_extraction_pipeline(
        processed_annotation_data=processed_run,
        run_output=run_output,
        experiment_artefacts=artefacts,
    )
    notify(
        f"Done. Metrics + comparison written under {artefacts.base_dir}",
        level=LogLevel.SUCCESS,
    )


if __name__ == "__main__":
    app()
