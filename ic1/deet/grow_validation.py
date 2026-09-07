"""Move x documents from the ML `train` split into `validation`.

Why
---
The classifier's threshold + recall-floor gate are validated on `validation`.
At ~12% prevalence, 350 val docs is only ~42 positives -- too few to confirm a
0.95 recall floor (CI ~+/-0.066). We deliberately do NOT tune with k-fold CV,
because tuning Optuna inside each fold lets it pick different hyperparameters per
fold, leaving no single model to ship. So instead of re-plumbing tuning, we buy
validation positives by moving labelled docs over from train.

The scarce resource is train POSITIVES (395 of 4000), not train size, so don't
over-move -- see the printed positive counts.

The move is:
- deterministic / reproducible (stable hash order over item ids), and
- sync-stable (`sync_deet_splits` only appends the deet 350/649 and never touches
  train, so moved docs are neither clobbered nor re-added).

Usage
-----
    python -m ic1.deet.grow_validation --x 1000 --dry-run
    python -m ic1.deet.grow_validation --x 1000
"""

from typing import Annotated

import pandas as pd
import typer

from ic1.core.config import TASKS, TaskName
from ic1.core.utils import uniform
from ic1.evaluation_splits.splits_model import EvaluationSplits


def _positive_lookup(task) -> dict[str, int]:
    """Map item_id -> label (1/0) from the shareable annotation frame."""
    df = (
        pd.read_csv(task.shareable_resolved_path)
        .rename(columns={"incl|1": "label"})[["item_id", "label"]]
        .dropna(subset="label")
        .astype({"label": int})
    )
    return dict(zip(df["item_id"].astype(str), df["label"]))


def _report(name: str, ids: list, labels: dict[str, int]) -> None:
    pos = sum(labels.get(str(i), 0) for i in ids)
    n = len(ids)
    prev = pos / n if n else 0.0
    typer.echo(f"  {name:11}: n={n:5}  positives={pos:4}  prevalence={prev:.3f}")


def main(
    x: Annotated[
        int, typer.Option(help="Number of docs to move from train into validation")
    ] = 1000,
    task: Annotated[
        TaskName, typer.Option(help="Which task's splits to modify")
    ] = TaskName.INOUT,
    dry_run: Annotated[
        bool, typer.Option(help="Report the move without writing")
    ] = False,
) -> None:
    task_config = TASKS[task.value]
    splits = EvaluationSplits.load(task_config.splits_path)
    labels = _positive_lookup(task_config)

    if x > len(splits.train):
        raise typer.BadParameter(f"x={x} exceeds train size {len(splits.train)}")

    # Deterministic, reproducible selection: stable hash order over train ids.
    ordered = sorted(splits.train, key=lambda i: uniform("grow_validation", str(i)))
    move = ordered[:x]
    move_set = set(move)

    new_train = [i for i in splits.train if i not in move_set]
    new_validation = splits.validation + move

    # Disjointness must hold.
    assert not (set(new_train) & set(new_validation))
    assert not (set(new_validation) & set(splits.test))

    typer.echo("BEFORE:")
    _report("train", splits.train, labels)
    _report("validation", splits.validation, labels)
    typer.echo(f"MOVING {x} docs (~{sum(labels.get(str(i), 0) for i in move)} positives):")
    typer.echo("AFTER:")
    _report("train", new_train, labels)
    _report("validation", new_validation, labels)

    if dry_run:
        typer.echo("dry run -- not written")
        raise typer.Exit()

    splits.train = new_train
    splits.validation = new_validation
    task_config.splits_path.write_text(splits.model_dump_json(indent=2))
    typer.echo(f"wrote {task_config.splits_path}")


if __name__ == "__main__":
    typer.run(main)
