"""Assign documents to dataset splits — deterministically, and recorded.

Splits are SEPARATE per task (``inout`` vs ``taxonomy``): only IN-included documents are
taxonomy-coded, so the taxonomy universe is exactly the set of taxonomy-assigned docs (no
cross-task sharing). See [[ic1-export-splits-design]].

Assignment is a stable hash of ``(task, item_id)`` -> uniform ``[0, 1)`` -> a split by
fixed cumulative boundaries, so a document's split NEVER changes as exports accumulate.
The result is *recorded* in a committed CSV (``<task>_splits.csv``); existing rows are
LOCKED — on re-run we keep the recorded split and only append newly-seen item_ids (a
recompute mismatch, e.g. after changing fractions, is reported but does not overwrite).

Universe = union of exported item_ids across all versions in data/exports/<task>/.
Classifiers must read splits ONLY from here (single source of truth -> leakage-safe).
"""

import csv
import hashlib
from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich import print

from ic1.core.config import TASKS, SHAREABLE_ROOT, DATASETS_DIR

# PROVISIONAL split fractions (insertion order defines the cumulative boundaries; must sum
# to ~1.0). The final ratios / phased (wave) strategy are tied to issues #1 and #5 and
# still to be confirmed — only the mechanism below is final.
FRACTIONS: dict[str, float] = {'dev': 0.15, 'val': 0.15, 'test': 0.30, 'train': 0.40}

FIELDNAMES = ['item_id', 'split', 'u', 'added']


def uniform(task: str, item_id: str) -> float:
    """Stable, language-independent uniform value in [0, 1) for (task, item_id)."""
    h = hashlib.sha256(f'{task}:{item_id}'.encode()).hexdigest()
    return int(h, 16) / 16**64


def assign_split(u: float, fractions: dict[str, float]) -> str:
    cum = 0.0
    for name, frac in fractions.items():
        cum += frac
        if u < cum:
            return name
    return next(reversed(fractions))  # floating-point guard


def exported_item_ids(task: str) -> set[str]:
    """Union of item_ids across every exported version for this task."""
    ids: set[str] = set()
    f = SHAREABLE_ROOT / f'{task}.csv'
    ids = set(pd.read_csv(f, usecols=['item_id'])['item_id'].astype(str))
    return ids


def splits_path(task: str) -> Path:
    return DATASETS_DIR / f'{task}_splits.csv'


def make_splits(task: str, fractions: dict[str, float] = FRACTIONS) -> None:
    universe = exported_item_ids(task)
    path = splits_path(task)

    existing: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding='utf-8') as fh:
            existing = {r['item_id']: r for r in csv.DictReader(fh)}

    rows = dict(existing)
    today = date.today().isoformat()
    new = mism = 0
    for iid in sorted(universe):
        u = uniform(task, iid)
        split = assign_split(u, fractions)
        if iid in rows:
            if rows[iid]['split'] != split:  # locked: keep the recorded split
                mism += 1
        else:
            rows[iid] = {'item_id': iid, 'split': split, 'u': f'{u:.6f}', 'added': today}
            new += 1

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for iid in sorted(rows):
            w.writerow(rows[iid])

    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r['split']] = counts.get(r['split'], 0) + 1
    print(f'[bold]{task}[/bold]: {len(rows)} docs ({new} new) -> {counts}  [dim]{path}[/dim]')
    if mism:
        print(f'[yellow]  {mism} recorded split(s) differ from current fractions — kept recorded (locked).[/yellow]')
    if not universe:
        print(f'[yellow]  no exports found under {EXPORT_ROOT / task}/ — run the export first.[/yellow]')


def main(task: Annotated[str, typer.Option(help="'inout', 'taxonomy', or 'all'")] = 'all') -> None:
    tasks = list(TASKS) if task == 'all' else [task]
    if any(t not in TASKS for t in tasks):
        raise typer.BadParameter(f'task must be one of {list(TASKS) + ["all"]}')
    for t in tasks:
        make_splits(t)


if __name__ == '__main__':
    typer.run(main)
