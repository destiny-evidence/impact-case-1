"""Frequency of taxonomy concepts in the taxonomy export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from ic1.core.config import TASKS, TaskName, settings

logger = logging.getLogger(__name__)


def column_counts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per '<label_key>|<value>' column."""
    cols = [c for c in df.columns if '|' in c and c != 'excl|1']
    counts = pd.DataFrame(
        {
            'column': cols,
            'selected': [int(df[c].sum()) for c in cols],
            'seen': [int(df[c].notna().sum()) for c in cols],
        }
    )
    return counts


def with_concepts(counts: pd.DataFrame, mapping_csv: Path) -> pd.DataFrame:
    """Attach concept metadata by joining counts.column onto mapping.col_pipe."""
    cols = ['col_pipe', 'label_key', 'value', 'pref_label', 'depth', 'broader_uri', 'concept_uri', 'scheme_name']
    mapping = pd.read_csv(mapping_csv, usecols=cols)
    out = counts.merge(mapping, left_on='column', right_on='col_pipe', how='left')

    unmatched = out['pref_label'].isna().sum()
    if unmatched:
        missing = out.loc[out['pref_label'].isna(), 'column'].head(5).tolist()
        raise AssertionError(
            f'{unmatched} export column(s) not in mapping (e.g. {missing}); '
            f'the export and {mapping_csv.name} were likely built from different taxonomy versions.'
        )

    n_children = mapping['broader_uri'].value_counts()
    out['n_children'] = out['concept_uri'].map(n_children).fillna(0).astype(int)
    out['is_leaf'] = out['n_children'].eq(0)


    return out.drop(columns='col_pipe').sort_values('selected', ascending=False, ignore_index=True)


def main(
    task: Annotated[TaskName, typer.Option(help='Annotation task')] = TaskName.TAXONOMY,
    target: Annotated[Path | None, typer.Option(help='Where to write the frequency CSV')] = None,
) -> None:
    """Build the per-column frequency table and write it to a shareable CSV."""
    export_csv = TASKS[task].shareable_resolved_path
    target = target or TASKS[task].frequencies_path  # add this property, mirroring difficulty_path

    counts = with_concepts(
        column_counts(pd.read_csv(export_csv)),
        settings.MAPPING_CSV
    )
    logger.info(f'{len(counts)} concept columns; top: {counts.head(5)[["column", "selected"]].to_dict("records")}')
    counts.to_csv(target, index=False)
    logger.info(f'Wrote frequency table to {target}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    typer.run(main)
