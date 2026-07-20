"""Export annotations for a NACSOS annotation scheme to CSV.

Companion to ``import_taxonomy.py``: use this to inspect the *exact* column naming
NACSOS produces for an export, so we can confirm the concept<->scheme mapping joins
on the right keys.

NACSOS names each annotation column by the label ``key`` and the choice ``value``.
There are two export code paths in nacsos_data:

  * ``wide_export_table`` derives its columns from the annotations that actually
    exist, so columns for *unused* choice values are silently missing. That makes the
    column set data-dependent and unstable -- bad for any downstream consumer that
    expects a fixed schema (one column per concept).

  * ``prepare_export_table`` (used here) takes an explicit ``labels`` list. Its
    ``_get_label_selects`` emits one column *per declared value*, used or not, so we
    get the complete, stable column set: exactly one ``<label_key>|<value>`` column
    per concept. Rows are long-format: one row per (item, annotator), with a
    ``username`` column ('RESOLVED' for resolved labels).

We build that complete ``labels`` list by reading the scheme back from the database
(the authoritative definition) and walking every label / nested child label.

The scheme id (SCHEME_ID) is imported from import_taxonomy.py so there is a single
source of truth; run the import there first.
"""

import asyncio
import json
from typing import Annotated

import pandas as pd
import sqlalchemy as sa
from nacsos_data.db.schemas import BotAnnotationMetaData
from rich import print
import typer

from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.schemas.annotations import AnnotationScheme, AssignmentScope
from nacsos_data.models.annotations import AnnotationSchemeLabel
from nacsos_data.models.nql import AnnotationFilter
from nacsos_data.util.annotations.export import LabelOptions, prepare_export_table

# Single source of truth: the scheme id is defined by the import script.
from ic1.core.config import settings, TaskName, TASKS

# Lightweight base columns to keep. The export otherwise carries the full document
# `text` (and keywords/authors) for *every* row, which bloats the file to hundreds of
# MB. None of that is needed to link an annotation back to an item or the taxonomy.
KEEP_BASE_COLS = [
    'item_id',
    'doi',
    'wos_id',
    'scopus_id',
    'openalex_id',
    's2_id',
    'pubmed_id',
    'dimensions_id',
    'publication_year',
    'source',
    'title',
    'user_id',
    'username',
]


def collect_label_options(labels: list[AnnotationSchemeLabel]) -> list[LabelOptions]:
    """Walk the scheme (incl. nested child labels) into a complete LabelOptions list.

    One LabelOptions per label key, declaring every choice value so the export emits a
    column for each concept regardless of whether it was ever annotated.
    """
    out: list[LabelOptions] = []
    for lab in labels:
        if lab.choices:
            values = [c.value for c in lab.choices]
            if lab.kind == 'multi':
                out.append(LabelOptions(key=lab.key, options_multi=values))
            elif lab.kind == 'single':
                out.append(LabelOptions(key=lab.key, options_int=values))
            elif lab.kind == 'bool':
                out.append(LabelOptions(key=lab.key, options_bool=[True, False]))
            # recurse into child labels nested under any choice
            for c in lab.choices:
                if c.children:
                    out.extend(collect_label_options(c.children))
        elif lab.kind == 'bool':
            out.append(LabelOptions(key=lab.key, options_bool=[True, False]))
    return out


def expected_label_columns(label_options: list[LabelOptions]) -> list[str]:
    """The full, deterministic set of `<key>|<value>` columns prepare_export_table emits."""
    cols: list[str] = []
    for lo in label_options:
        for v in lo.options_int or []:
            cols.append(f'{lo.key}|{v}')
        for v in lo.options_multi or []:
            cols.append(f'{lo.key}|{v}')
        if lo.options_bool:
            cols += [f'{lo.key}|0', f'{lo.key}|1']
    return cols


def pseudonymize(df: pd.DataFrame, pseudonym_map: dict[str, str] | None = None):
    if pseudonym_map is None:
        if settings.PSEUDONYM_MAP.exists():
            with open(settings.PSEUDONYM_MAP, 'r') as fp:
                pseudonym_map = json.load(fp)
        else:
            pseudonym_map = {username: f'coder_{ui:03}' for ui, username in enumerate(df['username'].unique())}
            with open(settings.PSEUDONYM_MAP, 'w') as fp:
                json.dump(pseudonym_map, fp, indent=2)

    if len(set(df['username'].unique()) - set(pseudonym_map)) > 0:
        raise AssertionError('Pseudonymization map does not cover all users in the dataframe')

    return df.replace(pseudonym_map).drop(columns=['user_id'], errors='ignore')


def main(
    kind: Annotated[TaskName, typer.Option(help='Type of annotations to export')] = TaskName.INOUT,
    export_all: Annotated[bool, typer.Option(help='Instead of using only configured scopes, export everything for in/out scheme')] = False,
) -> None:
    task = TASKS[kind]

    async def _main() -> None:
        db_engine = get_engine_async(settings=settings.DB)

        # 1. Resolve scheme -> project + labels, and gather its assignment scopes.
        async with db_engine.session() as session:
            scheme = (await session.execute(sa.select(AnnotationScheme).where(AnnotationScheme.annotation_scheme_id == task.scheme_id))).scalar_one_or_none()
            if scheme is None:
                raise SystemExit(f'No annotation scheme with id={task.scheme_id!r} found in the database.')

            project_id = str(scheme.project_id)
            labels = [AnnotationSchemeLabel.model_validate(label_def) for label_def in (scheme.labels or [])]

            if export_all:
                scope_ids = list(
                    (
                        await session.execute(
                            sa.select(sa.cast(AssignmentScope.assignment_scope_id, sa.TEXT)).where(AssignmentScope.annotation_scheme_id == task.scheme_id),
                        )
                    )
                    .scalars()
                    .all(),
                )
                resolved_ids = list(
                    (
                        await session.execute(
                            sa.select(sa.cast(BotAnnotationMetaData.bot_annotation_metadata_id, sa.TEXT)).where(
                                BotAnnotationMetaData.annotation_scheme_id == task.scheme_id,
                            ),
                        )
                    )
                    .scalars()
                    .all(),
                )
            else:
                scope_ids = task.scope_ids
                resolved_ids = task.resolved_ids

        label_options = collect_label_options(labels)
        label_cols = expected_label_columns(label_options)

        print(f'[bold]Scheme:[/bold] {scheme.name}  (project={project_id})')
        print(f'[bold]Labels:[/bold] {len(label_options)}  [bold]Concept columns:[/bold] {len(label_cols)}')
        print(f'[bold]Assignment scopes:[/bold] {len(scope_ids)} -> {scope_ids}')
        print(f'[bold]Resolution scopes:[/bold] {len(resolved_ids)} -> {resolved_ids}')

        if not scope_ids:
            print(
                '[yellow]No assignment scopes exist for this scheme yet. The export would have the '
                'full set of (empty) concept columns but no annotated rows. Create a scope + '
                'assignments first if you want populated rows.[/yellow]',
            )

        async with db_engine.session() as session:
            rows = await prepare_export_table(
                session=session,
                nql_filter=AnnotationFilter(incl=True, scopes=task.scope_ids),
                bot_annotation_metadata_ids=resolved_ids,
                assignment_scope_ids=scope_ids,
                user_ids=None,
                project_id=project_id,
                labels=label_options,
                ignore_hierarchy=True,
                ignore_repeat=True,
            )

        df = pd.DataFrame(rows)

        # 4. Keep only lightweight base columns + the full, deterministic concept column set.
        #    Every concept column is present even if prepare_export_table omitted one
        #    (e.g. zero matching rows), filled with NA -> stable schema for downstream.
        base_cols = [c for c in KEEP_BASE_COLS if c in df.columns]
        for c in label_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df.reindex(columns=base_cols + label_cols)

        df[label_cols] = df[label_cols].astype('Int8')
        df['publication_year'] = df['publication_year'].astype('Int16')
        # df['item_order'] = df['item_order'].astype('Int64').astype('Int32')
        # df['scope_order'] = df['scope_order'].astype('Int64').astype('Int16')

        print(f'[bold]Rows:[/bold] {df.shape[0]}  [bold]Total columns:[/bold] {df.shape[1]}')
        print(f'[bold]Base columns:[/bold] {base_cols}')
        print(f'[bold]Concept (label) columns:[/bold] {label_cols[:10]}{" ..." if len(label_cols) > 10 else ""}')

        df.to_csv(task.sensitive_path, index=False)
        print(f'[green]Wrote raw export ({df.shape[0]:,} rows x {df.shape[1]:,} cols) to {task.sensitive_path}[/green]')

        df[df['username'] == 'RESOLVED'].to_csv(task.resolved_path, index=False)
        shape = df[df['username'] == 'RESOLVED'].shape
        print(f'[green]Wrote resolved export ({shape[0]:,} rows x {shape[1]:,} cols) to {task.resolved_path}[/green]')

        df_pseudo = pseudonymize(df[df['username'] != 'RESOLVED'])
        df_pseudo.drop(columns=['user_id'], errors='ignore').to_csv(task.shareable_path, index=False)
        print(f'[green]Wrote resolved export ({df_pseudo.shape[0]:,} rows x {df_pseudo.shape[1]:,} cols) to {task.shareable_path}[/green]')

    asyncio.run(_main())


if __name__ == '__main__':
    typer.run(main)
