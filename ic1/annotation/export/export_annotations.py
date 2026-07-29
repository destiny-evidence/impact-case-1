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
import logging
from typing import Annotated
from rich.logging import RichHandler

import typer


from nacsos_data.db.connection import get_engine_async
from nacsos_data.models.nql import AnnotationFilter

from ic1.annotation.export.utils import get_task_infos, read_dataframes

# Single source of truth: the scheme id is defined by the import script.
from ic1.core.config import settings, TaskName, TASKS

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(markup=True, rich_tracebacks=True)])

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
    'text',
    'user_id',
    'username',
]
logger = logging.getLogger(__name__)


def main(
    kind: Annotated[TaskName, typer.Option(help='Type of annotations to export')] = TaskName.INOUT,
    export_all_annotated: Annotated[bool, typer.Option(help='Instead of using only configured scopes, export everything for in/out scheme')] = False,
) -> None:
    task = TASKS[kind]

    async def _main() -> None:
        db_engine = get_engine_async(settings=settings.DB)

        async with db_engine.session() as session:
            project_id, labels, label_options, label_cols, scope_ids, resolved_ids = await get_task_infos(
                task=task,
                session=session,
                fetch_all_ids=export_all_annotated,
            )
            label_cols, df, df_pseudo = await read_dataframes(
                session=session,
                project_id=project_id,
                label_cols=label_cols,
                resolved_ids=resolved_ids,
                scope_ids=scope_ids,
                labels=label_options,
                nql_filter=AnnotationFilter(incl=True, scopes=scope_ids),
                columns=KEEP_BASE_COLS,
            )

        logger.info(f'[bold]Rows:[/bold] {df.shape[0]}  [bold]Total columns:[/bold] {df.shape[1]}')
        logger.info(f'[bold]Concept (label) columns:[/bold] {label_cols[:10]}{" ..." if len(label_cols) > 10 else ""}')

        df.to_csv(task.sensitive_path, index=False)
        logger.info(f'[green]Wrote raw export ({df.shape[0]:,} rows x {df.shape[1]:,} cols) to {task.sensitive_path}[/green]')

        df[df['username'] == 'RESOLVED'].to_csv(task.resolved_path, index=False)
        shape = df[df['username'] == 'RESOLVED'].shape
        logger.info(f'[green]Wrote resolved export ({shape[0]:,} rows x {shape[1]:,} cols) to {task.resolved_path}[/green]')

        df_pseudo.drop(columns=['user_id'], errors='ignore').to_csv(task.shareable_path, index=False)
        logger.info(f'[green]Wrote resolved export ({df_pseudo.shape[0]:,} rows x {df_pseudo.shape[1]:,} cols) to {task.shareable_path}[/green]')

    asyncio.run(_main())


if __name__ == '__main__':
    typer.run(main)
