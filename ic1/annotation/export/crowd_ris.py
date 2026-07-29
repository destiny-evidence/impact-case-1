import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rispy import BaseWriter
from rispy.config import TAG_KEY_MAPPING, LIST_TYPE_TAGS, DELIMITED_TAG_MAPPING
from nacsos_data.db.connection import get_engine_async
from nacsos_data.models.nql import AnnotationFilter, AssignmentFilter, SubQuery, AbstractFilter

from ic1.annotation.export.utils import get_task_infos, read_dataframes

from ic1.core.config import settings, TaskName, TASKS

logger = logging.getLogger(__name__)


class FieldSet(str, Enum):
    COMPLETE = 'complete'
    MINIMAL = 'minimal'


class Sample(str, Enum):
    ANNOTATED = 'ANNOTATED'
    UNSEEN = 'UNSEEN'


class RisWriter(BaseWriter):
    START_TAG = 'TY'
    PATTERN = '{tag}  - {value}'
    DEFAULT_MAPPING = TAG_KEY_MAPPING
    DEFAULT_LIST_TAGS = LIST_TYPE_TAGS
    DEFAULT_DELIMITER_MAPPING = DELIMITED_TAG_MAPPING

    def set_header(self, count):
        return ''


def main(
    sample: Annotated[Sample, typer.Argument(help='Sample to export')],
    target: Annotated[Path, typer.Option(help='Target file to export to')],
    limit: Annotated[int, typer.Option(help='Limit the number of rows to export')] = 1000,
    split_inout: Annotated[bool, typer.Option(help='Split records in two separate files for includes and excludes')] = False,
    end_mark: Annotated[bool, typer.Option(help='Add "EF" mark at the end of the file')] = False,
    field_set: Annotated[FieldSet, typer.Option(help='')] = FieldSet.MINIMAL,
    kind: Annotated[TaskName, typer.Option(help='Type of annotations to export')] = TaskName.INOUT,
    export_all_annotated: Annotated[bool, typer.Option(help='Instead of using only configured scopes, export everything for in/out scheme')] = False,
    random_seed: Annotated[int | None, typer.Option(help='Seed for random number generator')] = None,
) -> None:
    if kind != TaskName.INOUT:
        raise NotImplementedError('RIS export is only supported for inout schemes')

    task = TASKS[kind]

    async def _main() -> None:
        db_engine = get_engine_async(settings=settings.DB)

        row: pd.Series
        df: pd.Series

        async with db_engine.session() as session:
            project_id, labels, label_options, label_cols, scope_ids, resolved_ids = await get_task_infos(
                task=task,
                session=session,
                fetch_all_ids=export_all_annotated,
            )

            _, df, _ = await read_dataframes(
                session=session,
                project_id=project_id,
                label_cols=label_cols,
                resolved_ids=resolved_ids,
                scope_ids=scope_ids,
                labels=label_options,
                nql_filter=(
                    AnnotationFilter(incl=True, scopes=scope_ids)
                    if sample == Sample.ANNOTATED
                    else SubQuery(and_=[SubQuery(not_=AssignmentFilter(mode=1, scheme=task.scheme_id)), AbstractFilter(comp='>=', size=20)])
                ),
            )

        if sample == Sample.ANNOTATED:
            df = df[df['username'] == 'RESOLVED']
        else:
            df = df[df['username'].isna()]
        df.fillna(None, inplace=True)

        entries_incl = []
        entries_excl = []
        for _, row in df.sample(n=limit, random_state=random_seed).iterrows():
            entry = {
                'type_of_reference': 'GEN',
                'id': row['item_id'],
                'doi': row['doi'],
                'title': row['title'],
                'abstract': row['text'],
                'publication_year': row['publication_year'],
            }
            if pd.notna(row['incl|1']):
                entry['notes'] = [f'Include = {row["incl|1"]}']

            if field_set == FieldSet.COMPLETE:
                entry |= {
                    'authors': [author['name'] for author in row['authors']] if row['authors'] is not None else None,
                    'keywords': row['keywords'],
                    'custom1': row['openalex_id'],
                    'custom2': row['scopus_id'],
                    'custom3': row['dimensions_id'],
                    'custom4': row['wos_id'],
                    'custom5': row['pubmed_id'],
                    'custom6': '; '.join([f'{col}={row[col]}' for col in label_cols if pd.notna(row[col])]),
                    'place_published': row['source'],
                }

            entry = {k: v for k, v in entry.items() if v is not None and (not pd.api.types.is_scalar(v) or pd.notna(v))}

            if pd.notna(row['incl|1']) and row['incl|1'] == 1:
                entries_incl.append(entry)
            else:
                entries_excl.append(entry)

        target.parent.mkdir(parents=True, exist_ok=True)
        if split_inout:
            logger.info(f'Writing {len(entries_incl)} included and {len(entries_excl)} excluded references to {target}')
            with open(target.with_suffix('.incl.ris'), 'w') as fp:
                fp.write(RisWriter().formats(references=entries_incl))
                if end_mark:
                    fp.write('\n\nEF')
            with open(target.with_suffix('.excl.ris'), 'w') as fp:
                fp.write(RisWriter().formats(references=entries_excl))
                if end_mark:
                    fp.write('\n\nEF')
        else:
            with open(target, 'w') as fp:
                fp.write(RisWriter().formats(references=entries_incl + entries_excl))
                if end_mark:
                    fp.write('\n\nEF')

    asyncio.run(_main())


if __name__ == '__main__':
    typer.run(main)
