import json
import logging
from typing import Any

import pandas as pd
import sqlalchemy as sa
from nacsos_data.db.schemas import BotAnnotationMetaData

from nacsos_data.db.schemas.annotations import AnnotationScheme, AssignmentScope
from nacsos_data.models.annotations import AnnotationSchemeLabel
from nacsos_data.util.annotations import get_ordering, read_bot_annotations
from nacsos_data.util.annotations.export import LabelOptions, prepare_export_table
from nacsos_data.util.annotations.resolve import read_scopes_for_scheme
from sqlalchemy.ext.asyncio import AsyncSession

from ic1.core.config import settings, TaskConfig

logger = logging.getLogger(__name__)


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
    usernames = df[df['username'].notna()]['username'].unique()
    if pseudonym_map is None:
        if settings.PSEUDONYM_MAP.exists():
            with open(settings.PSEUDONYM_MAP, 'r') as fp:
                pseudonym_map = json.load(fp)
        else:
            pseudonym_map = {username: f'coder_{ui:03}' for ui, username in enumerate(usernames)}

    # Add any new users to the map
    new_users = set(usernames) - set(pseudonym_map)
    if new_users:
        next_id = max(int(v.split('_')[1]) for v in pseudonym_map.values()) + 1 if pseudonym_map else 0
        for username in sorted(new_users):
            pseudonym_map[username] = f'coder_{next_id:03}'
            next_id += 1

    # Always save the (potentially updated) map
    with open(settings.PSEUDONYM_MAP, 'w') as fp:
        json.dump(pseudonym_map, fp, indent=2)

    return df.replace(pseudonym_map).drop(columns=['user_id'], errors='ignore')


async def get_task_infos(
    task: TaskConfig,
    session: AsyncSession,
    fetch_all_ids: bool = False,
) -> tuple[str, list[AnnotationSchemeLabel], list[LabelOptions], list[str], list[str], list[str]]:
    scheme = (await session.execute(sa.select(AnnotationScheme).where(AnnotationScheme.annotation_scheme_id == task.scheme_id))).scalar_one_or_none()
    if scheme is None:
        raise SystemExit(f'No annotation scheme with id={task.scheme_id!r} found in the database.')

    project_id = str(scheme.project_id)
    labels = [AnnotationSchemeLabel.model_validate(label_def) for label_def in (scheme.labels or [])]
    if fetch_all_ids:
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

    logger.info(f'[bold]Scheme:[/bold] {scheme.name}  (project={project_id})', extra={'markup': True})
    logger.info(f'[bold]Labels:[/bold] {len(label_options)}  [bold]Concept columns:[/bold] {len(label_cols)}', extra={'markup': True})
    logger.info(f'[bold]Assignment scopes:[/bold] {len(scope_ids)} -> {scope_ids}', extra={'markup': True})
    logger.info(f'[bold]Resolution scopes:[/bold] {len(resolved_ids)} -> {resolved_ids}', extra={'markup': True})

    if not scope_ids:
        logger.warning(
            '[yellow]No assignment scopes exist for this scheme yet. The export would have the '
            'full set of (empty) concept columns but no annotated rows. Create a scope + '
            'assignments first if you want populated rows.[/yellow]',
        )

    return project_id, labels, label_options, label_cols, scope_ids, resolved_ids


async def get_item_scopes(session: AsyncSession, scheme_id: str) -> pd.DataFrame:
    """Map each (item_id, username) of the scheme to its assignment scope id.

    Resolutions have username 'RESOLVED' and take the scope of their bot annotation metadata.
    Raises if one (item_id, username) falls in more than one scope.
    """
    rows = []
    for scope_id in await read_scopes_for_scheme(session=session, scheme_id=scheme_id, order_by_name=True):
        for entry in await get_ordering(session=session, assignment_scope_id=scope_id):
            rows += [{'item_id': str(entry.item_id), 'username': a.username, 'assignment_scope_id': str(scope_id)} for a in entry.assignments]

    bots = (await session.execute(sa.select(BotAnnotationMetaData).where(BotAnnotationMetaData.annotation_scheme_id == scheme_id))).scalars().all()
    for bot in bots:
        if bot.assignment_scope_id is None:
            logger.warning(f'Resolution {bot.bot_annotation_metadata_id} has no assignment scope; its items get no scope.')
            continue
        for ann in await read_bot_annotations(session=session, bot_annotation_metadata_id=str(bot.bot_annotation_metadata_id)):
            rows.append({'item_id': str(ann.item_id), 'username': 'RESOLVED', 'assignment_scope_id': str(bot.assignment_scope_id)})

    df = pd.DataFrame(rows, columns=['item_id', 'username', 'assignment_scope_id']).drop_duplicates()
    clash = df[df.duplicated(['item_id', 'username'], keep=False)]
    if not clash.empty:
        raise SystemExit(f'{clash[["item_id", "username"]].drop_duplicates().shape[0]} (item, username) pairs fall in more than one assignment scope.')
    return df


async def read_dataframes(
    session: AsyncSession,
    project_id: str,
    label_cols: list[str],
    resolved_ids: list[str] | None = None,
    scope_ids: list[str] | None = None,
    labels: list[LabelOptions] | None = None,
    nql_filter: Any = None,  # nacsos_data NQLFilter; its Annotated alias embeds ForwardRefs ty cannot resolve
    columns: list[str] | None = None,
    item_scopes: pd.DataFrame | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:

    if not scope_ids:
        logger.info(
            '[yellow]No assignment scopes exist for this scheme yet. The export would have the '
            'full set of (empty) concept columns but no annotated rows. Create a scope + '
            'assignments first if you want populated rows.[/yellow]',
        )
    rows = await prepare_export_table(
        session=session,
        nql_filter=nql_filter,
        bot_annotation_metadata_ids=resolved_ids,
        assignment_scope_ids=scope_ids,
        user_ids=None,
        project_id=project_id,
        labels=labels,
        ignore_hierarchy=True,
        ignore_repeat=True,
    )

    df = pd.DataFrame(rows)  # .rename(columns={'text': 'abstract'}, errors='ignore')
    label_cols = [col for col in label_cols if col in df.columns]

    df[label_cols] = df[label_cols].astype('Int8')
    df['publication_year'] = df['publication_year'].astype('Int16')
    # df['item_order'] = df['item_order'].astype('Int64').astype('Int32')
    # df['scope_order'] = df['scope_order'].astype('Int64').astype('Int16')

    if item_scopes is not None:
        df['item_id'] = df['item_id'].astype(str)
        df = df.merge(item_scopes, on=['item_id', 'username'], how='left')
        logger.info(f'[bold]Rows without an assignment scope:[/bold] {df["assignment_scope_id"].isna().sum()}', extra={'markup': True})

    if columns is not None:
        base_cols = [c for c in columns if c in df.columns]
        for c in label_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df.reindex(columns=base_cols + label_cols)
        logger.info(f'[bold]Base columns:[/bold] {base_cols}')

    logger.info(f'[bold]Rows:[/bold] {df.shape[0]}  [bold]Total columns:[/bold] {df.shape[1]}', extra={'markup': True})
    logger.info(f'[bold]Concept (label) columns:[/bold] {label_cols[:10]}{" ..." if len(label_cols) > 10 else ""}', extra={'markup': True})

    return label_cols, df, pd.concat([pseudonymize(df[df['username'] != 'RESOLVED']), df[df['username'] == 'RESOLVED']])
