"""Export the blind re-screen answers for the label-noise experiment (#28) from NACSOS.

Reads the annotations and assignment statuses on the re-screen schemes created by build_blind_task.py, joins
them onto the routing, and writes one row per routed record. The output carries resolver identity, so it is
written to the private directory.
"""

import asyncio
import logging
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.schemas import User
from nacsos_data.db.schemas.annotations import AnnotationScheme, Assignment, AssignmentScope
from nacsos_data.models.annotations import AnnotationSchemeLabel
from nacsos_data.models.nql import AnnotationFilter
from nacsos_data.util.annotations.export import prepare_export_table
from rich.logging import RichHandler

from ic1.annotation.export.utils import collect_label_options, expected_label_columns
from ic1.annotation.reannotation.build_blind_task import REVIEWER_USERNAMES
from ic1.annotation.reannotation.routing import OUT_CSV as ROUTING_CSV
from ic1.core.config import settings
from ic1.reporting.loaders.annotation import _decision

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(markup=True, rich_tracebacks=True)])
logger = logging.getLogger(__name__)

OUT_CSV = Path('data/private/reannotation/rescreen.csv')


async def read_rescreen() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotations (one row per item and reviewer) and assignments on the re-screen schemes."""
    db_engine = get_engine_async(settings=settings.DB)

    async with db_engine.session() as session:
        schemes = (await session.execute(sa.select(AnnotationScheme).where(AnnotationScheme.name.like('%re-screen%')))).scalars().all()
        if len(schemes) != len(REVIEWER_USERNAMES):
            raise SystemExit(f'Expected {len(REVIEWER_USERNAMES)} re-screen schemes, found {[s.name for s in schemes]}.')
        scheme_ids = [s.annotation_scheme_id for s in schemes]
        project_ids = {str(s.project_id) for s in schemes}
        if len(project_ids) != 1:
            raise SystemExit(f'Re-screen schemes span several projects: {project_ids}.')

        scope_ids = [
            str(scope_id)
            for scope_id in (await session.execute(sa.select(AssignmentScope.assignment_scope_id).where(AssignmentScope.annotation_scheme_id.in_(scheme_ids))))
            .scalars()
            .all()
        ]

        assignments = pd.DataFrame(
            (
                await session.execute(
                    sa.select(sa.cast(Assignment.item_id, sa.TEXT).label('item_id'), User.username, Assignment.status)
                    .join(User, User.user_id == Assignment.user_id)
                    .where(Assignment.assignment_scope_id.in_(scope_ids)),
                )
            )
            .mappings()
            .all(),
        )

        labels = [AnnotationSchemeLabel.model_validate(label) for label in (schemes[0].labels or [])]
        label_options = collect_label_options(labels)
        rows = await prepare_export_table(
            session=session,
            nql_filter=AnnotationFilter(incl=True, scopes=scope_ids),
            bot_annotation_metadata_ids=None,
            assignment_scope_ids=scope_ids,
            user_ids=None,
            project_id=project_ids.pop(),
            labels=label_options,
            ignore_hierarchy=True,
            ignore_repeat=True,
        )

    label_cols = expected_label_columns(label_options)
    annotations = pd.DataFrame(rows)
    annotations = annotations.reindex(columns=['item_id', 'username', *label_cols])
    annotations['item_id'] = annotations.item_id.astype(str)
    annotations = annotations.rename(columns={f'reason|{value}': f'reason|{name}' for value, name in reason_names(labels).items()})
    return annotations[annotations.username.notna()], assignments


def reason_names(labels: list[AnnotationSchemeLabel]) -> dict[int, str]:
    """Choice value -> choice name for the 'reason' label, wherever it is nested in the scheme."""
    for label in labels:
        if label.key == 'reason':
            return {choice.value: choice.name for choice in label.choices or []}
        for choice in label.choices or []:
            if choice.children and (names := reason_names(choice.children)):
                return names
    return {}


def build_rescreen(routing: pd.DataFrame, annotations: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """One row per routed record with the blind decision and reason, checked against the routing."""
    routed = routing[routing.reviewer.notna()].copy()
    routed['reviewer_username'] = routed.reviewer.map(REVIEWER_USERNAMES)

    if annotations.duplicated(['item_id', 'username']).any():
        raise SystemExit('A reviewer annotated an item more than once.')
    wrong = annotations.merge(routed[['item_id', 'reviewer_username']], on='item_id', how='left')
    wrong = wrong[wrong.username != wrong.reviewer_username]
    if not wrong.empty:
        raise SystemExit(f'{len(wrong)} annotations are not by the routed reviewer: {wrong[["item_id", "username"]].to_dict("records")}')

    reason_cols = [c for c in annotations.columns if c.startswith('reason|')]
    blind = pd.DataFrame(
        {
            'item_id': annotations.item_id,
            'blind_label': [_decision(a, b) for a, b in zip(annotations['incl|1'], annotations['incl|0'], strict=True)],
            'blind_reason': annotations[reason_cols].apply(lambda r: next((c.split('|')[1] for c in reason_cols if r[c] == 1), pd.NA), axis=1),
        },
    )
    status = assignments.rename(columns={'username': 'reviewer_username', 'status': 'assignment_status'})
    return routed.merge(status, on=['item_id', 'reviewer_username'], how='left', validate='one_to_one').merge(
        blind,
        on='item_id',
        how='left',
        validate='one_to_one',
    )


def main() -> None:
    routing = pd.read_csv(ROUTING_CSV, dtype={'item_id': str, 'reviewer': str, 'arm': str})
    annotations, assignments = asyncio.run(read_rescreen())
    rescreen = build_rescreen(routing, annotations, assignments)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rescreen.to_csv(OUT_CSV, index=False)
    logger.info(f'[green]Wrote {len(rescreen)} records to {OUT_CSV}[/green]')

    print(pd.crosstab(rescreen.reviewer, rescreen.assignment_status.astype(str), margins=True).to_string())
    missing = rescreen[rescreen.blind_label.isna()]
    print(f'\nRouted records without a decision: {len(missing)}')
    print(pd.crosstab(rescreen.cell, rescreen.blind_label.fillna(-1).astype(int), margins=True).to_string())


if __name__ == '__main__':
    main()
