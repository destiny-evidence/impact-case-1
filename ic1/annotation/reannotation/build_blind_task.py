"""Create the blind re-screen task for the label-noise experiment (#28) in NACSOS.

For each reviewer, this copies the in/out scheme, so the reviewer's earlier annotations are not shown, and
creates one assignment scope on the copy. The reviewer's assignments are the records routing.py gave them,
in a seeded shuffled order. Checks run before anything is written. The default is a dry run. Pass --write to
create the schemes, scopes and assignments in NACSOS.
"""

import asyncio
import logging
import uuid
from typing import Annotated

import numpy as np
import pandas as pd
import sqlalchemy as sa
import typer
from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.crud.annotations import upsert_annotation_scheme
from nacsos_data.db.schemas import User
from nacsos_data.db.schemas.annotations import Assignment, AssignmentScope, AnnotationScheme
from nacsos_data.models.annotations import AssignmentModel, AssignmentStatus
from nacsos_data.util.annotations.resolve import read_annotation_scheme
from rich.logging import RichHandler

from ic1.annotation.reannotation.routing import OUT_CSV, SEED
from ic1.core.config import settings, TaskName, TASKS

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(markup=True, rich_tracebacks=True)])
logger = logging.getLogger(__name__)

REVIEWER_USERNAMES = {'AM': 'andres.mena', 'MIM': 'maria-inti.metzendorf'}
AVAILABLE = ['AM', 'MIM']


def build_plan(routing: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Per reviewer, the routed records in a shuffled order, checked against the routing rules."""
    routed = routing[routing.reviewer != ''].copy()
    if routed.groupby('item_id').size().max() > 1:
        raise SystemExit('An item is routed to more than one reviewer.')

    self_rows = routed[routed.arm == 'self']
    if (self_rows.reviewer != self_rows.original_resolver).any():
        raise SystemExit('A self-routed record is not routed to its original resolver.')

    cross_rows = routed[routed.arm == 'cross']
    if (cross_rows.reviewer == cross_rows.original_resolver).any():
        raise SystemExit('A cross-routed record is routed to its original resolver.')

    rng = np.random.default_rng(seed)
    plan = {}
    for code in REVIEWER_USERNAMES:
        rows = routed[routed.reviewer == code]
        rows = rows.iloc[rng.permutation(len(rows))].assign(order=lambda d: range(len(d)))
        plan[code] = rows
    return plan


async def create_task(plan: dict[str, pd.DataFrame], write: bool) -> None:
    scheme_id = TASKS[TaskName.INOUT].scheme_id
    db_engine = get_engine_async(settings=settings.DB)

    async with db_engine.session() as session:
        original = await read_annotation_scheme(session=session, annotation_scheme_id=scheme_id)
        user_ids = dict(
            (
                await session.execute(
                    sa.select(User.username, User.user_id).where(User.username.in_(REVIEWER_USERNAMES.values())),
                )
            ).all(),
        )
        existing = (await session.execute(sa.select(AnnotationScheme.name).where(AnnotationScheme.name.like('%re-screen%')))).scalars().all()

    if len(user_ids) != len(REVIEWER_USERNAMES):
        raise SystemExit(f'Expected users {sorted(REVIEWER_USERNAMES.values())}, found {sorted(user_ids)}.')
    if existing:
        raise SystemExit(f'Re-screen schemes already exist: {existing}. Remove them before creating new ones.')

    for code, rows in plan.items():
        scheme = original.model_copy(
            update={
                'annotation_scheme_id': uuid.uuid4(),
                'name': f'{original.name} | re-screen {code}',
                'time_created': None,
            },
        )
        scope_id = uuid.uuid4()
        user_id = user_ids[REVIEWER_USERNAMES[code]]
        assignments = [
            AssignmentModel(
                assignment_id=uuid.uuid4(),
                assignment_scope_id=scope_id,
                annotation_scheme_id=scheme.annotation_scheme_id,
                order=int(row.order),
                item_id=row.item_id,
                user_id=user_id,
                status=AssignmentStatus.OPEN,
            )
            for row in rows.itertuples()
        ]

        arms = rows.arm.value_counts().to_dict()
        cells = rows.cell.value_counts().to_dict()
        logger.info(f'[bold]{code}[/bold] ({REVIEWER_USERNAMES[code]}): {len(assignments)} assignments, arms {arms}, cells {cells}')

        if not write:
            continue

        await upsert_annotation_scheme(scheme, db_engine)
        async with db_engine.session() as session:
            session.add(
                AssignmentScope(
                    assignment_scope_id=scope_id,
                    annotation_scheme_id=scheme.annotation_scheme_id,
                    name=f'Re-screen {code} (blind)',
                    description='Blind re-screen of resolved and unanimous test records.',
                ),
            )
            session.add_all([Assignment(**assignment.model_dump()) for assignment in assignments])
            await session.commit()
        logger.info(f'[green]Created scheme {scheme.name} and scope {scope_id}[/green]')


def main(
    write: Annotated[bool, typer.Option(help='Create the schemes, scopes and assignments in NACSOS. Without this, only print the plan.')] = False,
) -> None:
    routing = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
    plan = build_plan(routing)
    asyncio.run(create_task(plan, write))


if __name__ == '__main__':
    typer.run(main)
