import asyncio
import logging
import uuid
from pathlib import Path
from typing import Annotated

import typer
from nacsos_data.db import get_engine_async
from nacsos_data.db.schemas import AssignmentScope, Assignment
from nacsos_data.models.annotations import AssignmentConfigRandom
from nacsos_data.models.nql import AssignmentFilter, SubQuery
from nacsos_data.util.annotations.assignments import get_db_sample, distribute_assignments
from nacsos_data.util.conf import load_settings

ANNOTATOR_GROUPS = {
    'Group 01 (EM,IK,NS)': [
        '88aaae01-037d-46ae-9513-17dca9f74279',  # evalotte.mohren
        '14bf2e43-d335-48d3-8026-fc78540b2085',  # ismael.kawooya
        '9f165e3a-cf59-47f4-a27f-5e37251bdba5',  # nadia.soliman
    ],
    'Group 02 (KH,LM,PL)': [
        '2d5b57d1-1325-4978-bb65-218a90b7f680',  # kaitlyn.hair
        'def07c7d-6632-48e4-8454-6a2adb1a22e2',  # leah.mwai
        '23270298-61da-41b1-b9f7-1361147af172',  # pastan.lusiba
    ],
    'Group 03 (AF,MP,PO)': [
        '547b0282-3b5e-4905-a0bf-9024813a84a8',  # ailbhe.finnerty
        '22d8b3f1-f3fd-4297-807e-d7296a477968',  # maria.pontes
        '04b89a71-a97d-4d73-a9ee-22e40a644dd0',  # patrick.okwen
    ],
    'Group 04 (ANS,AW,IP,PS)': [
        '12b57791-00f9-49c2-a108-ba83e56ec6fa',  # anna.noel-storr
        '38dff84e-56c8-4880-8b6e-2bc03dbe0b90',  # alang.wung
        '5651e27a-508e-42da-ae4f-30a0a422922a',  # ignacio.pita
        '2b052e97-a991-4cd9-885d-ce148f91a89c',  # pauline.scheelbeek
    ],
    'Group 05 (JB,MC,PN)': [
        '07646efa-1312-4ea8-ba31-00aa5628c2d6',  # jayshrita.bhagabati
        'b4c20ee5-e415-4ac8-8e9d-77770311e38c',  # max.callaghan
        '6edee3b7-f0ac-4b3b-9ad2-ef9c162a4399',  # promise.nduku
    ]
    # Fallback
    # 'fc6127d9-e07c-4e70-97c2-f279899e689f',  # maria-inti.metzendorf
    # '34536463-6250-43f4-8318-2595c273fa7e',  # andres.mena
}


def main(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    project_id: Annotated[str, typer.Option(help='project uuid')] = 'db6ee519-afb5-4813-822b-bfbc7dfd2237',
    scheme_id: Annotated[str, typer.Option(help='Annotation scheme ID')] = '0689d927-f78d-46aa-bbcf-190ce156f707',
    batch_size: Annotated[int, typer.Option(help='Batch size for processing')] = 200,
    num_batches: Annotated[int, typer.Option(help='Number of batches')] = 5,
    batch_offset: Annotated[int, typer.Option(help='Number of first batch')] = 1,
    random_seed: Annotated[int, typer.Option(help='Random seed for reproducibility')] = 4243,
    loglevel: Annotated[str, typer.Option(help='Path to config file')] = 'INFO',
):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level=loglevel)
    # logging.getLogger('root').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logger = logging.getLogger('retrieve')

    settings = load_settings(config.resolve())

    async def _main():
        db_engine = get_engine_async(settings=settings.DB, debug=False)
        async with db_engine.session() as session:
            first = False  # True
            for group, users in ANNOTATOR_GROUPS.items():
                for batch in range(batch_offset, num_batches + batch_offset):
                    logger.info(f'Preparing batch {batch}/{num_batches} for {group}')
                    setup = AssignmentConfigRandom(
                        users={user: batch_size for user in users},
                        overlaps={len(users): batch_size},
                        random_seed=random_seed,
                        nql='NOT IS ASSIGNED WITH 0689d927-f78d-46aa-bbcf-190ce156f707',
                        nql_parsed=SubQuery(not_=AssignmentFilter(scheme=scheme_id, mode=6)),
                    )

                    scope_id = uuid.uuid4()
                    scope = AssignmentScope(
                        assignment_scope_id=scope_id,
                        annotation_scheme_id=scheme_id,
                        name=f'Round 01 | {group} | Batch {batch}/{num_batches}',
                        description=f'Initial round of assignments for {group}, batch {batch:02d}/{num_batches:02d}\n'
                                    f'Data was sampled from backfilled random sample of OpenAlex.',
                        config=setup.model_dump(),
                    )
                    session.add(scope)
                    await session.flush()

                    item_ids = await get_db_sample(
                        session=session,
                        project_id=project_id,
                        num_items=batch_size,
                        nql=setup.nql_parsed if not first else None,
                    )
                    assignments = distribute_assignments(
                        users=setup.users,
                        item_ids=item_ids,
                        overlaps=setup.overlaps,
                        random_seed=setup.random_seed,
                        assignment_scope_id=str(scope_id),
                        annotation_scheme_id=scheme_id,
                    )
                    session.add_all([Assignment(**assignment.model_dump()) for assignment in assignments])
                    await session.flush()
                    first = False

            logger.info(f'Final commit')
            await session.commit()

    asyncio.run(_main())
    logger.info('All done')


if __name__ == '__main__':
    typer.run(main)
