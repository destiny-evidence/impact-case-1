"""Prepare import file

This repeatedly draws random samples from solr until a target size is reached (unique on OpenAlex ID).
Optionally, you can specify a project so that already existing records are excluded from this sample.
"""
import logging
from pathlib import Path
from typing import Annotated

import typer
import sqlalchemy as sa
from nacsos_data.db import get_engine
from nacsos_data.util.academic.apis import OpenAlexSolrAPI
from nacsos_data.util.conf import load_settings
from ic1.core.ids import PROJECT_ID
from ic1.query.revisions.query_20260408 import MERGED

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='INFO')
# logging.getLogger('root').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def main(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Path to target file')],
    project_id: Annotated[str, typer.Option(help='project uuid')] = PROJECT_ID,
    check_existing: Annotated[bool, typer.Option(help='Check if OpenAlex IDs are already in project')] = True,
    batch_size: Annotated[int, typer.Option(help='Batch size for processing')] = 500,
    random_seed: Annotated[int, typer.Option(help='Random seed for reproducibility')] = 4243,
    target_size: Annotated[int, typer.Option(help='Target size of the random sample')] = 10000,
    loglevel: Annotated[str, typer.Option(help='Path to config file')] = 'INFO',
    loglevel_solr: Annotated[str, typer.Option(help='Path to config file')] = 'WARNING',
):
    logger = logging.getLogger('retrieve')
    logger.setLevel(loglevel)
    logger_solr = logger.getChild('solr')
    logger_solr.setLevel(loglevel_solr)
    settings = load_settings(config.resolve())
    query = MERGED['(CLIMATE AND HEALTH) OR ADAPTATION']
    api = OpenAlexSolrAPI(openalex_conf=settings.OPENALEX, batch_size=batch_size, logger=logger_solr)
    db_engine = get_engine(settings=settings.DB, debug=False)

    logger.info(f'Writing items to: {target}')
    logger.info(f'Running query: {query}')

    known_ids: set[str] = set()
    if check_existing:
        if project_id is None:
            raise typer.BadParameter('Must specify project ID for ID checking')
        logger.info(f'Retrieving known OpenAlex IDs from project "{project_id}"')
        with db_engine.session() as session:
            known_ids = set(
                session.execute(
                    sa.text('SELECT openalex_id FROM academic_item WHERE project_id = :project_id AND openalex_id IS NOT NULL;'),
                    params={'project_id': project_id},
                ).scalars().all(),
            )
        logger.info(f'Found {len(known_ids)} IDs in project')

    n_sampled = 0
    n_total = 0
    it = 0
    with open(target, 'w', encoding='utf-8') as f_target:
        while True:
            for item in api.fetch_translated(
                query=query,
                project_id=project_id,
                params={'fq': ['is_xpac:false'], 'rows': batch_size, 'sort': f'random_{random_seed+it} asc', 'cursorMark': None},
            ):
                n_sampled += 1
                if item.openalex_id in known_ids:
                    continue
                known_ids.add(item.openalex_id)
                f_target.write(item.model_dump_json() + '\n')

                n_total += 1
                if n_total >= target_size:
                    break

            logger.info(f'Sampled {n_sampled:,} and kept {n_total:,} so far (known IDs: {len(known_ids):,})')
            it += 1

            if n_total >= target_size:
                break

if __name__ == '__main__':
    typer.run(main)
