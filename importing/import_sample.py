"""Prepare import file based on list of IDs

This assumes, that you already constructed a list of (random) OpenAlex IDs and stored them in a file (one ID per line, no URL prefix).
For example, this might be the case when running the nacsos-academic-search backfilling pipeline.
"""
import logging
from itertools import batched
from pathlib import Path
from typing import Annotated

import typer
from nacsos_data.util.academic.apis import OpenAlexSolrAPI
from nacsos_data.util.conf import load_settings
from query_revisions.query_20260408 import MERGED

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='INFO')
# logging.getLogger('root').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def main(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    ids: Annotated[Path, typer.Option(help='Path to ids file')],
    target: Annotated[Path, typer.Option(help='Path to target file')],
    project_id: Annotated[str | None, typer.Option(help='project uuid')] = None,
    batch_size: Annotated[int, typer.Option(help='Batch size for processing')] = 5000,
    excl_xpac: Annotated[bool, typer.Option(help='Do not include xpac')] = True,
    ensure_abstract: Annotated[bool, typer.Option(help='Only use records with abstract')] = False,
    loglevel: Annotated[str, typer.Option(help='Path to config file')] = 'INFO',
    loglevel_solr: Annotated[str, typer.Option(help='Path to config file')] = 'WARNING',
):
    logger = logging.getLogger('retrieve')
    logger.setLevel(loglevel)
    logger_solr = logger.getChild('solr')
    logger_solr.setLevel(loglevel_solr)

    settings = load_settings(config)
    query = MERGED['(CLIMATE AND HEALTH) OR ADAPTATION']
    api = OpenAlexSolrAPI(openalex_conf=settings.OPENALEX, batch_size=batch_size, logger=logger_solr)

    filters = []
    if excl_xpac:
        filters.append('is_xpac:false')
    if ensure_abstract:
        filters.append('abstract:*')

    logger.info(f'Running query: {query}')
    logger.info(f'Extra filters: {filters}')
    logger.info(f'Writing items to: {target}')
    logger.info(f'Reading IDs from: {ids}')

    n_requested = 0
    n_found = 0
    with open(ids) as f_ids, open(target, 'w', encoding='utf-8') as f_target:
        for batch in batched(f_ids, n=batch_size, strict=False):
            batch_ids = [id_.strip() for id_ in batch]
            n_requested += len(batch_ids)
            for item in api.fetch_translated(
                query=query,
                project_id=project_id,
                params={'fq': [f'id: ({' '.join(batch_ids)})'] + filters},
            ):
                f_target.write(item.model_dump_json() + '\n')
                n_found += 1
            logger.info(f'Ran query with {n_requested:,} filter IDs and matched {n_found:,} records with query overlap')


if __name__ == '__main__':
    typer.run(main)
