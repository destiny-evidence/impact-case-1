import logging
import os
from pathlib import Path

import httpx

from ic1.query.revisions.query_20250101_api import query

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='DEBUG')
logging.getLogger('httpcore').setLevel(logging.WARNING)

FILE = Path('data/superset/climate_health_ids.txt')
if FILE.exists():
    logging.warning('Stopping here for safety so you do not accidentally overwrite the previous data')
    raise RuntimeError(f'File {FILE} already exists!')

FILE.parent.mkdir(parents=True, exist_ok=True)

query = query.replace('\n', ' ')

cursor = '*'
ids = 0
page_i = 0

with open(FILE, 'w') as f:
    while cursor is not None:
        page_i += 1
        res = httpx.get(
            'https://api.openalex.org/works',
            params={
                'filter': (
                    # title + abstract search
                    f'title_and_abstract.search: {query}'
                    # f'title_and_abstract.search.no_stem: {query}'  # raw/no stemming
                    # not open access
                    # f',open_access.is_oa:false'
                    # published by springer or elsevier
                    # f',primary_location.source.publisher_lineage:p4310320990|p4310319965'
                ),
                'select': 'id',
                'cursor': cursor,
                'per-page': 200,
            },
            headers={'api_key': os.getenv('API_KEY')},
            timeout=None,
        )
        page = res.json()
        cursor = page['meta']['next_cursor']
        logging.info(f'Retrieved {ids:,}/{page["meta"]["count"]:,}; currently on page {page_i}')

        # print(res.status_code)
        # print(json.dumps(dict(res.headers), indent=2))
        # print(res.text)
        # print(json.dumps(res.json(), indent=2))
        page_ids = [raw_work['id'][21:] for raw_work in page['results']]
        # ids += page_ids
        f.write('\n'.join(page_ids) + '\n')


#
#
# NACSOS database:
# SELECT academic_item.openalex_id
# FROM academic_item
#      JOIN m2m_import_item m2mii on academic_item.item_id = m2mii.item_id
# WHERE m2mii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b';
#
# SELECT DISTINCT ai.openalex_id
# FROM academic_item ai
#      JOIN m2m_import_item m2mii ON ai.item_id = m2mii.item_id
#      JOIN annotation a ON a.item_id = ai.item_id
# WHERE m2mii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
#   AND a.key = 'relevant' AND a.value_bool = TRUE;
#
# Evaluate overlaps:
# ids = {}
# with open('data/superset/climate_health_ids.txt') as f:
#     ids['OpenAlex'] = set(f.readlines())
# with open('data/superset/climate_health_ids_db.txt') as f:
#     ids['NACSOS'] = set(f.readlines())
# with open('data/superset/climate_health_ids_db_rel.txt') as f:
#     ids['NACSOS (rel)'] = set(f.readlines())
# with open('data/superset/climate_health_pred_incl.txt') as f:
#     ids['Predicted'] = set(f.readlines())
# for k1, v1 in ids.items():
#     print(f'{k1} has {len(v1):,} records')
#     for k2, v2 in ids.items():
#         if k1 != k2:
#             print(f'  {k2} has {len(v2):,} records')
#             print(f'   > Union: {len(v1 | v2):,} '
#                   f'| intersect: {len(v1 & v2):,} '
#                   f'| {k1} exclusive: {len(v1 - v2):,} '
#                   f'| {k2} exclusive {len(v2 - v1):,}')
