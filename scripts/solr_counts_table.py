import logging
import time
from itertools import chain, zip_longest
import httpx
import pandas as pd
from nacsos_data.util.conf import load_settings
from query_revisions.query_20260408 import CLIMATE, HEALTH, ADAPTATION, MERGED

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='INFO')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
# logging.getLogger('root').setLevel(logging.WARNING)
logger = logging.getLogger('counting')

conf = load_settings('.conf/secret.env')


def count(q: str) -> str:
    try:
        res = httpx.post(
            f'{conf.OPENALEX.solr_url}/select',
            data={
                'df': 'title_abstract',
                'defType': 'lucene',
                'q': q,
                'q.op': 'AND',
                'rows': 5,
            },
            timeout=120,
        ).json()
        return res["response"]["numFound"]
    except KeyError:
        return res['error']['msg']


counts = {}
for group, (name, query) in chain(
    zip_longest([], CLIMATE.items(), fillvalue='CLIMATE'),
    zip_longest([], HEALTH.items(), fillvalue='HEALTH'),
    zip_longest([], ADAPTATION.items(), fillvalue='ADAPTATION'),
    zip_longest([], MERGED.items(), fillvalue='MERGED'),
):
    start = time.time()
    cnt = count(query)
    if type(cnt) is int:
        counts[(group, name)] = {'Query size': cnt}
        cnt = f'{cnt:,}'
    else:
        counts[(group, name)] = {'Query size': pd.NA}

    logger.info(f'{group} ({name}): {cnt}   | took {time.time() - start:2f} seconds')

pd.DataFrame(counts).T.to_csv('notes/2026-04-09_counts.csv', index=True, header=True)
