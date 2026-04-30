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


def count(q: str, filters: list[str]) -> str:
    try:
        res = httpx.post(
            f'{conf.OPENALEX.solr_url}/select',
            data={
                'df': 'title_abstract',
                'fq': filters,
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
    counts[(group, name)] = {}
    for (selection, fqs) in [
        ('Count (#nofilter)', []),
        ('Count (w/ abstract)', ['abstract:*']),
        ('Count (excl xpac)', ['is_xpac:false']),
        ('Count (w/ abstract, excl xpac)', ['abstract:*', 'is_xpac:false']),
        ('Count (excl xpac, 1990–2024)', ['is_xpac:false', 'publication_year:[1990 TO 2024]']),
        ('Count (excl xpac, 1990–2024, language:en)', ['is_xpac:false', 'publication_year:[1990 TO 2024]', 'language:en']),
    ]:
        start = time.time()
        cnt = count(query, filters=fqs)
        end = time.time()
        if type(cnt) is int:
            counts[(group, name)] |= {selection: cnt}
            cnt = f'{cnt:,}'
        else:
            counts[(group, name)] |= {selection: pd.NA}
        logger.info(f'{group} ({name}): {cnt}   | took {time.time() - start:2f} seconds')

pd.DataFrame(counts).T.to_csv('notes/2026-04-29_counts.csv', index=True, header=True)
