import logging
import re
import httpx
from nacsos_data.util.conf import load_settings
from ic1.core.config import CONF_FILE
from ic1.query.revisions.query_20260408 import MERGED

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='INFO')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
# logging.getLogger('root').setLevel(logging.WARNING)
logger = logging.getLogger('counting')

conf = load_settings(CONF_FILE)


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


query = re.sub(r'\s+', ' ', MERGED['(CLIMATE AND HEALTH) OR ADAPTATION'])
for fltr in [
    [],
    ['is_xpac:false'],
    ['is_xpac:false', 'publication_year:[* TO 1990]'],
    ['is_xpac:false', 'publication_year:[* TO 1990]', 'language:en'],
]:
    cnt = count(query, filters=fltr)
    logger.info(f'Found {cnt:,} with {fltr}')
