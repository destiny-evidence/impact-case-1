import re
import logging
import httpx

from nacsos_data.util.academic.apis import OpenAlexAPI, OpenAlexSolrAPI
from nacsos_data.util.conf import load_settings
from query_revisions import CLIMATE, HEALTH, MERGED

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='INFO')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)
logger = logging.getLogger('query')
logger.setLevel(600)

conf = load_settings('.conf/secret.env')
#print(conf.OPENALEX)

comm = re.compile(r'# .*\n')
ws = re.compile(r'\s+')
wild = re.compile(r'[*?]+')
near = re.compile(r'W/(\d)')
phrase = re.compile(r'"([^"]+)"')

oa_api = OpenAlexAPI(logger=logger)
oa_solr = OpenAlexSolrAPI(openalex_conf=conf.OPENALEX, logger=logger)

for QUERIES in [MERGED, CLIMATE, HEALTH]:
    for k, query in QUERIES.items():
        print(k)

        query = comm.sub('', query)
        query = ws.sub(' ', query)
        print(f'  -> Original query: {query}')

        query_api = wild.sub('', query)
        query_api = near.sub('AND', query_api)
        query_api = query_api.replace('AND NOT', 'NOT')
        print(f'  -> OpenAlex API query: {query_api}')

        query_solr = phrase.sub(lambda m: m.group(1).replace(' ', ' W '), query)
        query_solr = query_solr.replace('AND NOT', 'NOT')
        query_solr = query_solr.replace('$', '?')
        query_solr = query_solr.replace(' *', ' ')
        query_solr = near.sub(lambda m: f'{int(m.group(1)) + 1}W', query_solr)
        print(f'  -> solr: {query_solr}')

        query_solr_nowc = wild.sub('', query_solr)
        print(f'  -> solr w/o wildcards: {query_solr_nowc}')

        print('  ---')

        try:
            print(f'  -> API: {oa_api.get_count(query="title_and_abstract.search:" + query_api, params={"include_xpac": False}):,}')
        except httpx.HTTPStatusError:
            print('  -> API: -ERROR-')
        try:
            print(f'  -> API + xpac: {oa_api.get_count(query="title_and_abstract.search:" + query_api, params={"include_xpac": True}):,}')
        except httpx.HTTPStatusError:
            print('  -> API + xpac: -ERROR-')

        try:
            res = httpx.post(
                f'{conf.OPENALEX.solr_url}/select', data={
                    'df': 'title_abstract',
                    'defType': 'lucene',
                    'q': query_api,
                    'q.op': 'AND',
                    'rows': 5,
                }, timeout=120,
            ).json()
            print(f'  -> solr (with API query):  {res['response']['numFound']:,}')

            res = httpx.post(
                f'{conf.OPENALEX.solr_url}/select', data={
                    'df': 'title_abstract',
                    'defType': 'lucene',
                    'q': f'{{!complexphrase v=\'{query_solr}\'}}',
                    'q.op': 'AND',
                    'rows': 5,
                }, timeout=120,
            ).json()
            print(f'  -> solr: {res['response']['numFound']:,}')

            res = httpx.post(
                f'{conf.OPENALEX.solr_url}/select', data={
                    'df': 'title_abstract',
                    'defType': 'lucene',
                    'q': f'{{!surround v=\'{query_solr_nowc}\'}}',
                    'q.op': 'AND',
                    'rows': 5,
                }, timeout=120,
            ).json()
            print(f'  -> solr w/o wildcards:  {res['response']['numFound']:,}')

        except KeyError as e:
            logging.exception(e)
            logging.error(res)
            raise e
