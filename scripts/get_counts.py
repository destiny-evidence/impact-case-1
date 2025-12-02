import re
import logging

from httpx import HTTPStatusError

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
print(conf.OPENALEX)

comm = re.compile(r'# .*\n')
ws = re.compile(r'\s+')
wild = re.compile(r'[*?]+')
near = re.compile(r'W/\d')
phrase = re.compile(r'"([^"]+)"')

oa_api = OpenAlexAPI()
oa_solr = OpenAlexSolrAPI(openalex_conf=conf.OPENALEX, logger=logger)

for QUERIES in [MERGED, CLIMATE, HEALTH]:
    for k, query in QUERIES.items():
        print(k)
        query = comm.sub('', query)
        query = ws.sub(' ', query)
        print(f'  -> Original query: {query}')
        query = near.sub('AND', query)
        query = query.replace('AND NOT', 'NOT')
        print(f'  -> NEAR/END removed: {query}')
        query_nowc = wild.sub('', query)
        print(f'  -> wildcards removed: {query}')
        query_solr = phrase.sub(lambda m: m.group(1).replace(' ', ' W '), query)
        print(f'  -> solr: {query_solr}')
        query_solr_nowc = wild.sub('', query_solr)
        print(f'  -> wildcards removed: {query_solr_nowc}')
        print('  ---')
        try:
            print(f'  -> API: {oa_api.get_count(query="title_and_abstract.search:" + query_nowc, params={"include_xpac": False}):,}')
        except HTTPStatusError:
            print('  -> API: -ERROR-')
        try:
            print(f'  -> API + xpac: {oa_api.get_count(query="title_and_abstract.search:" + query_nowc, params={"include_xpac": True}):,}')
        except HTTPStatusError:
            print('  -> API + xpac: -ERROR-')
        print(f'  -> solr (as API): {oa_solr.get_count(query=query_nowc).num_found:,}')
        print(f'  -> solr: {oa_solr.get_count(query=query_solr).num_found:,}')
        print(f'  -> solr w/o wildcards: {oa_solr.get_count(query=query_solr_nowc).num_found:,}')
