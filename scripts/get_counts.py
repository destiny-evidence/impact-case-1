import re
import logging
import httpx
from httpx import HTTPStatusError

# from nacsos_data.util.academic.apis import OpenAlexAPI, OpenAlexSolrAPI
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
# print(conf.OPENALEX)

comm = re.compile(r'# .*\n')
ws = re.compile(r'\s+')
wild = re.compile(r'[*?]+')
near = re.compile(r'W/(\d)')
phrase = re.compile(r'"([^"]+)"')

# oa_api = OpenAlexAPI(logger=logger)
# oa_solr = OpenAlexSolrAPI(openalex_conf=conf.OPENALEX, logger=logger)


# def count(q: str) -> str:
#     try:
#         res = httpx.post(
#             f'{conf.OPENALEX.solr_url}/select',
#             data={
#                 'df': 'title_abstract',
#                 'defType': 'lucene',
#                 'q': q,
#                 'q.op': 'AND',
#                 'rows': 5,
#             },
#             timeout=120,
#         ).json()
#         return f'{res["response"]["numFound"]:,}'
#     except KeyError:
#         return res['error']

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

        queries = {
            'title_abstract (-xpac)': lambda: httpx.get(
                'https://api.openalex.org/works',
                params={
                    'filter': f'title_and_abstract.search:{query_api}',
                    'select': 'id',
                    'per-page': 1,
                    'include_xpac': False,
                },
                timeout=120,
                headers={'api_key': conf.OPENALEX.API_KEY},
            ),
            'title_abstract (+xpac)': lambda: httpx.get(
                'https://api.openalex.org/works',
                params={
                    'filter': f'title_and_abstract.search:{query_api}',
                    'select': 'id',
                    'per-page': 1,
                    'include_xpac': True,
                },
                timeout=120,
                headers={'api_key': conf.OPENALEX.API_KEY},
            ),
            'search (+xpac)': lambda: httpx.get(
                'https://api.openalex.org/works',
                params={
                    'search': query_api,
                    'select': 'id',
                    'per-page': 1,
                    'include_xpac': True,
                    'api_key': conf.OPENALEX.API_KEY,
                },
                timeout=120,
            ),
            'search.exact (+xpac)': lambda: httpx.get(
                'https://api.openalex.org/works',
                params={
                    'search.exact': query_api,
                    'select': 'id',
                    'per-page': 1,
                    'include_xpac': True,
                    'api_key': conf.OPENALEX.API_KEY,
                },
                timeout=120,
            ),
        }

        for kq, req in queries.items():
            try:
                page = req()
                page.raise_for_status()
                res = page.json()
                print(f'  -> {kq}: {res["meta"]["count"]:,}')
            except HTTPStatusError as e:
                print(f'  -> {kq}: -ERROR-  -> {ws.sub(" ", page.text)}')

        #
        #
        # query_solr = phrase.sub(lambda m: m.group(1).replace(' ', ' W '), query)
        # query_solr = query_solr.replace('AND NOT', 'NOT')
        # query_solr = query_solr.replace('$', '?')
        # query_solr = query_solr.replace(' *', ' ')
        # query_solr = near.sub(lambda m: f'{int(m.group(1)) + 1}W', query_solr)
        # print(f'  -> phrases as W: {query_solr}')
        # query_solr_nowc = wild.sub('', query_solr)
        # print(f'  -> phrases as W w/o wildcards: {query_solr_nowc}')
        #
        # query_solr_quoted = query.replace('AND NOT', 'NOT')
        # query_solr_quoted = query_solr_quoted.replace('$', '?')
        # query_solr_quoted = query_solr_quoted.replace(' *', ' ')
        # query_solr_quoted = near.sub(lambda m: f'{int(m.group(1)) + 1}W', query_solr_quoted)
        # print(f'  -> phrases as quotes: {query_solr_quoted}')
        # query_solr_quoted_nowc = wild.sub('', query_solr_quoted)
        # print(f'  -> phrases as quotes w/o wildcards: {query_solr_quoted_nowc}')
        #
        # print('  ---')
        #
        # q = parse(query, expansions=expansions)
        # print(f'  -> surround | processed: ', end='')
        # print(count(f'{{!surround maxBasicQueries=100000}} {q}'))
        #
        # Qs = [
        #     ('phrases as W', query_solr),
        #     ('phrases as W w/o wildcards', query_solr_nowc),
        #     ('phrases as quotes', query_solr_quoted),
        #     ('phrases as quotes w/o wildcards', query_solr_quoted_nowc)
        # ]
        #
        # for desc, q in Qs:
        #     print(f'  -> standard | {desc}: ', end='')
        #     print(count(q))
        #
        # for desc, q in Qs:
        #     print(f'  -> complexphrase | {desc}: ', end='')
        #     print(count(f'{{!complexphrase v=\'{q}\'}}'))
        #
        # for desc, q in Qs:
        #     if desc != 'phrases as W':
        #         continue
        #     print(f'  -> surround | {desc}: ', end='')
        #     print(count(f'{{!surround maxBasicQueries=100000}} {q}'))
