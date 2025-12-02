import re
import logging

from httpx import HTTPStatusError, Client

from nacsos_data.util.academic.apis import OpenAlexAPI, OpenAlexSolrAPI
from nacsos_data.util.conf import load_settings

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level='INFO')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
# logging.getLogger('root').setLevel(logging.WARNING)
logger = logging.getLogger('query')

conf = load_settings('.conf/secret.env')

logger.setLevel(60)
comm = re.compile(r'# .*\n')
ws = re.compile(r'\s+')
wild = re.compile(r'[*?]+')
near = re.compile(r'W/\d')
phrase = re.compile(r'"([^"]+)"')

oa_solr = OpenAlexSolrAPI(openalex_conf=conf.OPENALEX, logger=logger)

QUERIES = [
    '{!complexphrase v="(school W uniform) AND primary"}',
    '{!complexphrase v="(school N uniform) AND primary"}',
    '{!complexphrase v="\\"school uniform\\" AND primary"}',
    '{!complexphrase v="\\"school uniform\\""}',
    '{!complexphrase v="\\"school uniform*\\""}',
    '{!complexphrase v="\\"school uniform+\\""}',
    '{!complexphrase v="\\"school uniform?\\""}',
    '{!complexphrase v="\\"school uniform??\\""}',
    '{!complexphrase v="\\"school uniforms~\\""}',
    '{!complexphrase v="\\"school uniforms\\""}',
    '{!complexphrase v="\\"school uniforms*\\""}',
    '{!complexphrase v="\\"school uniforms+\\""}',
    '{!complexphrase v="\\"school uniforms?\\""}',
    '{!surround v="school N uniform"}',
    #'{!surround v="school N uniform*"}',
    '{!surround v="school N uniform?"}',
    '{!surround v="school N uniforms"}',
    #'{!surround v="school N uniforms*"}',
    '{!surround v="school N uniforms?"}',
    '{!surround v="school W uniform"}',
    # '{!surround v="school W uniform*"}',
    '{!surround v="school W uniform?"}',
    '{!surround v="school W uniforms"}',
    # '{!surround v="school W uniforms*"}',
    '{!surround v="school W uniforms?"}',
    '{!complexphrase v="school N uniform"}',
    '{!complexphrase v="school N uniform*"}',
    '{!complexphrase v="school N uniform?"}',
    '{!complexphrase v="school N uniforms"}',
    '{!complexphrase v="school N uniforms*"}',
    '{!complexphrase v="school N uniforms?"}',
    '{!complexphrase v="school W uniform"}',
    '{!complexphrase v="school W uniform*"}',
    '{!complexphrase v="school W uniform?"}',
    '{!complexphrase v="school W uniforms"}',
    '{!complexphrase v="school W uniforms*"}',
    '{!complexphrase v="school W uniforms?"}',
    '"school uniform"',
    '"school uniforms"',
    '"school uniform?"',
    '"school uniforms?"',
    '"school uniform*"',
    'school uniform?',
    'school uniform*',
    'school uniform',
    'school uniforms',
    'school N uniform?',
    'school N uniform*',
    'school N uniform',
    'school N uniforms',
    'school N uniforms~1',
    'school N uniforms~2',
    'school N uniform~1',
    'school N uniform~2',
    'school W uniform?',
    'school W uniform*',
    'school W uniform',
    'school W uniforms',
    'school W uniforms~1',
    'school W uniforms~2',
    'school W uniform~1',
    'school W uniform~2',
    'uniform~1',
    'uniform~2',
    'uniforms~1',
    'uniforms~2',
]

for q in QUERIES:
    res = OpenAlexSolrAPI(openalex_conf=conf.OPENALEX, logger=logger).get_count(q)
    print(f'{q}   ->  {res.num_found:,}')


# {!complexphrase v="(school W uniform) AND primary"}   ->  131
# {!complexphrase v="(school N uniform) AND primary"}   ->  212
# {!complexphrase v="\"school uniform\" AND primary"}   ->  67
# {!complexphrase v="\"school uniform\""}   ->  696
# {!complexphrase v="\"school uniform*\""}   ->  1,378
# {!complexphrase v="\"school uniform+\""}   ->  696
# {!complexphrase v="\"school uniform?\""}   ->  850
# {!complexphrase v="\"school uniform??\""}   ->  12
# {!complexphrase v="\"school uniforms~\""}   ->  853
# {!complexphrase v="\"school uniforms\""}   ->  850
# {!complexphrase v="\"school uniforms*\""}   ->  856
# {!complexphrase v="\"school uniforms+\""}   ->  850
# {!complexphrase v="\"school uniforms?\""}   ->  0
# {!surround v="school N uniform"}   ->  740
# {!surround v="school N uniform?"}   ->  861
# {!surround v="school N uniforms"}   ->  860
# {!surround v="school N uniforms?"}   ->  0
# {!surround v="school W uniform"}   ->  696
# {!surround v="school W uniform?"}   ->  850
# {!surround v="school W uniforms"}   ->  850
# {!surround v="school W uniforms?"}   ->  0
# {!complexphrase v="school N uniform"}   ->  867
# {!complexphrase v="school N uniform*"}   ->  1,487
# {!complexphrase v="school N uniform?"}   ->  68
# {!complexphrase v="school N uniforms"}   ->  62
# {!complexphrase v="school N uniforms*"}   ->  64
# {!complexphrase v="school N uniforms?"}   ->  0
# {!complexphrase v="school W uniform"}   ->  569
# {!complexphrase v="school W uniform*"}   ->  921
# {!complexphrase v="school W uniform?"}   ->  47
# {!complexphrase v="school W uniforms"}   ->  44
# {!complexphrase v="school W uniforms*"}   ->  45
# {!complexphrase v="school W uniforms?"}   ->  0
# "school uniform"   ->  696
# "school uniforms"   ->  850
# "school uniform?"   ->  696
# "school uniforms?"   ->  850
# "school uniform*"   ->  696
# school uniform?   ->  1,892
# school uniform*   ->  13,701
# school uniform   ->  7,876
# school uniforms   ->  1,826
# school N uniform?   ->  68
# school N uniform*   ->  1,487
# school N uniform   ->  867
# school N uniforms   ->  62
# school N uniforms~1   ->  93
# school N uniforms~2   ->  93
# school N uniform~1   ->  867
# school N uniform~2   ->  867
# school W uniform?   ->  47
# school W uniform*   ->  921
# school W uniform   ->  569
# school W uniforms   ->  44
# school W uniforms~1   ->  54
# school W uniforms~2   ->  54
# school W uniform~1   ->  569
# school W uniform~2   ->  569
# uniform~1   ->  816,383
# uniform~2   ->  816,383
# uniforms~1   ->  37,021
# uniforms~2   ->  37,021