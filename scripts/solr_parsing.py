import re
import logging

import httpx

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

def count(q: str) -> str:
    try:
        res = httpx.post(
            f'{conf.OPENALEX.solr_url}/select', data={
                'df': 'title_abstract',
                'defType': 'lucene',
                'q': q,
                'q.op': 'AND',
                'rows': 5,
            }, timeout=120,
        ).json()
        return f'{res['response']['numFound']:,}'
    except KeyError:
        return res['error']['msg']

endings = [
    '',
    '?',
    '??',
    '*',
    '~',
    '~1',
    '~2'
]
endings += [f's{e}' for e in endings]

QUERIES = [
              f'school uniform{ending}' for ending in endings
          ] + [
              f'"school uniform{ending}"' for ending in endings
          ] + [
              f'school W uniform{ending}' for ending in endings
          ] + [
              f'school N uniform{ending}' for ending in endings
          ] + [
              f'"school uniform{ending}"~2' for ending in endings
          ] + [
              f'school 2W uniform{ending}' for ending in endings
          ] + [
              f'{{!complexphrase v=\'"school uniform{ending}"\'}}' for ending in endings
          ] + [
              f'{{!complexphrase v=\'school W uniform{ending}\'}}' for ending in endings
          ] + [
              f'{{!complexphrase v=\'school N uniform{ending}\'}}' for ending in endings
          ] + [
              f'{{!surround v=\'"school uniform{ending}"\'}}' for ending in endings
          ] + [
              f'{{!surround v=\'school W uniform{ending}\'}}' for ending in endings
          ] + [
              f'{{!surround v=\'school N uniform{ending}\'}}' for ending in endings
          ] + [
              'uniform',
              'uniform?',
              'uniform??',
              'uniforme?',
              'uniformed',
          ]

for q in QUERIES:
    print(f'{q}   -> ', end='')
    print(count(q))


# school uniform   ->  7,876
# school uniform?   ->  1,892
# school uniform??   ->  2,948
# school uniform*   ->  13,701
# school uniform~   ->  7,879
# school uniform~1   ->  7,879
# school uniform~2   ->  7,879
# school uniforms   ->  1,826
# school uniforms?   ->  0
# school uniforms??   ->  3
# school uniforms*   ->  1,840
# school uniforms~   ->  1,947
# school uniforms~1   ->  1,947
# school uniforms~2   ->  1,947
# "school uniform"   ->  696
# "school uniform?"   ->  696
# "school uniform??"   ->  696
# "school uniform*"   ->  696
# "school uniform~"   ->  696
# "school uniform~1"   ->  0
# "school uniform~2"   ->  1
# "school uniforms"   ->  850
# "school uniforms?"   ->  850
# "school uniforms??"   ->  850
# "school uniforms*"   ->  850
# "school uniforms~"   ->  850
# "school uniforms~1"   ->  1
# "school uniforms~2"   ->  1
# school W uniform   ->  569
# school W uniform?   ->  47
# school W uniform??   ->  266
# school W uniform*   ->  921
# school W uniform~   ->  569
# school W uniform~1   ->  569
# school W uniform~2   ->  569
# school W uniforms   ->  44
# school W uniforms?   ->  0
# school W uniforms??   ->  1
# school W uniforms*   ->  45
# school W uniforms~   ->  54
# school W uniforms~1   ->  54
# school W uniforms~2   ->  54
# school N uniform   ->  867
# school N uniform?   ->  68
# school N uniform??   ->  498
# school N uniform*   ->  1,487
# school N uniform~   ->  867
# school N uniform~1   ->  867
# school N uniform~2   ->  867
# school N uniforms   ->  62
# school N uniforms?   ->  0
# school N uniforms??   ->  1
# school N uniforms*   ->  64
# school N uniforms~   ->  93
# school N uniforms~1   ->  93
# school N uniforms~2   ->  93
# "school uniform"~2   ->  841
# "school uniform?"~2   ->  841
# "school uniform??"~2   ->  841
# "school uniform*"~2   ->  841
# "school uniform~"~2   ->  841
# "school uniform~1"~2   ->  1
# "school uniform~2"~2   ->  1
# "school uniforms"~2   ->  923
# "school uniforms?"~2   ->  923
# "school uniforms??"~2   ->  923
# "school uniforms*"~2   ->  923
# "school uniforms~"~2   ->  923
# "school uniforms~1"~2   ->  3
# "school uniforms~2"~2   ->  2
# school 2W uniform   ->  3
# school 2W uniform?   ->  0
# school 2W uniform??   ->  0
# school 2W uniform*   ->  3
# school 2W uniform~   ->  3
# school 2W uniform~1   ->  3
# school 2W uniform~2   ->  3
# school 2W uniforms   ->  0
# school 2W uniforms?   ->  0
# school 2W uniforms??   ->  0
# school 2W uniforms*   ->  0
# school 2W uniforms~   ->  0
# school 2W uniforms~1   ->  0
# school 2W uniforms~2   ->  0
# {!complexphrase v='"school uniform"'}   ->  696
# {!complexphrase v='"school uniform?"'}   ->  850
# {!complexphrase v='"school uniform??"'}   ->  12
# {!complexphrase v='"school uniform*"'}   ->  1,378
# {!complexphrase v='"school uniform~"'}   ->  696
# {!complexphrase v='"school uniform~1"'}   ->  696
# {!complexphrase v='"school uniform~2"'}   ->  696
# {!complexphrase v='"school uniforms"'}   ->  850
# {!complexphrase v='"school uniforms?"'}   ->  0
# {!complexphrase v='"school uniforms??"'}   ->  1
# {!complexphrase v='"school uniforms*"'}   ->  856
# {!complexphrase v='"school uniforms~"'}   ->  853
# {!complexphrase v='"school uniforms~1"'}   ->  853
# {!complexphrase v='"school uniforms~2"'}   ->  853
# {!complexphrase v='school W uniform'}   ->  569
# {!complexphrase v='school W uniform?'}   ->  47
# {!complexphrase v='school W uniform??'}   ->  266
# {!complexphrase v='school W uniform*'}   ->  921
# {!complexphrase v='school W uniform~'}   ->  569
# {!complexphrase v='school W uniform~1'}   ->  569
# {!complexphrase v='school W uniform~2'}   ->  569
# {!complexphrase v='school W uniforms'}   ->  44
# {!complexphrase v='school W uniforms?'}   ->  0
# {!complexphrase v='school W uniforms??'}   ->  1
# {!complexphrase v='school W uniforms*'}   ->  45
# {!complexphrase v='school W uniforms~'}   ->  54
# {!complexphrase v='school W uniforms~1'}   ->  54
# {!complexphrase v='school W uniforms~2'}   ->  54
# {!complexphrase v='school N uniform'}   ->  867
# {!complexphrase v='school N uniform?'}   ->  68
# {!complexphrase v='school N uniform??'}   ->  498
# {!complexphrase v='school N uniform*'}   ->  1,487
# {!complexphrase v='school N uniform~'}   ->  867
# {!complexphrase v='school N uniform~1'}   ->  867
# {!complexphrase v='school N uniform~2'}   ->  867
# {!complexphrase v='school N uniforms'}   ->  62
# {!complexphrase v='school N uniforms?'}   ->  0
# {!complexphrase v='school N uniforms??'}   ->  1
# {!complexphrase v='school N uniforms*'}   ->  64
# {!complexphrase v='school N uniforms~'}   ->  93
# {!complexphrase v='school N uniforms~1'}   ->  93
# {!complexphrase v='school N uniforms~2'}   ->  93
# {!surround v='"school uniform"'}   ->  0
# {!surround v='"school uniform?"'}   ->  0
# {!surround v='"school uniform??"'}   ->  0
# {!surround v='"school uniform*"'}   ->  0
# {!surround v='"school uniform~"'}   ->  0
# {!surround v='"school uniform~1"'}   ->  0
# {!surround v='"school uniform~2"'}   ->  0
# {!surround v='"school uniforms"'}   ->  0
# {!surround v='"school uniforms?"'}   ->  0
# {!surround v='"school uniforms??"'}   ->  0
# {!surround v='"school uniforms*"'}   ->  0
# {!surround v='"school uniforms~"'}   ->  0
# {!surround v='"school uniforms~1"'}   ->  0
# {!surround v='"school uniforms~2"'}   ->  0
# {!surround v='school W uniform'}   ->  696
# {!surround v='school W uniform?'}   ->  850
# {!surround v='school W uniform??'}   ->  12
# {!surround v='school W uniform*'}   ->  Exceeded maximum of 1000 basic queries.
# {!surround v='school W uniform~'}   ->  0
# {!surround v='school W uniform~1'}   ->  0
# {!surround v='school W uniform~2'}   ->  0
# {!surround v='school W uniforms'}   ->  850
# {!surround v='school W uniforms?'}   ->  0
# {!surround v='school W uniforms??'}   ->  1
# {!surround v='school W uniforms*'}   ->  856
# {!surround v='school W uniforms~'}   ->  0
# {!surround v='school W uniforms~1'}   ->  0
# {!surround v='school W uniforms~2'}   ->  0
# {!surround v='school N uniform'}   ->  740
# {!surround v='school N uniform?'}   ->  861
# {!surround v='school N uniform??'}   ->  15
# {!surround v='school N uniform*'}   ->  Exceeded maximum of 1000 basic queries.
# {!surround v='school N uniform~'}   ->  0
# {!surround v='school N uniform~1'}   ->  0
# {!surround v='school N uniform~2'}   ->  0
# {!surround v='school N uniforms'}   ->  860
# {!surround v='school N uniforms?'}   ->  0
# {!surround v='school N uniforms??'}   ->  1
# {!surround v='school N uniforms*'}   ->  866
# {!surround v='school N uniforms~'}   ->  0
# {!surround v='school N uniforms~1'}   ->  0
# {!surround v='school N uniforms~2'}   ->  0
# uniform   ->  816,005
# uniform?   ->  29,822
# uniform??   ->  246,461
# uniforme?   ->  13,605
# uniformed   ->  7,414