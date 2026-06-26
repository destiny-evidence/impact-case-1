import re
import pandas as pd
from httpx import Client
from nacsos_data.util.conf import load_settings
from ic1.core.config import CONF_FILE
from ic1.query.revisions import MERGED

wc = re.compile(r'["*\-\s](\w+?)\*')

conf = load_settings(CONF_FILE)
stats = []
with Client(timeout=120) as client:
    for term in set(wc.findall(MERGED['CLIMATE AND HEALTH'])):
        url = (
            f'{conf.OPENALEX.solr_url}/terms'
            f'?facet=true'
            f'&indent=true'
            f'&q.op=OR'
            f'&q=*%3A*'
            f'&terms.fl=title_abstract'
            f'&terms.limit=100'
            f'&terms.prefix={term}'
            f'&terms.stats=true'
            f'&terms.ttf=true'
            f'&terms=true'
            f'&useParams='
        )
        print(url)

        response = client.get(url)
        terms = response.json()['terms']['title_abstract']
        stats.extend(
            [
                {
                    'prefix': term,
                    'term': terms[i],
                    'df': terms[i + 1]['df'],
                    'ttf': terms[i + 1]['ttf'],
                }
                for i in range(0, len(terms), 2)
            ],
        )

(pd.DataFrame(stats).sort_values(['prefix', 'df'], ascending=False).to_csv('notes/2026-02-12_wildcards-new_index.csv', index=False))

df = pd.read_csv('notes/2026-02-12_wildcards-new_index.csv')
for g, vs in df[df['df'] > 10].groupby('prefix'):
    print(f"'{g}': {vs['term'].tolist()}")
