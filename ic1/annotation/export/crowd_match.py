"""Resolve Cochrane Crowd "Climate Quest" records to NACSOS items and write the crowd labels.

The crowd CSVs carry no NACSOS id. Without `--ris`, records are matched to the project's items on DOI, then on
normalised title (+ year). With `--ris` (the file imported into the Crowd platform, whose `ID` field is the
NACSOS item_id), records are matched against the RIS entries instead, falling back to the position of the record
in the import when the platform's CRS IDs are shown to follow the RIS order. The DB matching is then a cross-check.
Items already assigned to an evaluation split are dropped from the crowd set.
"""

import logging
import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import sqlalchemy as sa
import typer
from nacsos_data.db import get_engine

from ic1.core.config import settings, TASKS, TaskName
from ic1.core.ids import PROJECT_ID
from ic1.evaluation_splits.splits_model import EvaluationSplits

logger = logging.getLogger(__name__)

TASK = TASKS[TaskName.INOUT]
MIN_TEXT_LEN = 20
POSITION_AGREEMENT = 0.99  # share of key-matched records that must agree with the CRS-ID/RIS-position offset
RIS_FIELDS = {'ID': 'item_id', 'M3': 'item_id', 'MISC3': 'item_id', 'DO': 'doi', 'TI': 'title', 'T1': 'title', 'PY': 'year', 'Y1': 'year', 'DA': 'year'}
UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)


def norm_doi(doi: object) -> str | None:
    if not isinstance(doi, str) or not doi.strip():
        return None
    return re.sub(r'^(https?://(dx\.)?doi\.org/|doi:)', '', doi.strip().lower()) or None


def norm_title(title: object) -> str | None:
    if not isinstance(title, str):
        return None
    out = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', title.lower())).strip()
    return out or None


def add_keys(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """Add normalised DOI, title and title+year keys."""
    df = df.copy()
    df['doi_n'] = df['doi'].map(norm_doi)
    df['title_n'] = df['title'].map(norm_title)
    year = pd.to_numeric(df[year_col], errors='coerce').astype('Int64').astype(str)
    df['ty'] = df['title_n'].where(df['title_n'].isna(), df['title_n'] + '|' + year)
    return df


def read_ris(path: Path) -> pd.DataFrame:
    """One row per RIS record, in file order: item_id, doi, title, year."""
    rows: list[dict] = []
    cur: dict = {}
    for line in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        m = re.match(r'^([A-Z][A-Z0-9]*)\s+-\s?(.*)$', line)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        field = RIS_FIELDS.get(tag)
        if tag == 'TY':
            cur = {}
        elif tag == 'ER':
            rows.append(cur)
            cur = {}
        elif field == 'item_id':
            if UUID.match(val):
                cur.setdefault('item_id', val.lower())
        elif field:
            cur.setdefault(field, val[:4] if field == 'year' else val)
    df = pd.DataFrame(rows)
    if df.empty or 'item_id' not in df:
        raise ValueError(f'No NACSOS item ids (UUIDs in an ID field) found in {path}')
    for col in ('doi', 'title', 'year'):
        if col not in df:
            df[col] = None
    if df['item_id'].isna().any() or df['item_id'].duplicated().any():
        raise ValueError(f'{path}: records without an item id, or duplicated item ids')
    return df[['item_id', 'doi', 'title', 'year']]


def _unique_map(df: pd.DataFrame, key: str) -> tuple[dict[str, str], set[str]]:
    """key -> item_id for keys that occur once, plus the set of keys that occur more than once."""
    d = df.dropna(subset=[key])
    counts = d[key].value_counts()
    ambiguous = set(counts[counts > 1].index)
    unique = d[~d[key].isin(ambiguous)].set_index(key)['item_id'].astype(str).to_dict()
    return unique, ambiguous


def match_records(crowd: pd.DataFrame, ref: pd.DataFrame, prefix: str) -> tuple[list[str | None], list[str]]:
    """Match crowd records to reference rows (columns item_id, doi_n, ty, title_n) on DOI, title+year, then title.

    A title-based match is rejected when both sides have a DOI and the DOIs conflict (neither is a prefix of the other,
    which tolerates the suffixes the crowd platform truncates), so generic titles cannot link different papers.
    """
    ref_doi = ref.set_index('item_id')['doi_n'].to_dict()

    def conflict(doi: str | None, iid: str) -> bool:
        other = ref_doi.get(iid)
        return bool(isinstance(doi, str) and isinstance(other, str) and not (doi.startswith(other) or other.startswith(doi)))

    doi_map, doi_amb = _unique_map(ref, 'doi_n')
    ty_map, ty_amb = _unique_map(ref, 'ty')
    title_map, title_amb = _unique_map(ref, 'title_n')
    ids: list[str | None] = []
    how: list[str] = []
    for _, r in crowd.iterrows():
        iid, m = None, 'unmatched'
        if r['doi_n'] in doi_map:
            iid, m = doi_map[r['doi_n']], 'doi'
        elif pd.notna(r['ty']) and r['ty'] in ty_map and not conflict(r['doi_n'], ty_map[r['ty']]):
            iid, m = ty_map[r['ty']], 'title+year'
        elif pd.notna(r['title_n']) and r['title_n'] in title_map and not conflict(r['doi_n'], title_map[r['title_n']]):
            iid, m = title_map[r['title_n']], 'title'
        elif r['doi_n'] in doi_amb or r['ty'] in ty_amb or r['title_n'] in title_amb:
            m = 'ambiguous'
        ids.append(iid)
        how.append(f'{prefix}:{m}' if iid else m)
    return ids, how


def apply_ris_position(crowd: pd.DataFrame, ris: pd.DataFrame) -> pd.DataFrame:
    """Fill unresolved records from the position of the record in the RIS if CRS IDs follow the RIS order.

    The offset (CRS ID minus RIS position) is estimated from records that matched on their keys; positions
    are only used if almost all of those records agree on one offset.
    """
    pos = {iid: i for i, iid in enumerate(ris['item_id'])}
    crs = pd.to_numeric(crowd['crs_id'], errors='coerce')
    keyed = crowd['item_id'].notna() & crs.notna() & crowd['item_id'].isin(pos)
    offsets = (crs[keyed] - crowd.loc[keyed, 'item_id'].map(pos)).astype(int)
    if offsets.empty:
        print('RIS position check: no key-matched records to estimate an offset from')
        return crowd
    offset = int(offsets.mode().iloc[0])
    agreement = float((offsets == offset).mean())
    print(f'\n== RIS position check ==\nCRS ID - RIS position: modal offset {offset}, agreement {agreement:.1%} of {len(offsets)} key-matched records')
    if agreement < POSITION_AGREEMENT:
        print('CRS IDs do not follow the RIS order; position not used')
        return crowd

    crowd = crowd.copy()
    todo = crowd['item_id'].isna() & crs.notna()
    claimed = set(crowd['item_id'].dropna())
    filled = 0
    for i in crowd.index[todo]:
        p = int(crs[i]) - offset
        if 0 <= p < len(ris) and ris['item_id'].iat[p] not in claimed:
            crowd.at[i, 'item_id'] = ris['item_id'].iat[p]
            crowd.at[i, 'match'] = 'ris:position'
            claimed.add(ris['item_id'].iat[p])
            filled += 1
    print(f'filled {filled} of {int(todo.sum())} unresolved records from the RIS position')
    return crowd


def main(
    records: Annotated[Path, typer.Option(help='Crowd Records CSV')] = settings.CROWD_ROOT / 'Climate_Quest_Records_Nov2026v2.csv',
    ris: Annotated[Path | None, typer.Option(help='RIS file that was imported into the Crowd platform (ID = NACSOS item_id)')] = None,
    out_private: Annotated[Path, typer.Option(help='Sensitive output with title and text (DVC tier)')] = TASK.crowd_train_path,
    out_shareable: Annotated[Path, typer.Option(help='Shareable output: item ids and labels only')] = TASK.crowd_ids_path,
    project_id: Annotated[str, typer.Option()] = PROJECT_ID,
) -> None:
    crowd = pd.read_csv(records, encoding='utf-8-sig', dtype=str).rename(
        columns={'Crowd ID': 'crowd_id', 'Final Decision': 'decision', 'Title': 'title', 'Year': 'year', 'DOI': 'doi', 'CRS ID': 'crs_id'}
    )
    crowd['label'] = (crowd['decision'] == 'Possibly relevant').astype(int)
    crowd = add_keys(crowd, 'year')

    engine = get_engine(settings=settings.DB, debug=False)
    with engine.session() as session:
        items = pd.read_sql(
            sa.text(
                'SELECT a.item_id::text AS item_id, a.doi, a.title, i.text, a.publication_year '
                'FROM academic_item a JOIN item i ON i.item_id = a.item_id WHERE a.project_id = :p'
            ),
            session.connection(),
            params={'p': project_id},
        )
    logger.info(f'Loaded {len(items)} project items')
    items_k = add_keys(items, 'publication_year')

    # DB matching on the metadata in the crowd file
    crowd['db_item_id'], crowd['db_match'] = match_records(crowd, items_k, 'db')
    crowd['item_id'], crowd['match'] = crowd['db_item_id'], crowd['db_match']

    if ris is not None:
        ris_df = read_ris(ris)
        print(f'\n== RIS ==\n{len(ris_df)} records, {ris_df["item_id"].isin(items["item_id"]).sum()} found in NACSOS')
        ris_k = add_keys(ris_df, 'year')
        crowd['item_id'], crowd['match'] = match_records(crowd, ris_k, 'ris')
        crowd = apply_ris_position(crowd, ris_df)
        both = crowd['item_id'].notna() & crowd['db_item_id'].notna()
        agree = (crowd.loc[both, 'item_id'] == crowd.loc[both, 'db_item_id']).sum()
        print(f'RIS and DB matches agree on {agree} of {int(both.sum())} records where both resolved')
        fallback = crowd['item_id'].isna() & crowd['db_item_id'].notna()
        crowd.loc[fallback, ['item_id', 'match']] = crowd.loc[fallback, ['db_item_id', 'db_match']].to_numpy()
        print(f'{int(fallback.sum())} records resolved only by the DB match')
        absent = ris_df[~ris_df['item_id'].isin(crowd['item_id'].dropna())]
        out_missing = out_private.with_name(f'{TASK.name}_crowd_missing.csv')
        absent.to_csv(out_missing, index=False)
        print(f'{len(absent)} RIS records have no crowd result; listed in {out_missing}')

    print('\n== match method ==')
    print(crowd['match'].value_counts().to_string())
    matched = crowd[crowd['item_id'].notna()].copy()
    print(f'matched {len(matched)} of {len(crowd)}; distinct item_ids {matched["item_id"].nunique()}')
    print(f'records sharing an item_id with another record: {int(matched.duplicated("item_id", keep=False).sum())}')

    splits = EvaluationSplits.load(TASK.splits_path)
    print('\n== overlap with evaluation splits ==')
    in_split = pd.Series(False, index=matched.index)
    for name in ('deet', 'train', 'validation', 'test'):
        hit = matched['item_id'].isin(getattr(splits, name))
        print(f'{name:11s}: {int(hit.sum())}')
        in_split |= hit
    matched = matched[~in_split]

    not_in_db = ~matched['item_id'].isin(items['item_id'])
    print(f'\nitems not found in the project: {int(not_in_db.sum())}')
    matched = matched[~not_in_db]
    matched = matched.merge(items[['item_id', 'text']].rename(columns={'text': 'db_text'}), on='item_id', how='left')
    short = matched['db_text'].fillna('').str.len() < MIN_TEXT_LEN
    print(f'items with text shorter than {MIN_TEXT_LEN} chars: {int(short.sum())}')
    matched = matched[~short].drop_duplicates('item_id')
    items_idx = items.set_index('item_id')
    matched['title'] = matched['item_id'].map(items_idx['title'])
    matched['text'] = matched['db_text']

    print(f'\nusable crowd training items: {len(matched)}; positives {int(matched["label"].sum())} ({matched["label"].mean():.1%})')

    out_private.parent.mkdir(parents=True, exist_ok=True)
    matched[['item_id', 'crowd_id', 'crs_id', 'match', 'title', 'text', 'label']].to_csv(out_private, index=False)
    out_shareable.parent.mkdir(parents=True, exist_ok=True)
    matched[['item_id', 'label']].to_csv(out_shareable, index=False)
    unresolved = crowd[~crowd['crowd_id'].isin(matched['crowd_id'])]
    unresolved[['crowd_id', 'crs_id', 'match', 'title', 'year', 'doi']].to_csv(out_private.with_name(f'{TASK.name}_crowd_unmatched.csv'), index=False)
    print(f'wrote {out_private}, {out_shareable} and the list of {len(unresolved)} records that are not in the training set')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
