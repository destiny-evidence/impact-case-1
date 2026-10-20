"""Route the 649 test records to blind re-screen reviewers for the label-noise experiment (#28).

Each record gets a cell: its original label type (unanimous coders or resolved) crossed with whether the
deployed high-recall LLM agrees with that label. Resolved records also get the resolver who originally
resolved them. Resolvers come from the round-1 resolver sheet by assignment scope, and from the resolution
names in the database where the sheet has no entry. Resolved records are routed either back to that
resolver (self) or to the other available resolver (cross). EM is no longer available, so resolved records
by EM are forced to cross. Records resolved by both AM and MIM are routed to AM and MIM in alternation, and
marked 'both'. Unanimous records are split between AM and MIM.

The output carries resolver identity, so it is written to the private directory.
"""

import asyncio
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.schemas.bot_annotations import BotAnnotationMetaData
from nacsos_data.util.annotations import get_ordering

from ic1.core.config import settings, TASKS, TaskName

GOLD_CSV = Path(
    'ic1/deet/projects/inout/data-extraction-experiments/2026-09-03_18-58-32_FULL_ABSTRACT_TEST/goldstandard_llm_comparison.csv',
)
EXPORT_CSV = Path('data/private/exports/inout.csv')
ROUND_1_XLSX = Path('data/private/Round1_ReviewerNoise-and-Counting_20072026_MIM_AM_EM.xlsx')
OUT_CSV = Path('data/private/reannotation/routing.csv')

AVAILABLE = ['AM', 'MIM']
SEED = 20261013

# Every disagreement record is re-screened. Agreement controls are random samples from their cells.
DISAGREEMENT_CELLS = ['unanimous-disagree', 'resolved-disagree']
CONTROL_SAMPLE = {'unanimous-agree': 30, 'resolved-agree': 50}

# Resolution names end in the resolver's first name, in brackets.
NAME_TO_RESOLVER = {'Andres': 'AM', 'Lotte': 'EM', 'Maria-Inti': 'MIM'}
NAME_PATTERN = re.compile(r'\(([^()]*)\)\s*$')

# This scope has two resolution rows. AM resolved the first SPLIT_POSITION items in item order, and both AM and
# MIM resolved the rest.
SPLIT_SCOPE = 'eb958485-753e-4c75-ba1c-5a5a9f99cbd7'
SPLIT_POSITION = 86


def load_test_records() -> pd.DataFrame:
    """One row per test document with the deployed high-recall LLM verdict."""
    gold = pd.read_csv(GOLD_CSV, dtype=str)
    high_recall = gold[gold.attribute_label == 'include - high recall']
    return pd.DataFrame({'item_id': high_recall.external_id.values, 'llm_include': (high_recall.llm_extraction == 'True').values})


def load_original_labels() -> pd.DataFrame:
    """One row per item with its original label, whether the coders disagreed, and its assignment scope.

    The export gives every item a RESOLVED row, so disagreement is read from the coder rows: an item is
    resolved when its coders gave more than one distinct decision. Unanimous items take the coders'
    shared decision. Resolved items take the RESOLVED row's decision.
    """
    export = pd.read_csv(EXPORT_CSV, dtype=str, keep_default_na=False)
    export['label'] = pd.to_numeric(export['incl|1'], errors='coerce')

    coders = export[export.username != 'RESOLVED'].groupby('item_id').agg(
        distinct=('label', 'nunique'),
        coder_label=('label', 'first'),
        scope=('assignment_scope_id', 'first'),
    )
    resolutions = export[export.username == 'RESOLVED'].set_index('item_id')[['label']]

    out = coders.join(resolutions, how='left')
    out['resolved'] = out.distinct > 1
    out['label'] = out.label.where(out.resolved, out.coder_label)
    return out[['label', 'resolved', 'scope']]


def load_scope_resolvers() -> dict[str, str]:
    """Map assignment scope id to the resolver who resolved it, from the round-1 resolver sheet."""
    counts = pd.read_excel(ROUND_1_XLSX, sheet_name='Counting_Resolutions_Inclusion', dtype=str)
    counts = counts[counts.Resolver.isin(AVAILABLE + ['EM', 'AM/MIM'])]
    return dict(zip(counts.ID, counts.Resolver))


async def load_resolver_sources() -> tuple[dict[str, str], dict[str, str]]:
    """Resolvers from the resolution names, and the per-item resolver for the split scope.

    Returns (scope id -> resolver, item id -> resolver for SPLIT_SCOPE).
    """
    scheme_id = TASKS[TaskName.INOUT].scheme_id
    db_engine = get_engine_async(settings=settings.DB)

    async with db_engine.session() as session:
        bots = (
            await session.execute(
                sa.select(BotAnnotationMetaData.assignment_scope_id, BotAnnotationMetaData.name).where(
                    BotAnnotationMetaData.annotation_scheme_id == scheme_id,
                ),
            )
        ).all()
        order = await get_ordering(session=session, assignment_scope_id=SPLIT_SCOPE)

    name_resolvers = {}
    for scope_id, name in bots:
        match = NAME_PATTERN.search(name or '')
        if str(scope_id) == SPLIT_SCOPE or match is None or match.group(1).strip() not in NAME_TO_RESOLVER:
            continue
        name_resolvers[str(scope_id)] = NAME_TO_RESOLVER[match.group(1).strip()]

    split_items = {
        str(entry.item_id): ('AM' if position < SPLIT_POSITION else 'AM/MIM') for position, entry in enumerate(order)
    }
    return name_resolvers, split_items


def route_records(name_resolvers: dict[str, str], split_items: dict[str, str], seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    test = load_test_records()
    labels = load_original_labels()
    scope_resolvers = {**name_resolvers, **load_scope_resolvers()}

    df = test.merge(labels.reset_index(), on='item_id', how='left', validate='one_to_one')
    df['label_type'] = np.where(df.resolved.fillna(False).astype(bool), 'resolved', 'unanimous')
    df['llm_agrees'] = df.label.notna() & (df.llm_include == (df.label == 1))
    df['cell'] = df.label_type + '-' + np.where(df.llm_agrees, 'agree', 'disagree')
    df.loc[df.label.isna(), 'cell'] = 'unlabelled'
    df['original_resolver'] = df.item_id.map(split_items).fillna(df.scope.map(scope_resolvers))

    df['in_task'] = df.cell.isin(DISAGREEMENT_CELLS)
    for cell, n in CONTROL_SAMPLE.items():
        pool = df.index[df.cell == cell]
        df.loc[rng.choice(pool, size=n, replace=False), 'in_task'] = True
    sampled = df[df.in_task].cell.value_counts()
    df['sampling_fraction'] = df.cell.map(sampled / df.cell.value_counts())

    df['arm'] = None
    df['reviewer'] = None

    unanimous = (df.label_type == 'unanimous') & df.label.notna() & df.in_task
    order = rng.permutation(np.flatnonzero(unanimous))
    df.loc[df.index[order], 'reviewer'] = [AVAILABLE[i % len(AVAILABLE)] for i in range(len(order))]
    df.loc[unanimous, 'arm'] = 'unanimous'

    resolved = (df.label_type == 'resolved') & df.label.notna() & df.in_task
    both_count = 0
    for idx in df.index[resolved]:
        resolver = df.at[idx, 'original_resolver']
        if resolver in AVAILABLE:
            if rng.random() < 0.5:
                df.at[idx, 'arm'], df.at[idx, 'reviewer'] = 'self', resolver
            else:
                df.at[idx, 'arm'], df.at[idx, 'reviewer'] = 'cross', AVAILABLE[1 - AVAILABLE.index(resolver)]
        elif resolver == 'EM':
            df.at[idx, 'arm'], df.at[idx, 'reviewer'] = 'cross', AVAILABLE[rng.integers(len(AVAILABLE))]
        elif resolver == 'AM/MIM':
            df.at[idx, 'arm'], df.at[idx, 'reviewer'] = 'both', AVAILABLE[both_count % len(AVAILABLE)]
            both_count += 1
        else:
            df.at[idx, 'arm'] = 'unmatched'

    # Unanimous records can go to either reviewer, so they move from the heavier load to the lighter one.
    loads = df.reviewer.value_counts()
    heavy, light = loads.idxmax(), loads.idxmin()
    move = (loads[heavy] - loads[light]) // 2
    candidates = df.index[(df.arm == 'unanimous') & (df.reviewer == heavy)]
    df.loc[rng.choice(candidates, size=move, replace=False), 'reviewer'] = light

    return df[
        ['item_id', 'cell', 'label_type', 'llm_agrees', 'label', 'scope', 'original_resolver', 'in_task', 'sampling_fraction', 'arm', 'reviewer']
    ]


def main() -> None:
    name_resolvers, split_items = asyncio.run(load_resolver_sources())
    routing = route_records(name_resolvers, split_items)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    routing.to_csv(OUT_CSV, index=False)

    print(f'Wrote {len(routing)} records to {OUT_CSV}')
    print(pd.crosstab(routing.cell, routing.arm, margins=True).to_string())
    print(routing.reviewer.value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
