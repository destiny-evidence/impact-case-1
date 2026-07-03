"""Export NACSOS annotations for a scheme into versioned, two-tier artifacts.

Works for any scheme (IN/OUT and TAXONOMY). Uses ``prepare_export_table`` with an
explicit, complete ``labels`` list so every label/value gets a stable ``<key>|<value>``
column whether or not it was ever annotated (see ic1/annotation/scheme/import_taxonomy.py
and [[nacsos-choice-value-contiguous]]). Rows are long-format: one per (item, coder).

Each run writes a *versioned snapshot* with two tiers (see ic1-export-splits-design):

  * sensitive  -> data/exports/<task>/<version>/full.csv
      Full table incl. item text/title and REAL usernames. gitignored; DVC-tracked.
  * shareable  -> data/exports/<task>/<version>/shareable.csv
      Obfuscated: pseudonymous coders, keyed on item_id, label columns + non-sensitive
      metadata only (ids, publication_year, source); NO text/title. Publishable.

A tiny, non-sensitive provenance manifest is committed to git per version under
ic1/annotation/export/manifests/. The username->pseudonym map is stable across versions
and stored in .conf/ (sensitive, gitignored).
"""

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
import sqlalchemy as sa
import typer
from rich import print

from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.schemas.annotations import AnnotationScheme, Assignment, AssignmentScope
from nacsos_data.db.schemas.bot_annotations import BotAnnotationMetaData
from nacsos_data.models.annotations import AnnotationSchemeLabel
from nacsos_data.models.nql import FieldFilters
from nacsos_data.util.annotations.export import LabelOptions, prepare_export_table
from frictionless import Package as FPackage, Resource as FResource, validate as f_validate

from ic1.core.config import CONF_FILE
from ic1.core.ids import INOUT_SCHEME_ID, TAXONOMY_SCHEME_ID, TAXONOMY_SCOPE_IDS, INOUT_SCOPE_IDS
from ic1.annotation.scheme.import_taxonomy import MAPPING_JSON

class TaskConfig(TypedDict):
    scheme_id: str
    scope_ids: list[str]


TASKS: dict[str, TaskConfig] = {
    'inout': {'scheme_id': INOUT_SCHEME_ID, 'scope_ids': INOUT_SCOPE_IDS},
    'taxonomy': {'scheme_id': TAXONOMY_SCHEME_ID, 'scope_ids': TAXONOMY_SCOPE_IDS},
}

SENSITIVE_ROOT = Path('data/private/exports')  # gitignored; never committed
SHAREABLE_ROOT = Path('data/exports')  # git-tracked; Frictionless-described
PSEUDONYM_MAP = Path('.conf/coder_pseudonyms.json')  # gitignored; sensitive, stable

# Non-sensitive item metadata kept in the shareable tier (NEVER text/title/authors/abstract).
SHAREABLE_FIELDS = [
    {'name': 'item_id', 'type':'string', 'description': 'NACSOS item id (UUID) of item being annotated'},
    {'name': 'title', 'type':'string', 'description': 'Title of the record'},
    {'name': 'doi', 'type':'string', 'description': 'DOI (if known/present)'},
    {'name': 'wos_id', 'type':'string', 'description': 'Web of Science ID (if known/present)'},
    {'name': 'scopus_id', 'type':'string', 'description': 'Scopus ID (if known/present)'},
    {'name': 'openalex_id', 'type':'string', 'description': 'Openalex ID (if known/present)'},
    {'name': 's2_id', 'type':'string', 'description': 'SemanticScholar ID (if known/present)'},
    {'name': 'pubmed_id', 'type':'string', 'description': 'PubMed ID (if known/present)'},
    {'name': 'dimensions_id}', 'type':'string', 'description': 'Dimensions ID (if known/present)'},
    {'name': 'publication_year', 'type':'string', 'description': 'Publication year'},
    {'name': 'source', 'type':'string', 'description': 'Journal (or other publication venue)'}
]
SHAREABLE_META = [f["name"] for f in SHAREABLE_FIELDS]
FULL_FIELDS = SHAREABLE_FIELDS + [
    {'name': 'text', 'type': 'string', 'description':'Abstract'},
    {'name': 'user_id', 'type': 'string', 'description': 'ID of user making annotation'}
]
FULL_META = [f["name"] for f in FULL_FIELDS]
# Rows with no annotator (NULL user_id) are relabelled to this (not a real coder).
UNANNOTATED = '(unannotated)'

DATAPACKAGE = Path('datapackage.json')


def collect_label_options(labels: list[AnnotationSchemeLabel]) -> list[LabelOptions]:
    """Walk the scheme (incl. nested child labels) into a complete LabelOptions list.

    One LabelOptions per label key, declaring every choice value so the export emits a
    column for each concept regardless of whether it was ever annotated.
    """
    out: list[LabelOptions] = []
    for lab in labels:
        if lab.choices:
            values = [c.value for c in lab.choices]
            if lab.kind == 'multi':
                out.append(LabelOptions(key=lab.key, options_multi=values))
            elif lab.kind == 'single':
                out.append(LabelOptions(key=lab.key, options_int=values))
            elif lab.kind == 'bool':
                out.append(LabelOptions(key=lab.key, options_bool=[True, False]))
            for c in lab.choices:
                if c.children:
                    out.extend(collect_label_options(c.children))
        elif lab.kind == 'bool':
            out.append(LabelOptions(key=lab.key, options_bool=[True, False]))
    return out


def expected_label_columns(label_options: list[LabelOptions]) -> list[str]:
    """Get the full, deterministic set of `<key>|<value>` columns prepare_export_table emits."""
    cols: list[str] = []
    for lo in label_options:
        for v in lo.options_int or []:
            cols.append(f'{lo.key}|{v}')
        for v in lo.options_multi or []:
            cols.append(f'{lo.key}|{v}')
        if lo.options_bool:
            cols += [f'{lo.key}|0', f'{lo.key}|1']
    return cols


def pseudonymise(usernames: list[str]) -> dict[str, str]:
    """Map real usernames to stable `coder_NN` pseudonyms.

    The mapping is persisted (and reused) so a coder keeps the same pseudonym across
    every export version. New coders get the next free number. Sentinels like
    ``(unannotated)`` are passed through unchanged.
    """
    mapping: dict[str, str] = {}
    if PSEUDONYM_MAP.exists():
        mapping = json.loads(PSEUDONYM_MAP.read_text(encoding='utf-8'))

    real = sorted(u for u in set(usernames) if u and u != UNANNOTATED and u != 'RESOLVED')
    n = len(mapping)
    changed = False
    for u in real:
        if u not in mapping:
            n += 1
            mapping[u] = f'coder_{n:03d}'
            changed = True

    if changed:
        PSEUDONYM_MAP.parent.mkdir(parents=True, exist_ok=True)
        PSEUDONYM_MAP.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding='utf-8')

    # Sentinels map to themselves.
    return {**mapping, UNANNOTATED: UNANNOTATED, 'RESOLVED': 'RESOLVED'}

def write_datapackage(manifests: list[dict | None]) -> None:
    valid = [m for m in manifests if m is not None]
    if not valid:
        return
    resource_descriptors = []
    for m in valid:
        for (root, fields) in zip([SHAREABLE_ROOT, SENSITIVE_ROOT], [SHAREABLE_FIELDS, FULL_FIELDS], strict=True):

            path = str(root / f'{m["task"]}.csv')
            name = f'{m["task"]}-annotations'
            if root==SENSITIVE_ROOT:
                name+="-private"
            r = FResource(name=name, path=path)
            r.infer()
            d = r.to_descriptor()
            manual_fields = fields + m["label_field_metadata"]
            field_lookup = {f['name']: f for f in manual_fields}
            for field in d["schema"]["fields"]:
                if field["name"] in field_lookup:
                    field.update(field_lookup[field['name']])

            if m['task']=='taxonomy':
                mapping: list[dict] = json.loads(MAPPING_JSON.read_text())
                taxonomy_lookup = {
                    concept['col_pipe']: concept
                    for concept in mapping
                }
                for field in d["schema"]["fields"]:
                    rec = taxonomy_lookup.get(field['name'])
                    if rec:
                        field['title'] = rec['pref_label']
                        if rec['definition']:
                            field['description'] = rec['definition']
                        field['concept_id'] = rec['concept_id']
                        field['concept_uri'] = rec['concept_uri']
                        field['scheme_name'] = rec['scheme_name']
            d.update({
                'scope_ids': m['scope_ids'],
                'exported': m['created'],
                'n_items': m['n_items'],
                'n_coders': m['n_coders'],
                'n_label_columns': m['n_label_columns'],
            })
            if root==SENSITIVE_ROOT:
                d.update({"access": "restricted"})
            resource_descriptors.append(d)
    pkg = FPackage(name='impact-case-1-annotations').to_descriptor()
    pkg['resources'] = resource_descriptors
    DATAPACKAGE.write_text(json.dumps(pkg, indent=2), encoding='utf-8')
    print(f'[green]  wrote[/green] {DATAPACKAGE}')

def label_field_metadata(labels: list[AnnotationSchemeLabel]) -> list[dict]:
    meta = []
    def walk(labs):
        for lab in labs:
            if lab.choices:
                for choice in lab.choices:
                    meta.append({
                        'name': f'{lab.key}|{choice.value}',
                        'title': f'{lab.name} - {choice.name}',
                        'type': 'boolean',
                        "trueValues": ["1", "1.0", "true", "True"],
                        "falseValues": ["0", "0.0", "false", "False"],
                        'description': choice.hint or lab.hint or '',
                    })
                    if choice.children:
                        walk(choice.children)
            elif lab.kind == 'bool':
                for v, label_str in [(0, 'No'), (1, 'Yes')]:
                    meta.append({
                        'name': f'{lab.key}|{v}',
                        'title': f'{lab.name} - {label_str}',
                        'type': 'boolean',
                        "trueValues": ["1", "1.0", "true", "True"],
                        "falseValues": ["0", "0.0", "false", "False"],
                        'description': lab.hint or '',
                    })
    walk(labels)
    return meta

async def export_scheme(task: str, task_config: TaskConfig, show_scopes: bool) -> dict | None:
    """Export one scheme into a versioned sensitive + shareable pair; return the manifest."""
    db_engine = get_engine_async(conf_file=CONF_FILE)

    async with db_engine.session() as session:
        scheme = (
            await session.execute(sa.select(AnnotationScheme).where(AnnotationScheme.annotation_scheme_id == task_config['scheme_id']))
        ).scalar_one_or_none()
        if scheme is None:
            print(f'[red]No annotation scheme with id={task_config["scheme_id"]!r} ({task}); skipping.[/red]')
            return None

        project_id = str(scheme.project_id)
        labels = [AnnotationSchemeLabel.model_validate(d) for d in (scheme.labels or [])]
        scope_ids = [
            str(s)
            for s in (
                await session.execute(sa.select(AssignmentScope.assignment_scope_id).where(AssignmentScope.annotation_scheme_id == task_config['scheme_id']))
            )
            .scalars()
            .all()
        ]
        if show_scopes:
            print(f'Scope IDs for {task}')
            print(scope_ids)
            return
        else:
            scope_ids = task_config['scope_ids']
        assigned_item_ids = [
            str(i)
            for i in (await session.execute(sa.select(Assignment.item_id).distinct().where(Assignment.assignment_scope_id.in_(scope_ids)))).scalars().all()
        ]
        bot_meta_ids = [
            str(b)
            for b in (
                await session.execute(
                    sa.select(BotAnnotationMetaData.bot_annotation_metadata_id).where(BotAnnotationMetaData.assignment_scope_id.in_(scope_ids))
                )
            ).scalars()
        ]

    label_cols = expected_label_columns(collect_label_options(labels))
    print(f'[bold]{task}[/bold] ({scheme.name}): {len(scope_ids)} scopes, {len(assigned_item_ids)} assigned docs, {len(label_cols)} concept columns')

    if not assigned_item_ids:
        print(f'[yellow]No assigned documents for {task}; nothing to export.[/yellow]')
        return None

    async with db_engine.session() as session:
        rows = await prepare_export_table(
            session=session,
            nql_filter=FieldFilters(field='item_id', values=assigned_item_ids),
            bot_annotation_metadata_ids=bot_meta_ids,
            assignment_scope_ids=scope_ids,
            user_ids=None,
            project_id=project_id,
            labels=collect_label_options(labels),
            ignore_hierarchy=True,
            ignore_repeat=True,
        )

    df = pd.DataFrame(rows)
    for c in label_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.sort_values(['item_id','username']).reset_index(drop=True)

    sensitive_path = SENSITIVE_ROOT / f'{task}.csv'
    shareable_path = SHAREABLE_ROOT / f'{task}.csv'
    SENSITIVE_ROOT.mkdir(parents=True, exist_ok=True)
    SHAREABLE_ROOT.mkdir(parents=True, exist_ok=True)

    # --- sensitive tier: everything, real usernames + text/title ---
    full_cols = [c for c in FULL_META if c not in label_cols] + label_cols
    df.reindex(columns=full_cols).to_csv(sensitive_path, index=False)

    # --- shareable tier: obfuscated coders, no text/title ---
    pseudo = pseudonymise(df['username'].dropna().tolist() if 'username' in df.columns else [])
    shareable = df.copy()
    shareable['coder'] = shareable['username'].map(lambda u: pseudo.get(u, u)) if 'username' in shareable else UNANNOTATED
    share_cols = [c for c in SHAREABLE_META if c in shareable.columns] + ['coder'] + label_cols
    shareable.reindex(columns=share_cols).to_csv(shareable_path, index=False)

    # --- provenance manifest (committed; non-sensitive) ---
    n_coders = sum(1 for u in (pseudo) if u not in {UNANNOTATED, 'RESOLVED'})
    manifest = {
        'task': task,
        'scheme_id': task_config['scheme_id'],
        'scheme_name': scheme.name,
        'project_id': project_id,
        'created': datetime.now(timezone.utc).isoformat(),
        'scope_ids': scope_ids,
        'n_rows': int(df.shape[0]),
        'n_items': int(df['item_id'].nunique()) if 'item_id' in df.columns else 0,
        'n_coders': n_coders,
        'n_label_columns': len(label_cols),
        'label_field_metadata': label_field_metadata(labels)
    }

    print(
        f'[green]  wrote[/green] {sensitive_path} + {shareable_path}  '
        f'({manifest["n_rows"]} rows, {manifest["n_items"]} items, {n_coders} coders)'
    )
    return manifest


async def _run(tasks: dict[str, TaskConfig], show_scopes) -> None:
    results = await asyncio.gather(*[export_scheme(t, conf, show_scopes=show_scopes) for t, conf in tasks.items()])
    if not show_scopes:
        write_datapackage(list(results))
        report = f_validate(str(DATAPACKAGE))
        if report.valid:
            print('[green]datapackage valid[/green]')
        else:
            for err in report.flatten(['message']):
                print(f'[yellow]validation warning:[/yellow] {err[0]}')


def main(
    task: Annotated[str, typer.Option(help="'inout', 'taxonomy', or 'all'")] = 'all',
    show_scopes: Annotated[bool, typer.Option(help='show scopes and exit')] = False,
) -> None:

    if task == 'all':
        selected = TASKS
    elif task in TASKS:
        selected = {task: TASKS[task]}
    else:
        raise typer.BadParameter(f'task must be one of {list(TASKS) + ["all"]}')
    asyncio.run(_run(selected, show_scopes=show_scopes))


if __name__ == '__main__':
    typer.run(main)
