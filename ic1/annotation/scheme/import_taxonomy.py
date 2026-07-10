"""Import the DESTINY taxonomy into NACSOS and emit a concept<->scheme mapping.

This builds a NACSOS Annotation Scheme from the SKOS taxonomy *and* writes a mapping
file that lets you recover the taxonomy concept behind every annotation column in a
NACSOS export.

Why the mapping is needed
-------------------------
NACSOS export columns are named ``<label_key>|<value>`` (or ``<label_key>:<value>`` in
the other export path). ``label_key`` is the annotation label and ``value`` is the
*local index* of the chosen option within that label. The NACSOS UI requires those
values to be ``0..n`` contiguous per label, so the index carries no concept identity on
its own -- you must know which (label_key, value) pair maps to which concept. That pair
is run-specific, so the mapping MUST be regenerated together with the scheme (this
script does both from a single parse).

Output
------
* destiny_taxonomy_nacsos_mapping.csv / .json -- one row per concept, with the
  (label_key, value) pair plus convenience strings matching either export separator.
"""

import asyncio
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import SKOS, RDF, DCTERMS
from rich import print

from nacsos_data.models.annotations import (
    AnnotationSchemeLabelChoice,
    AnnotationSchemeLabel,
    AnnotationSchemeModel,
)
from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.crud.annotations import upsert_annotation_scheme

from ic1.core.ids import PROJECT_ID, TAXONOMY_SCHEME_ID
from ic1.core.config import CONF_FILE, VOCAB_FILE, MAPPING_CSV, MAPPING_JSON

import string

# --- configuration -----------------------------------------------------------------

# Set False to build + validate + write the mapping without touching the database.
WRITE_TO_DB = True

MAPPING_FIELDS = [
    'annotation_scheme_id',
    'project_id',
    'scheme_uri',
    'scheme_name',
    'label_key',
    'value',
    'concept_uri',
    'concept_id',
    'pref_label',
    'definition',
    'broader_uri',
    'depth',
    'sub_label_key',
    'col_pipe',  # matches prepare_export_table columns:  <label_key>|<value>
    'col_colon',  # matches wide_export_table columns:     ...|<label_key>:<value>
]


class KeyRegistry:
    """Generates NACSOS-safe, globally-unique label keys."""

    def __init__(self) -> None:
        self.used_keys: set[str] = set()

    def make_key(self, raw: str) -> str:
        clean = re.sub(r'[^a-z0-9\-_]', '_', raw.lower())
        clean = re.sub(r'[-_]{2,}', '_', clean)
        base_clean = clean
        counter = 1
        while clean in self.used_keys:
            counter_letter = string.ascii_lowercase[counter - 1]
            clean = f'{base_clean}_{counter_letter}'
            counter += 1
        self.used_keys.add(clean)
        return clean

    def concept_uri_to_safe_key(self, uri: str) -> str:
        """e.g. .../actors -> k_actors  (prefixed so it is never purely numeric)."""
        code = uri.split('/')[-1].split('#')[-1]
        return self.make_key(f'k_{code}')


def parse_vocabulary_to_nacsos(ttl_filepath: str, project_id: str, scheme_id: str):
    g = Graph()
    g.parse(ttl_filepath, format='turtle')

    registry = KeyRegistry()

    # 1. Concept schemes (top-level annotation dimensions)
    schemes = []
    for s, _p, _o in g.triples((None, RDF.type, SKOS.ConceptScheme)):
        title = g.value(s, DCTERMS.title) or g.value(s, SKOS.prefLabel)
        desc = g.value(s, DCTERMS.description) or g.value(s, SKOS.definition)
        schemes.append(
            {
                'uri': str(s),
                'name': str(title) if title else str(s).split('/')[-1],
                'description': str(desc) if desc else None,
            }
        )

    # 2. Concepts
    concepts = {}
    for s, _p, _o in g.triples((None, RDF.type, SKOS.Concept)):
        uri = str(s)
        pref_label = g.value(s, SKOS.prefLabel)
        definition = g.value(s, SKOS.definition)
        in_scheme = g.value(s, SKOS.inScheme)
        broader = g.value(s, SKOS.broader)
        concepts[uri] = {
            'uri': uri,
            'code': uri.split('/')[-1],
            'name': str(pref_label) if pref_label else uri.split('/')[-1],
            'hint': str(definition) if definition else None,
            'in_scheme': str(in_scheme) if in_scheme else None,
            'broader': str(broader) if broader else None,
            'children': [],
        }

    # 3. Reconstruct the concept tree
    top_level_concepts_by_scheme = {s['uri']: [] for s in schemes}
    for _uri, concept in concepts.items():
        parent_uri = concept['broader']
        scheme_uri = concept['in_scheme']
        if parent_uri and parent_uri in concepts:
            concepts[parent_uri]['children'].append(concept)
        elif scheme_uri in top_level_concepts_by_scheme:
            top_level_concepts_by_scheme[scheme_uri].append(concept)

    # 4. Build choices, capturing the (label_key, value) -> concept mapping as we go.
    mapping: list[dict] = []

    def build_choices(concept_list: list, label_key: str, depth: int, scheme_uri: str, scheme_name: str) -> list[AnnotationSchemeLabelChoice]:
        choices_output = []
        # Order concepts alphabetically by prefLabel at every level. local_idx (the choice
        # `value`) follows this order -> 0..n contiguous (UI requirement) and deterministic.
        for local_idx, c in enumerate(sorted(concept_list, key=lambda x: x['name'].casefold())):
            choice_node = AnnotationSchemeLabelChoice(name=c['name'], hint=c['hint'], value=local_idx)

            sub_label_key = None
            if c['children']:
                sub_label_key = registry.make_key(f'{label_key}_sub')
                child_label = AnnotationSchemeLabel(
                    name=f'{c["name"]} Category',
                    key=sub_label_key,
                    kind='multi',
                    required=False,
                    dropdown=len(c['children']) > 10,
                    choices=build_choices(c['children'], sub_label_key, depth + 1, scheme_uri, scheme_name),
                )
                choice_node.children = [child_label]

            mapping.append(
                {
                    'annotation_scheme_id': scheme_id,
                    'project_id': project_id,
                    'scheme_uri': scheme_uri,
                    'scheme_name': scheme_name,
                    'label_key': label_key,
                    'value': local_idx,
                    'concept_uri': c['uri'],
                    'concept_id': c['code'],
                    'pref_label': c['name'],
                    'definition': c['hint'],
                    'broader_uri': c['broader'],
                    'depth': depth,
                    'sub_label_key': sub_label_key,
                    'col_pipe': f'{label_key}|{local_idx}',
                    'col_colon': f'{label_key}:{local_idx}',
                }
            )
            choices_output.append(choice_node)
        return choices_output

    # 5. One top-level label per concept scheme (dimensions ordered alphabetically by title)
    nacsos_labels = []
    for scheme in sorted(schemes, key=lambda s: s['name'].casefold()):
        top_concepts = top_level_concepts_by_scheme[scheme['uri']]
        if not top_concepts:
            continue
        scheme_key = registry.concept_uri_to_safe_key(scheme['uri'])
        nacsos_labels.append(
            AnnotationSchemeLabel(
                name=scheme['name'],
                key=scheme_key,
                hint=scheme['description'],
                kind='multi',
                required=False,
                dropdown=True,
                choices=build_choices(top_concepts, scheme_key, 1, scheme['uri'], scheme['name']),
            )
        )

    # The SKOS file carries no vocabulary-level title/description/version (only the 21
    # per-dimension ConceptScheme titles, which feed the labels above). The version only
    # exists in the filename, so derive the scheme name from it and build the description
    # from the source file plus the parsed counts.
    fname = Path(ttl_filepath).name
    vmatch = re.search(r'v(\d+)[-.](\d+)', fname)
    version = f'v{vmatch.group(1)}.{vmatch.group(2)}' if vmatch else None
    scheme_name = f'DESTINY Taxonomy {version}' if version else 'DESTINY Taxonomy'
    scheme_description = f'Generated from the DESTINY SKOS vocabulary ({fname}): {len(nacsos_labels)} dimensions, {len(mapping)} concepts.'

    master_scheme = AnnotationSchemeModel(
        annotation_scheme_id=scheme_id,
        project_id=project_id,
        name=scheme_name,
        description=scheme_description,
        labels=nacsos_labels,
    )

    validate_mapping(mapping, concepts)
    return master_scheme, mapping


def validate_mapping(mapping: list[dict], concepts: dict) -> None:
    """Fail loudly unless the mapping is complete and the UI invariants hold."""
    # (a) Coverage: every concept maps exactly once.
    mapped_uris = [m['concept_uri'] for m in mapping]
    missing = set(concepts) - set(mapped_uris)
    if missing:
        raise AssertionError(f'{len(missing)} concept(s) not mapped, e.g. {sorted(missing)[:5]}')
    if len(mapped_uris) != len(set(mapped_uris)):
        raise AssertionError('A concept was mapped more than once.')
    if len(mapped_uris) != len(concepts):
        raise AssertionError(f'Mapping rows ({len(mapped_uris)}) != concepts ({len(concepts)}).')

    # (b) Per-label values must be 0..n-1 contiguous and unique (NACSOS UI requirement).
    by_label: dict[str, list[int]] = {}
    for m in mapping:
        by_label.setdefault(m['label_key'], []).append(m['value'])
    for key, values in by_label.items():
        if sorted(values) != list(range(len(values))):
            raise AssertionError(f'Label {key!r} values are not 0..n contiguous: {sorted(values)}')

    # (c) (label_key, value) pairs unique across the scheme.
    pairs = [(m['label_key'], m['value']) for m in mapping]
    if len(pairs) != len(set(pairs)):
        raise AssertionError('Duplicate (label_key, value) pair detected.')


def write_mapping(mapping: list[dict]) -> None:
    with MAPPING_CSV.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(mapping)
    with MAPPING_JSON.open('w', encoding='utf-8') as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)


async def main() -> None:
    scheme, mapping = parse_vocabulary_to_nacsos(VOCAB_FILE, PROJECT_ID, TAXONOMY_SCHEME_ID)

    n_labels = len(scheme.labels)
    n_sub = sum(1 for m in mapping if m['sub_label_key'])
    print(f'[bold]Scheme:[/bold] {scheme.name}  id={scheme.annotation_scheme_id}')
    print(f'[bold]Concept schemes (top-level labels):[/bold] {n_labels}')
    print(f'[bold]Concepts mapped:[/bold] {len(mapping)}  (of which {n_sub} have sub-labels)')

    write_mapping(mapping)
    print(f'[green]Wrote mapping ({len(mapping)} rows) to[/green] {MAPPING_CSV.name} / {MAPPING_JSON.name}')

    if not WRITE_TO_DB:
        print('[yellow]WRITE_TO_DB is False -- scheme not upserted.[/yellow]')
        return

    # upsert_orm writes time_created from the model; set it so re-imports (UPDATE) don't
    # null out the NOT NULL column.
    scheme.time_created = datetime.now(timezone.utc)

    db_engine = get_engine_async(conf_file=CONF_FILE)
    scheme_id = await upsert_annotation_scheme(annotation_scheme=scheme, db_engine=db_engine)
    print(f'[green]Upserted annotation scheme:[/green] {scheme_id}')


if __name__ == '__main__':
    asyncio.run(main())
