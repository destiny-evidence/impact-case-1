"""Import a taxonomy into NACSOS."""

import asyncio
from rdflib import Graph
from rdflib.namespace import SKOS, RDF, DCTERMS
from uuid import uuid4

from nacsos_data.models.annotations import (
    AnnotationSchemeLabelChoice,
    AnnotationSchemeLabel,
    AnnotationSchemeModel,
)
from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.crud.annotations import upsert_annotation_scheme
from rich import print
import re


class KeyRegistry:
    def __init__(self):
        self.used_keys: set[str] = set()

    def concept_uri_to_safe_key(self, uri: str) -> str:
        """
        Converts a URI like .../DSNY000162 into a strict text key: k_dsny000162
        This guarantees NACSOS never interprets the key as a number.
        """
        code = uri.split('/')[-1].split('#')[-1].lower()
        # Prepend 'k_' to guarantee it starts with letters and isn't purely numeric
        clean = f"k_{code}"
        clean = re.sub(r'[^a-z0-9\-_]', '_', clean)
        clean = re.sub(r'[-_]{2,}', '_', clean)

        base_clean = clean
        counter = 1
        while clean in self.used_keys:
            clean = f"{base_clean}_{counter}"
            counter += 1

        self.used_keys.add(clean)
        return clean

def parse_vocabulary_to_nacsos(ttl_filepath: str, project_id) -> AnnotationSchemeModel:
    g = Graph()
    g.parse(ttl_filepath, format="turtle")

    registry = KeyRegistry()

    schemes = []
    for s, _p, _o in g.triples((None, RDF.type, SKOS.ConceptScheme)):
        title = g.value(s, DCTERMS.title) or g.value(s, SKOS.prefLabel)
        desc = g.value(s, DCTERMS.description) or g.value(s, SKOS.definition)
        schemes.append({
            "uri": str(s),
            "name": str(title) if title else str(s).split('/')[-1],
            "description": str(desc) if desc else None
        })

    # 3. Extract All Concepts and assign a unique tracking integer value
    concepts = {}
    choice_value_counter = 0

    for s, _p, _o in g.triples((None, RDF.type, SKOS.Concept)):
        uri = str(s)
        pref_label = g.value(s, SKOS.prefLabel)
        definition = g.value(s, SKOS.definition)
        in_scheme = g.value(s, SKOS.inScheme)
        broader = g.value(s, SKOS.broader)


        concepts[uri] = {
            "uri": uri,
            "name": str(pref_label) if pref_label else uri.split('/')[-1],
            "hint": str(definition) if definition else None,
            "value": choice_value_counter,
            "in_scheme": str(in_scheme) if in_scheme else None,
            "broader": str(broader) if broader else None,
            "children": []
        }
        choice_value_counter += 1

    # 4. Reconstruct parent-child relations for tree traversal
    top_level_concepts_by_scheme = {s["uri"]: [] for s in schemes}

    for _uri, concept in concepts.items():
        parent_uri = concept["broader"]
        scheme_uri = concept["in_scheme"]

        if parent_uri and parent_uri in concepts:
            concepts[parent_uri]["children"].append(concept)
        elif scheme_uri in top_level_concepts_by_scheme:
            top_level_concepts_by_scheme[scheme_uri].append(concept)

    # 5. Helper function to recursively compile choices & children
    def build_choices(concept_list: list) -> list[AnnotationSchemeLabelChoice]:
        choices_output = []
        for local_idx, c in enumerate(concept_list):
            choice_node = AnnotationSchemeLabelChoice(
                name=c["name"],
                hint=c["hint"],
                value=local_idx
            )
            # If this concept has sub-concepts, nest them into a nested label hierarchy
            if c["children"]:
                # The sub-concepts need to sit inside an explicit child label dimension
                sub_label_key = f"{c['uri'].split('/')[-1]}_sub"
                child_label = AnnotationSchemeLabel(
                    name=f"{c['name']} Category",
                    key=sub_label_key,
                    kind="multi",  # Can change to 'single' depending on choice strictness
                    required=False,
                    dropdown=True if len(c["children"]) > 10 else False,
                    choices=build_choices(c["children"])
                )
                choice_node.children = [child_label]

            choices_output.append(choice_node)
        return choices_output

    # 6. Build the master list of top level labels per Concept Scheme
    nacsos_labels = []
    for scheme in schemes:
        top_concepts = top_level_concepts_by_scheme[scheme["uri"]]
        if not top_concepts:
            continue

        scheme_key = registry.concept_uri_to_safe_key(scheme["uri"])

        # Build the wrapper label for this dimension
        scheme_label = AnnotationSchemeLabel(
            name=scheme["name"],
            key=scheme_key,
            hint=scheme["description"],
            kind="multi",  # Multi-select allowed for top level domains
            required=False,
            dropdown=True,  # Default to true for cleaner layout navigation
            choices=build_choices(top_concepts)
        )
        nacsos_labels.append(scheme_label)

    # 7. Package into final Project Annotation Scheme Model
    master_scheme = AnnotationSchemeModel(
        annotation_scheme_id=str(uuid4()),
        project_id=project_id,
        name="DESTINY Taxonomy Scheme",
        description="Auto-generated annotation schema parsed from the DESTINY SKOS vocabulary Turtle file.",
        labels=nacsos_labels
    )

    return master_scheme

async def main():
    TARGET_PROJECT_ID = "db6ee519-afb5-4813-822b-bfbc7dfd2237"
    vocab_file = "vocabulary.ttl"
    scheme = parse_vocabulary_to_nacsos(vocab_file, TARGET_PROJECT_ID)

    print(scheme)

    db_engine = get_engine_async(conf_file="secret.env")

    scheme_id = await upsert_annotation_scheme(
        annotation_scheme=scheme,
        db_engine=db_engine
    )

if __name__ == "__main__":
    asyncio.run(main())

