"""Prune a SKOS taxonomy by removing concept schemes.

Removes every ``skos:ConceptScheme`` whose ``dct:title`` contains the marker
string (default: "automation planned"), together with the ``skos:Concept``
resources that belong to it via ``skos:inScheme``.
"""

from pathlib import Path

import typer
from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS

app = typer.Typer(add_completion=False)

REMOVALS = set({
    "Population - Vulnerability",
    "Health - Metrics",
    "Governance scale",
    "Actors",
    "Geographic features and biomes",
    "Interventions / responses / solutions",
    "Climate zones",
    "Health - Exposure",
    "Human settlements",
    "Population - Age",
    "Population - Gender/sex",
})
print(REMOVALS)

def _depth(graph: Graph, concept: "URIRef", cache: dict) -> int:
    """Depth of a concept: 1 for top concepts (no broader), else 1 + shallowest parent."""
    if concept in cache:
        return cache[concept]
    cache[concept] = 1  # provisional value breaks any broader-cycles
    parents = list(graph.objects(concept, SKOS.broader))
    cache[concept] = 1 if not parents else 1 + min(_depth(graph, p, cache) for p in parents)
    return cache[concept]


@app.command()
def prune(
    taxonomy: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the taxonomy (Turtle) file to prune.",
    ),
    marker: str = typer.Option(
        "automation planned",
        help="Case-insensitive substring in dct:title that flags a scheme for removal.",
    ),
    max_depth: int | None = typer.Option(
        None,
        help="Remove concepts deeper than this level"
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the pruned taxonomy. Defaults to the input path with a '_pruned' suffix.",
    ),
) -> None:
    """Remove concept schemes whose title contains MARKER, plus their concepts."""
    graph = Graph()
    graph.parse(taxonomy, format="turtle")

    marker_lower = marker.lower()
    schemes_to_remove = {
        scheme
        for scheme in graph.subjects(RDF.type, SKOS.ConceptScheme)
        if any(
            marker_lower in str(title).lower()
            or str(title) in REMOVALS
            for title in graph.objects(scheme, DCTERMS.title)
        )
    }

    deep_concepts = set()
    if max_depth is not None:
        cache: dict = {}
        deep_concepts = {
            c
            for c in graph.subjects(RDF.type, SKOS.Concept)
            if _depth(graph, c, cache) > max_depth
        }

    if not schemes_to_remove and not deep_concepts:
        typer.echo("Nothing to remove; check --marker / --max-depth.")
        raise typer.Exit()

    subjects_to_remove = set(schemes_to_remove)
    for scheme in schemes_to_remove:
        subjects_to_remove.update(graph.subjects(SKOS.inScheme, scheme))
    subjects_to_remove.update(deep_concepts)


    for subject in subjects_to_remove:
        graph.remove((subject, None, None))
        graph.remove((None, None, subject))

    destination = output or taxonomy.with_name(
        f"{taxonomy.stem}_pruned{taxonomy.suffix}"
    )
    graph.serialize(destination=destination, format="turtle")

    typer.echo(
        f"Removed {len(schemes_to_remove)} scheme(s) and "
        f"{len(subjects_to_remove) - len(schemes_to_remove)} concept(s)."
    )
    for scheme in sorted(schemes_to_remove, key=str):
        typer.echo(f"  - {scheme}")
    typer.echo(f"Wrote pruned taxonomy to {destination}")


if __name__ == "__main__":
    app()
