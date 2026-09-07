"""Prune a SKOS taxonomy by removing concept schemes.

Removes every ``skos:ConceptScheme`` whose ``dct:title`` contains the marker
string (default: "automation planned"), together with the ``skos:Concept``
resources that belong to it via ``skos:inScheme``.
"""

from pathlib import Path

import typer
from rdflib import Graph
from rdflib.namespace import DCTERMS, RDF, SKOS

app = typer.Typer(add_completion=False)


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
            for title in graph.objects(scheme, DCTERMS.title)
        )
    }

    if not schemes_to_remove:
        typer.echo(f"No concept schemes matched marker {marker!r}; nothing to do.")
        raise typer.Exit()

    subjects_to_remove = set(schemes_to_remove)
    for scheme in schemes_to_remove:
        subjects_to_remove.update(graph.subjects(SKOS.inScheme, scheme))

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
