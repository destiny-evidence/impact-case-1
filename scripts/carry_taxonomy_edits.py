"""Copy SKOS definitions and scope notes from one taxonomy file onto another.

For every concept URI present in both files, the target's skos:definition and
skos:scopeNote are replaced with the source's. Writes the target with a suffix
(default '_edited'). Concepts only in one file are left untouched.
"""

from pathlib import Path

import typer
from rdflib import Graph
from rdflib.namespace import RDF, SKOS

FIELDS = (SKOS.definition, SKOS.scopeNote)

def main(
    target: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Taxonomy (Turtle) file to write definitions/scope notes onto.",
    ),
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        exists=True,
        dir_okay=False,
        help="Taxonomy file to take definitions/scope notes from.",
    ),
    suffix: str = typer.Option("_edited", help="Suffix for the output filename."),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit output path. Defaults to the target name plus the suffix.",
    ),
) -> None:
    """Overlay SOURCE's definitions and scope notes onto TARGET's concepts."""
    src_g, tgt_g = Graph(), Graph()
    src_g.parse(source, format="turtle")
    tgt_g.parse(target, format="turtle")

    target_concepts = set(tgt_g.subjects(RDF.type, SKOS.Concept))
    updated = 0
    for concept in src_g.subjects(RDF.type, SKOS.Concept):
        if concept not in target_concepts:
            continue
        for field in FIELDS:
            tgt_g.remove((concept, field, None))
            for value in src_g.objects(concept, field):
                tgt_g.add((concept, field, value))
        updated += 1

    destination = output or target.with_name(f"{target.stem}{suffix}{target.suffix}")
    tgt_g.serialize(destination=destination, format="turtle")

    typer.echo(f"Overlaid definitions/scope notes onto {updated} shared concept(s).")
    typer.echo(f"Wrote {destination}")


if __name__ == "__main__":
    typer.run(main)
