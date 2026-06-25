"""Export annotations for a NACSOS annotation scheme to CSV.

Companion to ``import_taxonomy.py``: use this to inspect the *exact* column naming
NACSOS produces for an export, so we can confirm the concept<->scheme mapping joins
on the right keys.

NACSOS names each annotation column by the label ``key`` and the choice ``value``.
There are two export code paths in nacsos_data:

  * ``wide_export_table`` derives its columns from the annotations that actually
    exist, so columns for *unused* choice values are silently missing. That makes the
    column set data-dependent and unstable -- bad for any downstream consumer that
    expects a fixed schema (one column per concept).

  * ``prepare_export_table`` (used here) takes an explicit ``labels`` list. Its
    ``_get_label_selects`` emits one column *per declared value*, used or not, so we
    get the complete, stable column set: exactly one ``<label_key>|<value>`` column
    per concept. Rows are long-format: one row per (item, annotator), with a
    ``username`` column ('RESOLVED' for resolved labels).

We build that complete ``labels`` list by reading the scheme back from the database
(the authoritative definition) and walking every label / nested child label.

The scheme id (SCHEME_ID) is imported from import_taxonomy.py so there is a single
source of truth; run the import there first.
"""

import asyncio
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from rich import print

from nacsos_data.db.connection import get_engine_async
from nacsos_data.db.schemas.annotations import AnnotationScheme, Assignment, AssignmentScope
from nacsos_data.models.annotations import AnnotationSchemeLabel
from nacsos_data.models.nql import FieldFilters
from nacsos_data.util.annotations.export import LabelOptions, prepare_export_table

# Single source of truth: the scheme id is defined by the import script.
from import_taxonomy import ANNOTATION_SCHEME_ID as SCHEME_ID

# secret.env lives at the repo root (one level above taxonomy/)
CONF_FILE = str(Path(__file__).resolve().parents[1] / "secret.env")
OUT_CSV = Path(__file__).resolve().parent / "annotation_export.csv"

# Lightweight base columns to keep. The export otherwise carries the full document
# `text` (and keywords/authors) for *every* row, which bloats the file to hundreds of
# MB. None of that is needed to link an annotation back to an item or the taxonomy.
KEEP_BASE_COLS = [
    "item_id", "doi", "wos_id", "scopus_id", "openalex_id", "s2_id", "pubmed_id",
    "dimensions_id", "publication_year", "source", "title", "user_id", "username",
]


def collect_label_options(labels: list[AnnotationSchemeLabel]) -> list[LabelOptions]:
    """Walk the scheme (incl. nested child labels) into a complete LabelOptions list.

    One LabelOptions per label key, declaring every choice value so the export emits a
    column for each concept regardless of whether it was ever annotated.
    """
    out: list[LabelOptions] = []
    for lab in labels:
        if lab.choices:
            values = [c.value for c in lab.choices]
            if lab.kind == "multi":
                out.append(LabelOptions(key=lab.key, options_multi=values))
            elif lab.kind == "single":
                out.append(LabelOptions(key=lab.key, options_int=values))
            elif lab.kind == "bool":
                out.append(LabelOptions(key=lab.key, options_bool=[True, False]))
            # recurse into child labels nested under any choice
            for c in lab.choices:
                if c.children:
                    out.extend(collect_label_options(c.children))
        elif lab.kind == "bool":
            out.append(LabelOptions(key=lab.key, options_bool=[True, False]))
    return out


def expected_label_columns(label_options: list[LabelOptions]) -> list[str]:
    """The full, deterministic set of `<key>|<value>` columns prepare_export_table emits."""
    cols: list[str] = []
    for lo in label_options:
        for v in (lo.options_int or []):
            cols.append(f"{lo.key}|{v}")
        for v in (lo.options_multi or []):
            cols.append(f"{lo.key}|{v}")
        if lo.options_bool:
            cols += [f"{lo.key}|0", f"{lo.key}|1"]
    return cols


async def main() -> None:
    db_engine = get_engine_async(conf_file=CONF_FILE)

    # 1. Resolve scheme -> project + labels, and gather its assignment scopes.
    async with db_engine.session() as session:
        scheme = (
            await session.execute(
                sa.select(AnnotationScheme).where(AnnotationScheme.annotation_scheme_id == SCHEME_ID)
            )
        ).scalar_one_or_none()
        if scheme is None:
            raise SystemExit(f"No annotation scheme with id={SCHEME_ID!r} found in the database.")

        project_id = str(scheme.project_id)
        labels = [AnnotationSchemeLabel.model_validate(label_def) for label_def in (scheme.labels or [])]

        scope_ids = [
            str(s)
            for s in (
                await session.execute(
                    sa.select(AssignmentScope.assignment_scope_id).where(
                        AssignmentScope.annotation_scheme_id == SCHEME_ID
                    )
                )
            ).scalars().all()
        ]

        # Items actually assigned in this scheme's scopes. Without a filter, the export
        # spans the *whole project corpus* (one row per item, annotations left-joined on).
        # We push these ids into the items query (item_id IN ...) so the DB only touches
        # the assigned documents -- incl. assigned-but-unannotated -- instead of scanning
        # all ~35k project items.
        assigned_item_ids = [
            str(i)
            for i in (
                await session.execute(
                    sa.select(Assignment.item_id)
                    .distinct()
                    .where(Assignment.assignment_scope_id.in_(scope_ids))
                )
            ).scalars().all()
        ]

    label_options = collect_label_options(labels)
    label_cols = expected_label_columns(label_options)

    print(f"[bold]Scheme:[/bold] {scheme.name}  (project={project_id})")
    print(f"[bold]Labels:[/bold] {len(label_options)}  [bold]Concept columns:[/bold] {len(label_cols)}")
    print(f"[bold]Assignment scopes:[/bold] {len(scope_ids)} -> {scope_ids}")
    print(f"[bold]Assigned documents:[/bold] {len(assigned_item_ids)}")

    if not scope_ids:
        print(
            "[yellow]No assignment scopes exist for this scheme yet. The export would have the "
            "full set of (empty) concept columns but no annotated rows. Create a scope + "
            "assignments first if you want populated rows.[/yellow]"
        )

    # 2. Build the export table with the *explicit, complete* label set, restricting the
    #    items query to assigned documents (item_id IN ...) so we never scan the corpus.
    #    ignore_hierarchy=True -> flat columns; ignore_repeat=True -> `<key>|<value>` (no repeat suffix).
    item_filter = FieldFilters(field="item_id", values=assigned_item_ids)
    async with db_engine.session() as session:
        rows = await prepare_export_table(
            session=session,
            nql_filter=item_filter,
            bot_annotation_metadata_ids=None,
            assignment_scope_ids=scope_ids,
            user_ids=None,
            project_id=project_id,
            labels=label_options,
            ignore_hierarchy=True,
            ignore_repeat=True,
        )

    df = pd.DataFrame(rows)

    # 3. Relabel no-annotator rows. prepare_export_table sets username via
    #    coalesce(User.username, 'RESOLVED'), so assigned-but-unannotated documents
    #    (NULL user_id) read as 'RESOLVED' despite not being resolved. We pass no bot
    #    scopes here, so NULL user_id unambiguously means "no annotation yet".
    if "username" in df.columns and "user_id" in df.columns:
        df.loc[df["user_id"].isna(), "username"] = "(unannotated)"

    # 4. Keep only lightweight base columns + the full, deterministic concept column set.
    #    Every concept column is present even if prepare_export_table omitted one
    #    (e.g. zero matching rows), filled with NA -> stable schema for downstream.
    base_cols = [c for c in KEEP_BASE_COLS if c in df.columns]
    for c in label_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.reindex(columns=base_cols + label_cols)

    print(f"[bold]Rows:[/bold] {df.shape[0]}  [bold]Total columns:[/bold] {df.shape[1]}")
    print(f"[bold]Base columns:[/bold] {base_cols}")
    print(f"[bold]Concept (label) columns:[/bold] {label_cols[:10]}{' ...' if len(label_cols) > 10 else ''}")

    df.to_csv(OUT_CSV, index=False)
    print(f"[green]Wrote export ({df.shape[0]} rows x {df.shape[1]} cols) to {OUT_CSV}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
