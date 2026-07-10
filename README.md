# DESTinY Impact Case 1 — Climate & Health Map

Code for IC1: building a map of the climate & health evidence base. Covers literature
search, NACSOS annotation, ML classification, and LLM-assisted review.

## Repo structure

```
ic1/
├── core/           # Shared config, IDs, and TaskConfig dataclass
├── query/
│   ├── revisions/  # Versioned search query definitions
│   ├── experiments/# Query development and count scripts
│   └── ingest/     # Import items into NACSOS (random sample, manual)
├── annotation/
│   ├── scheme/     # SKOS taxonomy → NACSOS scheme import + concept mapping
│   ├── assignments/# Scope/assignment creation scripts
│   └── export/     # Export annotations → two-tier CSVs + datapackage.json
├── deet/           # LLM annotation projects (create + sync)
└── evaluation_splits/ # EvaluationSplits model and per-task split JSON files

data/
├── exports/        # Shareable annotation CSVs (git-tracked, Frictionless-described)
└── private/        # Sensitive data — text, real usernames (DVC-tracked, gitignored)
```

## Setup

```bash
uv sync
dvc pull
```

Environment: create `.conf/secret.env` with database credentials (see `.conf/example.env`).

## Database connection

NACSOS runs on a remote host. Map port 5433 locally:

```bash
ssh -N -L 5433:localhost:5432 se164 -J ts01
```

Requires a PIK account with access to the NACSOS host.

## Pipeline

Scripts are run from the repo root.

### 1. Query, schemes, and assignments

- Run query: `ic1/query/ingest`.
- Create an in/out scheme (manually on NACSOS).
- Import the taxonomy scheme: `ic1/annotation/scheme/import_taxonomy.py`
- Create assignments (either manually or with `ic1/assignments/`)


### 2. Export annotations and resolve

```bash
python ic1/annotation/export/export_annotations.py --task inout
```

Writes `data/exports/inout.csv` (shareable) and `data/private/exports/inout.csv`
(sensitive, DVC-tracked). Also writes/updates `datapackage.json` describing all resources.

This calls `ic1.annotation.export.resolve_annotations.resolve_annotations()`, but this can also be called as a separate script.

Resolving filters the sensitive export to RESOLVED rows only → `data/private/exports/inout_resolved.csv`.

### 4. Create deet project

```bash
python ic1/deet/create_deet_project.py --task inout
```

Assigns 1000 docs to `deet` split (deterministic hash), remainder to `train`. Writes
`ic1/evaluation_splits/inout_splits.json` and creates the deet project under
`ic1/deet/projects/inout/`.

### 4. Sync deet splits

*(Run after deet annotation is complete)*

```bash
python ic1/deet/sync_deet_splits.py --task inout
```

Reads deet's output, promotes deet docs to `validation`/`test` in the splits JSON.

## Data

`datapackage.json` at the repo root describes all resources (shareable and restricted).
Shareable CSVs in `data/exports/` are long-format: one row per (item, coder), with
pseudonymous coder IDs. Private CSVs include item text and real usernames — access via
`dvc pull` requires SSH access to `ts01.pik-potsdam.de`.

Version the private data after each export:

```bash
dvc add data/private
git add data/private.dvc
git commit -m "update private export"
dvc push
```