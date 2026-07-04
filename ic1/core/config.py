"""config for IC1 scripts."""
from typing import TypedDict
from pathlib import Path

from ic1.core.ids import (
    INOUT_SCHEME_ID,
    TAXONOMY_SCHEME_ID,
    INOUT_SCOPE_IDS,
    TAXONOMY_SCOPE_IDS
)

class TaskConfig(TypedDict):
    scheme_id: str
    scope_ids: list[str]


TASKS: dict[str, TaskConfig] = {
    'inout': {'scheme_id': INOUT_SCHEME_ID, 'scope_ids': INOUT_SCOPE_IDS},
    'taxonomy': {'scheme_id': TAXONOMY_SCHEME_ID, 'scope_ids': TAXONOMY_SCOPE_IDS},
}

CONF_FILE = '.conf/secret.env'

DATAPACKAGE = Path('datapackage.json')

DATASETS_DIR = Path('ic1/annotation/datasets')
SENSITIVE_ROOT = Path('data/private/exports')  # gitignored; never committed
SHAREABLE_ROOT = Path('data/exports')  # git-tracked; Frictionless-described
PSEUDONYM_MAP = Path('.conf/coder_pseudonyms.json')  # gitignored; sensitive, stable