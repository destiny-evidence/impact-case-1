"""config for IC1 scripts."""
from typing import TypedDict
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from ic1.core.ids import (
    INOUT_SCHEME_ID,
    TAXONOMY_SCHEME_ID,
    INOUT_SCOPE_IDS,
    TAXONOMY_SCOPE_IDS
)

class TaskName(str, Enum):
    INOUT = 'inout'
    TAXONOMY = 'taxonomy'
    ALL = 'all'

@dataclass
class TaskConfig:
    name: str
    scheme_id: str
    scope_ids: list[str]

    @property
    def shareable_path(self) -> Path:
        return SHAREABLE_ROOT / f'{self.name}.csv'

    @property
    def sensitive_path(self) -> Path:
        return SENSITIVE_ROOT / f'{self.name}.csv'

    @property
    def resolved_path(self) -> Path:
        return SENSITIVE_ROOT / f'{self.name}_resolved.csv'


TASKS: dict[str, TaskConfig] = {
    'inout': TaskConfig(
        name= "inout",
        scheme_id = INOUT_SCHEME_ID,
        scope_ids = INOUT_SCOPE_IDS
    ),
    'taxonomy': TaskConfig(
        name="taxonomy",
        scheme_id=TAXONOMY_SCHEME_ID,
        scope_ids=TAXONOMY_SCOPE_IDS
    )
}

CONF_FILE = '.conf/secret.env'

DATAPACKAGE = Path('datapackage.json')

DATASETS_DIR = Path('ic1/annotation/datasets')
SENSITIVE_ROOT = Path('data/private/exports')  # gitignored; never committed
SHAREABLE_ROOT = Path('data/exports')  # git-tracked; Frictionless-described
PSEUDONYM_MAP = Path('.conf/coder_pseudonyms.json')  # gitignored; sensitive, stable