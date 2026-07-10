"""config for IC1 scripts."""
from pathlib import Path
from enum import Enum
from dataclasses import dataclass


from ic1.core.ids import (
    INOUT_SCHEME_ID,
    TAXONOMY_SCHEME_ID,
    INOUT_SCOPE_IDS,
    TAXONOMY_SCOPE_IDS
)

DEET_N = 100

class TaskName(str, Enum):
    INOUT = 'inout'
    TAXONOMY = 'taxonomy'
    ALL = 'all'

@dataclass
class TaskConfig:
    task: TaskName
    scheme_id: str
    scope_ids: list[str]

    @property
    def name(self) -> str:
        return self.task.value

    @property
    def shareable_path(self) -> Path:
        return SHAREABLE_ROOT / f'{self.name}.csv'

    @property
    def sensitive_path(self) -> Path:
        return SENSITIVE_ROOT / f'{self.name}.csv'

    @property
    def resolved_path(self) -> Path:
        return SENSITIVE_ROOT / f'{self.name}_resolved.csv'

    @property
    def deet_data_path(self) -> Path:
        return SENSITIVE_ROOT / f'{self.name}_deet.csv'

    @property
    def deet_project_path(self) -> Path:
        return Path('ic1/deet/projects') / self.name

    @property
    def splits_path(self) -> Path:
        return Path('ic1/evaluation_splits') / f'{self.name}_splits.json'


TASKS: dict[str, TaskConfig] = {
    'inout': TaskConfig(
        task= TaskName.INOUT,
        scheme_id = INOUT_SCHEME_ID,
        scope_ids = INOUT_SCOPE_IDS
    ),
    'taxonomy': TaskConfig(
        task=TaskName.TAXONOMY,
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
