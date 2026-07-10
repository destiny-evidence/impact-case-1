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

    @property
    def classifier_results_path(self) -> Path:
        """The place where classifier results are stored."""
        return Path('ic1/classify') / self.name / 'results'

    @property
    def ml_model_runs_path(self) -> Path:
        """Path to a file containing model run results."""
        return self.classifier_results_path / 'model_runs.jsonl'

    @property
    def ml_test_result_path(self) -> Path:
        """Path to a file with test results."""
        return self.classifier_results_path / "test.json"

    @property
    def ml_model_path(self) -> Path:
        return self.classifier_results_path / 'model'

    @property
    def ml_model_predictions_path(self) -> Path:
        return self.classifier_results_path / 'ml_predictions.csv'


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
SCHEME_DIR = Path('ic1/annotation/scheme')
VOCAB_FILE = str(SCHEME_DIR / 'destiny-1-4-version-1-4-of-the-destiny-taxonomy.ttl')
MAPPING_CSV = SCHEME_DIR / 'destiny_taxonomy_nacsos_mapping.csv'
MAPPING_JSON = SCHEME_DIR / 'destiny_taxonomy_nacsos_mapping.json'
SENSITIVE_ROOT = Path('data/private/exports')  # gitignored; never committed
SHAREABLE_ROOT = Path('data/exports')  # git-tracked; Frictionless-described
PSEUDONYM_MAP = Path('.conf/coder_pseudonyms.json')  # gitignored; sensitive, stable
