"""config for IC1 scripts."""

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

try:
    from nacsos_data.util.conf import DatabaseConfig
except:
    from pydantic import BaseModel
    class DatabaseConfig(BaseModel):
        pass

from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings

from ic1.core.ids import INOUT_SCHEME_ID, TAXONOMY_SCHEME_ID, TAXONOMY_SCOPE_IDS, TAXONOMY_SCOPE_IDS_RESOLVED, INOUT_SCOPE_IDS_ANNOTATE, INOUT_SCOPE_IDS_RESOLVED


class TaskName(str, Enum):
    INOUT = 'inout'
    TAXONOMY = 'taxonomy'
    ALL = 'all'


class SchemeFilesConfig(BaseSettings):
    BASE: Path = Path('data/scheme')
    VOCAB: Path = Field(default_factory=lambda data: data['BASE'] / 'destiny-1-4-version-1-4-of-the-destiny-taxonomy.ttl')
    CSV: Path = Field(default_factory=lambda data: data['BASE'] / 'destiny_taxonomy_nacsos_mapping.csv')
    JSON: Path = Field(default_factory=lambda data: data['BASE'] / 'destiny_taxonomy_nacsos_mapping.json')


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv('CONF_FILE', '.conf/secret.env'),
        extra='allow',
        env_nested_delimiter='__',
        # env_prefix=
    )
    DEET_N: int = 1000
    DATAPACKAGE: Path = Path('datapackage.json')
    DB: DatabaseConfig = Field(default_factory=DatabaseConfig)
    SCHEME: SchemeFilesConfig = Field(default_factory=SchemeFilesConfig)

    DATASETS_DIR: Path = Path('ic1/annotation/datasets')
    SCHEME_DIR: Path = Path('ic1/annotation/scheme')
    VOCAB_FILE: Path = SCHEME_DIR / 'destiny-1-4-version-1-4-of-the-destiny-taxonomy.ttl'
    MAPPING_CSV: Path = SCHEME_DIR / 'destiny_taxonomy_nacsos_mapping.csv'
    MAPPING_JSON: Path = SCHEME_DIR / 'destiny_taxonomy_nacsos_mapping.json'
    MODELS_ROOT: Path = Path('data/models')
    SENSITIVE_ROOT: Path = Path('data/private/exports')  # gitignored; never committed
    SHAREABLE_ROOT: Path = Path('data/exports')  # git-tracked; Frictionless-described
    PSEUDONYM_MAP: Path = Field(default_factory=lambda data: data['SENSITIVE_ROOT'] / 'coder_pseudonyms.json')  # gitignored; sensitive, stable
    LOGGING_DIR: Path = Path('data/logs')
    OFFLINE_MODELS_DIR: Path = Path('data/.cache/models')


settings = Config()


@dataclass
class TaskConfig:
    task: TaskName
    scheme_id: str
    scope_ids: list[str]
    resolved_ids: list[str]

    dev_mode: bool = False

    @property
    def name(self) -> str:
        return self.task.value

    @property
    def shareable_path(self) -> Path:
        return settings.SHAREABLE_ROOT / f'{self.name}.csv'

    @property
    def sensitive_path(self) -> Path:
        return settings.SENSITIVE_ROOT / f'{self.name}.csv'

    @property
    def resolved_path(self) -> Path:
        return settings.SHAREABLE_ROOT / f'{self.name}_resolved.csv'

    @property
    def deet_data_path(self) -> Path:
        return settings.SENSITIVE_ROOT / f'{self.name}_deet.csv'

    @property
    def deet_project_path(self) -> Path:
        return Path('ic1/deet/projects') / self.name

    @property
    def splits_path(self) -> Path:
        return settings.SHAREABLE_ROOT / f'{self.name}_splits.json'

    @property
    def classifier_results_path(self) -> Path:
        """The place where classifier results are stored."""
        if self.dev_mode:
            return settings.MODELS_ROOT / 'testing' / self.name / 'results'
        else:
            return settings.MODELS_ROOT / self.name / 'results'

    @property
    def tuning_results_path(self) -> Path:
        """The place where tuning results are stored."""
        if self.dev_mode:
            return settings.MODELS_ROOT / 'testing' / self.name / 'tuning'
        else:
            return settings.MODELS_ROOT / self.name / 'tuning'

    @property
    def ml_model_runs_path(self) -> Path:
        """Path to a file containing model run results."""
        return self.classifier_results_path / 'model_runs.jsonl'

    @property
    def ml_test_result_path(self) -> Path:
        """Path to a file with test results."""
        return self.classifier_results_path / 'test.json'

    @property
    def ml_model_path(self) -> Path:
        return self.classifier_results_path / 'model'

    @property
    def ml_model_predictions_path(self) -> Path:
        return self.classifier_results_path / 'ml_predictions.csv'


TASKS: dict[TaskName, TaskConfig] = {
    TaskName.INOUT: TaskConfig(
        task=TaskName.INOUT,
        scheme_id=INOUT_SCHEME_ID,
        scope_ids=INOUT_SCOPE_IDS_ANNOTATE,
        resolved_ids=INOUT_SCOPE_IDS_RESOLVED,
    ),
    TaskName.TAXONOMY: TaskConfig(
        task=TaskName.TAXONOMY,
        scheme_id=TAXONOMY_SCHEME_ID,
        scope_ids=TAXONOMY_SCOPE_IDS,
        resolved_ids=TAXONOMY_SCOPE_IDS_RESOLVED,
    ),
}
