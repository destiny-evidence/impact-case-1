"""Defines a model to store how data is split for evaluation"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path

from ic1.core.config import TaskName


class EvaluationSplits(BaseModel):
    task: TaskName
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deet: list[str] = []
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []

    @classmethod
    def load(cls, path: Path) -> 'EvaluationSplits':
        return cls.model_validate_json(path.read_text())

    def add_items(
        self, deet_ids: list[str] = [], train_ids: list[str] = [], validation_ids: list[str] = [], test_ids: list[str] = [], exclude_deet: bool = False
    ) -> None:
        """Add items to splits, as long as they are not already assigned somewhere."""
        already_assigned = set(self.train + self.validation + self.test)
        if not exclude_deet:
            already_assigned |= set(self.deet)
        self.deet.extend(iid for iid in deet_ids if iid not in already_assigned)
        self.train.extend(iid for iid in train_ids if iid not in already_assigned)
        self.validation.extend(iid for iid in validation_ids if iid not in already_assigned)
        self.test.extend(iid for iid in test_ids if iid not in already_assigned)
