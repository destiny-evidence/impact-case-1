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
        self,
        deet_ids: list[str] | None = None,
        train_ids: list[str] | None = None,
        validation_ids: list[str] | None = None,
        test_ids: list[str] | None = None,
        exclude_deet: bool = False,
    ) -> None:
        """Add items to splits, as long as they are not already assigned somewhere."""
        already_assigned = set(self.train + self.validation + self.test)
        if not exclude_deet:
            already_assigned |= set(self.deet)

        if deet_ids is not None:
            self.deet += list(set(deet_ids) - already_assigned)
        if train_ids is not None:
            self.train += list(set(train_ids) - already_assigned)
        if validation_ids is not None:
            self.validation += list(set(validation_ids) - already_assigned)
        if test_ids is not None:
            self.test += list(set(test_ids) - already_assigned)
