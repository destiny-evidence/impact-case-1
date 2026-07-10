"""Data models and abcs for classifiers and classifier runs."""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from sklearn.metrics import precision_score, recall_score, f1_score
from numpy.typing import NDArray, ArrayLike
import numpy as np
import time
from typing import Any
import hashlib, json

class ModelRun(BaseModel):
    model: str
    config: dict[str, Any]
    precision: float
    recall: float
    f1: float
    threshold: float
    train_hash: str
    val_hash: str
    fit_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BaseClassifier(ABC):
    name: str
    config: dict[str, Any]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    fit_time: float = 0.0

    def fit(self, x: list[str], y: list[int]) -> None:
        start = time.perf_counter()
        self._fit(x, y)
        self.fit_time = time.perf_counter() - start

    @abstractmethod
    def _fit(self, x: list[str], y: list[int]) -> None: ...

    @abstractmethod
    def tune(self,
        x_train: list[str],
        y_train: list[int],
        x_val: list[str],
        y_val: list[int]
    ) -> list[ModelRun]:
        """Try out a load of configs, and return a model run for each."""

    @abstractmethod
    def predict(self, x: list[str]) -> NDArray[np.int_]: ...

    @abstractmethod
    def predict_proba(self, x: list[str]) -> NDArray[np.float16]: ...

def score(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float
) -> dict[str, float]:
    """Turn probs into ints using threshold, and score"""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred))
    }

def hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]
