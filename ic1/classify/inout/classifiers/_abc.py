from pathlib import Path
from typing import Any
from abc import abstractmethod, ABC

from ic1.classify.inout.utils import TuningFold

from sklearn.base import BaseEstimator, ClassifierMixin


class ClassifierBase(ABC, BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
    ):
        self.model_params_ = model_params or {}

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model for re-use and deployment."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> 'ClassifierBase':
        """Load a model instance from the `path` saved by self.save()"""
        raise NotImplementedError

    @classmethod
    def from_path(cls, path: Path) -> 'ClassifierBase':
        if (path / 'model.sklearn').exists():
            from .sklearn import SklearnClassifier

            return SklearnClassifier.load(path)

        from .huggingface import HuggingfaceClassifier

        return HuggingfaceClassifier.load(path)

    @classmethod
    def from_run(cls, run: TuningFold) -> 'ClassifierBase':
        from ic1.classify.inout.configs import MODEL_CONFIGS

        return MODEL_CONFIGS[run.model].get_model(**run.params)
