import logging
from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING

import joblib
import numpy as np

from ._abc import ClassifierBase

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
logger = logging.getLogger(__name__)


class SklearnClassifier(ClassifierBase):
    def __init__(
        self,
        pipeline: Callable[..., 'Pipeline'],  # | None = None,
        model_params: dict[str, Any] | None = None,
    ):
        super().__init__(model_params=model_params)
        self.get_pipeline = pipeline

        self.model_: 'Pipeline | None' = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: list[str], y: list[int]) -> 'SklearnClassifier':
        self.model_ = self.get_pipeline(**{k: v for k, v in self.model_params_.items() if k not in {'downsampling', 'ngram_range_max'}})
        self.model_.fit(X, y)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        if self.model_ is None:
            raise RuntimeError('Model must be trained before dumping non-preview params!')

        return {
            'model_params': self.model_.get_params(),
            'classes_': self.classes_,
            'pipeline': self.get_pipeline,
        }

    def predict_proba(self, X: list[str]) -> np.ndarray:
        if not self.model_:
            raise RuntimeError('Model must be trained before predicting!')
        y_pred: np.ndarray
        if hasattr(self.model_, 'predict_proba'):
            y_pred = self.model_.predict_proba(X)[:, 1]
        else:
            y_pred = self.model_.predict(X)[:, 1]
        logger.debug(f'  > Predictions include {(y_pred > 0.5).sum():,} records at threshold >0.5')
        return y_pred

    def predict(self, X: list[str]) -> np.ndarray:
        if not self.classes_:
            raise RuntimeError('Model must be trained before predicting!')
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path: Path) -> None:
        target = str((path / 'model.sklearn').resolve())
        logger.info(f'Saving trained model to {target}')
        if not self.model_:
            raise RuntimeError('Model must be trained before it can be saved!')
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            self.get_params()
            | {
                'model_': self.model_,
            },
            target,
        )

    @classmethod
    def load(cls, path: Path) -> 'SklearnClassifier':
        source = str((path / 'model.sklearn').resolve())
        logger.info(f'Loading trained model from {source}')
        info = joblib.load(source)
        classes = info.pop('classes_')
        model = info.pop('model_')
        classifier = cls(**info)
        classifier.model_ = model
        classifier.classes_ = classes
        return classifier
