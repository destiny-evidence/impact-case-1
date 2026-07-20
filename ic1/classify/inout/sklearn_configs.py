"""A list of Sklearn Classifiers and parameter spaces to validate."""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import product
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from ic1.classify.base import BaseClassifier, ModelRun, score, hash_ids
from numpy.typing import NDArray
import numpy as np
from pathlib import Path
import joblib


class SklearnClassifier(BaseClassifier):
    def __init__(self, name: str, pipeline: Pipeline, param_grid: dict[str, Any], thresholds: list[float] | None = None):
        self.name = name
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.thresholds = thresholds or []
        self.config = {}

    def _fit(self, x: list[str], y: list[int]) -> None:
        self.pipeline.fit(x, y)

    def predict_proba(self, x: list[str]) -> NDArray[np.float16]:
        result = self.pipeline.predict_proba(x)[:, 1]
        assert isinstance(result, np.ndarray)
        return result

    def tune(self, x_train: list[str], y_train: list[int], x_val: list[str], y_val: list[int]) -> list[ModelRun]:
        """Run through all parameters and thresholds, saving results as ModelRuns."""
        param_keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        runs: list[ModelRun] = []
        train_hash = hash_ids(x_train)
        val_hash = hash_ids(x_val)
        for combination in product(*values):
            params = dict(zip(param_keys, combination, strict=True))
            self.pipeline.set_params(**params)
            self.fit(x_train, y_train)
            val_proba = self.pipeline.predict_proba(x_val)[:, 1]
            for threshold in self.thresholds:
                metrics = score(y_val, val_proba, threshold)
                runs.append(
                    ModelRun(
                        model=self.name,
                        config=params,
                        precision=metrics['precision'],
                        recall=metrics['recall'],
                        f1=metrics['f1'],
                        threshold=threshold,
                        train_hash=train_hash,
                        val_hash=val_hash,
                        fit_time=self.fit_time,
                    ),
                )

        return runs

    def save(self, path: Path) -> None:
        """save model to disk"""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / 'model.joblib')

    @classmethod
    def load(cls, path: Path) -> 'SklearnClassifier':
        return joblib.load(path / 'model.joblib')


CONFIGS: list[SklearnClassifier] = [
    SklearnClassifier(
        name='svm_tfidf',
        pipeline=Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', CalibratedClassifierCV(SVC(class_weight='balanced'), ensemble=False)),
            ],
        ),
        param_grid={
            'vect__max_df': (0.5, 0.8),
            'vect__min_df': (5, 15),
            'vect__ngram_range': ((1, 1), (1, 2)),
            'clf__estimator__kernel': ['linear'],
            'clf__estimator__C': [10, 1, 1e2, 1e3],
        },
    )
]
