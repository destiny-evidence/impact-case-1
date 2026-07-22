"""Abstract base class for classifiers."""

import logging
from copy import deepcopy
import optuna
from optuna import Study, Trial
import numpy as np
from typing import Any, Callable, Type, TYPE_CHECKING

from ic1.core.utils import downsampling_mask, mask_list
from ..utils import evaluate, TuningFold

if TYPE_CHECKING:
    from .classifiers import Classifier
    from .configs import ClassifierConfig


logger = logging.getLogger(__name__)


class ClassifierHelper:
    def __init__(
        self,
        config: 'Type[ClassifierConfig]',
        tuning_trials: int | None = None,
        tuning_jobs: int = 1,
        thresholds: list[float] | None = None,
        hp_space: Callable[['Trial'], dict[str, Any]] | None = None,
        model_params: dict[str, Any] | None = None,
    ):
        self.config = config
        self.hp_space = hp_space or (lambda trial: {})
        self.model_params = model_params or {}
        self.thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.tuning_trials = tuning_trials
        self.tuning_jobs = tuning_jobs

    def _train_params(self, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = self.config.get_params(trial=trial) | self.model_params
        if trial:
            params |= self.hp_space(trial=trial)  # type: ignore[call-arg]
        return params

    def train(self, X: list[str], y: list[int], model_params: dict[str, Any] | None = None) -> 'Classifier':
        logger.info('Training model...')
        model_params = self._train_params() | (model_params or {})
        model = self.config.get_model(**model_params)
        y_ = np.array(y)
        sampling = model_params.pop('downsampling', 0)
        mask = downsampling_mask(y_, sampling=sampling)
        model.fit(mask_list(X, mask), y_[mask])
        return model

    def test(self, model: 'Classifier', X: list[str], y: list[int]) -> list[dict[str, float]]:
        logger.info('Testing model...')
        y_pred = model.predict_proba(X)
        return [evaluate(y_true=np.array(y), y_pred=y_pred, threshold=th) for th in self.thresholds]

    def tune(self, X_train: list[str], y_train: list[int], X_test: list[str], y_test: list[int], scoring: str = 'F1', threshold: float = 0.5) -> Study:
        # TODO: Check which sampler makes most sense: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(self.tuning_trials * 0.5))
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(
            lambda trial: self._run_trial(trial=trial, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring, threshold=threshold),
            n_trials=self.tuning_trials,
            n_jobs=self.tuning_jobs,
        )
        logger.info(f'Best trial: {study.best_trial.user_attrs["model_params"]}')
        logger.debug(f'Hyper-parameter-tuning for {self.config.name} done with best score {study.best_value}')
        return study

    def _run_trial(
        self, trial: Trial, X_train: list[str], y_train: list[int], X_test: list[str], y_test: list[int], scoring: str = 'F1', threshold: float = 0.5
    ) -> float:
        model_params = self._train_params(trial=trial)
        model_params_ = deepcopy(model_params)
        sampling = model_params.pop('downsampling', 0)

        model = self.config.get_model(**model_params)

        if sampling > 0:
            y = np.array(y_train)
            mask = downsampling_mask(y, sampling=sampling)
            logger.debug(f'Downsampling from {y.shape[0]:,} ({y.sum():,} incl) to {mask.sum():,} ({y[mask].sum():,} incl)')
            logger.debug(f'Preparing tuning trial with model_params: {model_params}')

            logger.info(f'Fitting model in tuning trial {trial.number} on {mask.sum():,} samples (downsampled)')
            model.fit(mask_list(X_train, mask), y[mask])
        else:
            model.fit(X_train, y_train)

        logger.debug('Predicting on tuning training data')
        y_pred = model.predict_proba(X_train)
        scores_self = evaluate(y_true=np.array(y_train), y_pred=y_pred, threshold=threshold)
        logger.debug(f'Self scores: {scores_self}')

        logger.debug('Predicting on tuning test data')
        y_pred = model.predict_proba(X_test)
        scores_test = evaluate(y_true=np.array(y_test), y_pred=y_pred, threshold=threshold)
        logger.debug(f'Test scores: {scores_test}')

        trial.set_user_attr('scores_self', scores_self)
        trial.set_user_attr('scores_test', scores_test)
        trial.set_user_attr('model_params', model_params_)

        objective = scores_test[scoring]
        return 0 if np.isnan(objective) else objective

    def best_from_study(self, study: Study, X: list[str], y: list[int]) -> 'Classifier':
        return self.train(X=X, y=y, model_params=study.best_trial.user_attrs['model_params'])

    @classmethod
    def from_run(cls, run: TuningFold) -> 'ClassifierHelper':
        from ic1.classify.inout.configs import MODEL_CONFIGS

        return cls(
            config=MODEL_CONFIGS[run.model],
            model_params=run.params,
        )
