from typing import Any, TYPE_CHECKING
from abc import abstractmethod, ABC

from ic1.core.config import settings

if TYPE_CHECKING:
    from optuna import Trial
    from torch import Tensor
    from sklearn.pipeline import Pipeline
    from ..classifiers import Classifier, HuggingfaceClassifier, SklearnClassifier


class _ClassifierConfig(ABC):
    @property  # type:ignore [misc]
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_model(cls, **kwargs: Any) -> 'Classifier':
        """Instantiate a model with the given hyperparameters"""
        raise NotImplementedError()

    @classmethod
    def get_params(cls, trial: 'Trial | None' = None, class_weights: 'list[float] | Tensor | None' = None) -> dict[str, Any]:
        params = cls.params_base(trial=trial) | cls.params_default(trial=trial)

        if class_weights:
            params['class_weights'] = class_weights

        return params

    @classmethod
    @abstractmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        """Default hyper-parameter parameters for this model; if trial specified, adds hyper-parameter search space"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def params_base(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        """Basic parameters that are the same across all configs for models of this kind; if trial specified, adds hyper-parameter search space"""
        raise NotImplementedError()


class _SklearnClassifierConfig(_ClassifierConfig, ABC):
    @classmethod
    @abstractmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        raise NotImplementedError()

    @classmethod
    def get_model(cls, **kwargs: Any) -> 'SklearnClassifier':
        from ..classifiers import SklearnClassifier

        return SklearnClassifier(pipeline=cls.get_pipeline, model_params=cls.get_params() | kwargs)

    @classmethod
    def params_base(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        if not trial:
            return {}
        return {
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }


class _HuggingfaceClassifierConfig(_ClassifierConfig, ABC):
    @property  # type:ignore [misc]
    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def get_model(cls, **kwargs: Any) -> 'HuggingfaceClassifier':
        from ..classifiers import HuggingfaceClassifier

        params = cls.get_params() | kwargs
        return HuggingfaceClassifier(model_name=params['model_name'], model_params=params)

    @classmethod
    def params_base(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            'model_name': cls.model_name,
            'output_dir': str(settings.OFFLINE_MODELS_DIR),
            'optim': 'adamw_torch',
            'save_strategy': 'no',
            'use_class_weights': 1,
            'learning_rate': 5e-3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 12,
            'num_train_epochs': 3,
            'weight_decay': 0.01,
        }

        if trial:
            params |= {
                'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
                # 'use_class_weights': trial.suggest_categorical('use_class_weights', [0, 1]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
                'num_train_epochs': trial.suggest_int('num_train_epochs', 1, 8),
                'weight_decay': trial.suggest_float('weight_decay', 0, 0.3),
            }

        return params
