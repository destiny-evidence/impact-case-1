from typing import Any, ClassVar, TYPE_CHECKING
from abc import abstractmethod, ABC

from ic1.core.config import settings

if TYPE_CHECKING:
    from optuna import Trial
    from torch import Tensor
    from sklearn.pipeline import Pipeline
    from ..classifiers import Classifier, HuggingfaceClassifier, SklearnClassifier


class _ClassifierConfig(ABC):
    name: ClassVar[str]  # each concrete config sets this to its literal name

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
    model_name: ClassVar[str]  # each concrete config sets this to its HF repo id
    max_length: ClassVar[int] = 512  # tokens kept per document (title + abstract); longer documents are truncated
    micro_batch_size: ClassVar[int] = 8  # per-device batch size; gradient accumulation raises it to the effective batch size of 16
    lr_range: ClassVar[tuple[float, float]] = (1e-5, 1e-4)  # fine-tuning learning rate search range

    @classmethod
    def get_model(cls, **kwargs: Any) -> 'HuggingfaceClassifier':
        from ..classifiers import HuggingfaceClassifier

        params = cls.get_params() | kwargs
        return HuggingfaceClassifier(model_name=params['model_name'], model_max_length=cls.max_length, model_params=params)

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        return {}

    @classmethod
    def params_base(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        micro = cls.micro_batch_size
        params: dict[str, Any] = {
            'model_name': cls.model_name,
            'output_dir': str(settings.OFFLINE_MODELS_DIR),
            'optim': 'adamw_torch',
            'save_strategy': 'no',
            'use_class_weights': 1,
            'learning_rate': 3e-5,
            'per_device_train_batch_size': micro,
            'gradient_accumulation_steps': 16 // micro,  # effective batch size 16
            'per_device_eval_batch_size': 2 * micro,
            'num_train_epochs': 3,
            'weight_decay': 0.01,
            'warmup_steps': 0.1,  # a float in [0, 1) is a share of the total steps
            'seed': 42,
            'data_seed': 42,
        }

        if trial:
            params |= {
                'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
                'learning_rate': trial.suggest_float('learning_rate', *cls.lr_range, log=True),
                'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 4),
                'weight_decay': trial.suggest_float('weight_decay', 0, 0.3),
            }

        return params
