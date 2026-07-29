from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from optuna import Trial

from ._abc import _HuggingfaceClassifierConfig


class ClimateBertConfig(_HuggingfaceClassifierConfig):
    name = 'climatebert'
    model_name = 'climatebert/distilroberta-base-climate-f'

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        if trial is None:
            return {}
        batch_size = trial.suggest_int('per_device_train_batch_size', 2, 16)
        return {
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
        }


# class TinyBertConfig(_HuggingfaceClassifierConfig):
#     name = 'tinybert'
#     model_name = 'prajjwal1/bert-tiny'
#
#     @classmethod
#     def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
#         if trial is None:
#             return {}
#         batch_size = trial.suggest_int('per_device_train_batch_size', 2, 32)
#         return {
#             'optim': trial.suggest_categorical('optim', ['adamw_torch', 'adafactor', 'lion_32bit']),
#             'per_device_train_batch_size': batch_size,
#             'per_device_eval_batch_size': batch_size,
#         }
class T5Config(_HuggingfaceClassifierConfig):
    name = 't5-small'
    model_name = 'google-t5/t5-small'

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        if trial is None:
            return {}
        batch_size = trial.suggest_int('per_device_train_batch_size', 2, 32)
        return {
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
        }


class SciBertConfig(_HuggingfaceClassifierConfig):
    name = 'scibert'
    model_name = 'allenai/scibert_scivocab_uncased'

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
        }
        if trial is not None:
            batch_size = trial.suggest_int('per_device_train_batch_size', 2, 16)
            params |= {
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
            }
        return params


class SciNCLBertConfig(_HuggingfaceClassifierConfig):
    name = 'scincl'
    model_name = 'malteos/scincl'

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        if trial is None:
            return {}
        batch_size = trial.suggest_int('per_device_train_batch_size', 2, 16)
        return {
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
        }


# TODO: https://huggingface.co/FacebookAI/roberta-large
# TODO: https://huggingface.co/google/flan-t5-base
