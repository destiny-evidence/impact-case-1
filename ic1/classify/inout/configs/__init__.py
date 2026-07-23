from typing import Type

from .isoforest import IsolationForestClassifierConfig
from .lgbm import LightGBMClassifierConfig
from .nb import NaiveBayesClassifierConfig
from .randforest import RandomForestClassifierConfig
from .regression import RegressionClassifierConfig
from .sgd import SGDClassifierConfig
from .svm import SVMClassifierConfig
from .transformers import SciBertConfig, T5Config, SciNCLBertConfig, ClimateBertConfig

type ClassifierConfig = (
    IsolationForestClassifierConfig
    | LightGBMClassifierConfig
    | NaiveBayesClassifierConfig
    | RandomForestClassifierConfig
    | RegressionClassifierConfig
    | SGDClassifierConfig
    | SVMClassifierConfig
    | SciBertConfig
    | T5Config
    | SciNCLBertConfig
    | ClimateBertConfig
)

MODEL_CONFIGS: dict[str, Type[ClassifierConfig]] = {
    config.name.upper(): config  # type: ignore[attr-defined, misc]
    for config in [
        RegressionClassifierConfig,
        SVMClassifierConfig,
        LightGBMClassifierConfig,
        SGDClassifierConfig,
        NaiveBayesClassifierConfig,
        IsolationForestClassifierConfig,
        RandomForestClassifierConfig,
        SciBertConfig,
        T5Config,
        SciNCLBertConfig,
        ClimateBertConfig,
    ]
}

__all__ = [
    'MODEL_CONFIGS',
    'IsolationForestClassifierConfig',
    'LightGBMClassifierConfig',
    'SGDClassifierConfig',
    'NaiveBayesClassifierConfig',
    'RandomForestClassifierConfig',
    'SVMClassifierConfig',
    'SGDClassifierConfig',
    'SciBertConfig',
    'T5Config',
    'SciNCLBertConfig',
    'ClimateBertConfig',
    'ClassifierConfig',
]
