from .huggingface import HuggingfaceClassifier
from .sklearn import SklearnClassifier
from ._abc import ClassifierBase

type Classifier = HuggingfaceClassifier | SklearnClassifier

__all__ = ['HuggingfaceClassifier', 'SklearnClassifier', 'Classifier', 'ClassifierBase']
