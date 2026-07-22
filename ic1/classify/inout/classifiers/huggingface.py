"""
A list of transformer Classifiers and parameter spaces to validate.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from pathlib import Path
import json
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning

from ic1.core.config import settings
from ic1.classify.inout.utils import compute_class_weights, evaluate
from ._abc import ClassifierBase

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import TokenizersBackend
    from transformers.trainer_utils import PredictionOutput


logger = logging.getLogger('classify.inout.transformer')
logging.getLogger('urllib3').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

_custom_classes = None


def _get_custom_classes():  # type: ignore[no-untyped-def]
    global _custom_classes
    if _custom_classes is not None:
        return _custom_classes  # type: ignore[unreachable]

    import torch
    from torch import nn
    from transformers import Trainer, TrainingArguments
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()  # type: ignore[no-untyped-call]

    @dataclass
    class CustomTrainingArguments(TrainingArguments):
        use_class_weights: bool | int = field(default=False, metadata={'help': 'Whether to use class weights in loss function'})
        class_weights: list[float] | np.ndarray | None = field(default=None, metadata={'help': 'The weights for each class to be passed to the loss function'})
        model_name: str = field(default='prajjwal1/bert-tiny', metadata={'help': 'Name of the huggingface model'})

    class CustomTrainer(Trainer):
        args: CustomTrainingArguments

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self.activation = nn.Softmax(dim=1)
            self.loss = nn.CrossEntropyLoss

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None) -> tuple[torch.Tensor, PredictionOutput] | torch.Tensor:  # type: ignore[no-untyped-def]
            y_true = inputs.pop('labels')
            outputs = model(**inputs)
            y_pred = self.activation(outputs.logits)

            criterion = self.loss(weight=self.args.class_weights if self.args.use_class_weights else None)  # type: ignore[arg-type]
            loss = criterion(y_pred, y_true)

            return (loss, outputs) if return_outputs else loss

        def predict_proba(self, test_dataset: Dataset) -> np.ndarray:
            predictions = self.predict(test_dataset).predictions
            logits = predictions if torch.is_tensor(predictions) else torch.tensor(predictions)
            # return self.activation(logits).numpy()
            return logits.numpy()  # FIXME: does this still work? returning unscaled logits should be better for ranking

    _custom_classes = (CustomTrainingArguments, CustomTrainer)
    return _custom_classes


def evaluate_trainer(predictions: PredictionOutput) -> dict[str, Any]:
    import torch
    from torch import tensor

    with torch.no_grad():
        return evaluate(y_true=tensor(predictions.label_ids), y_pred=torch.softmax(tensor(predictions.predictions), dim=1)[:, 1])


class HuggingfaceClassifier(ClassifierBase):
    def __init__(
        self,
        model_name: str,
        model_max_length: int = 512,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_params=model_params)
        self.model_name = model_name
        self.model_max_length = model_max_length

        CustomTrainingArguments, CustomTrainer = _get_custom_classes()  # type: ignore[no-untyped-call]

        self.model_: CustomTrainer | None = None  # type: ignore[valid-type] # noqa: F821
        self.tokenizer_: TokenizersBackend | None = None
        self.classes_: np.ndarray | None = None

    @property
    def num_labels(self) -> int:
        if self.classes_ is None:
            raise RuntimeError('Model not initialised')
        return len(self.classes_)

    def fit(self, X: list[str], y: list[int]) -> 'HuggingfaceClassifier':
        self.train(x=X, y=y)
        return self

    def train(
        self,
        x: list[str] | None = None,
        y: list[int] | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        CustomTrainingArguments, CustomTrainer = _get_custom_classes()  # type: ignore[no-untyped-call]

        if dataset is None and x is None:
            raise RuntimeError('Must provide dataset or list of texts')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_params = self.model_params_
        if dataset is None and x is not None:
            logger.info(f'Preparing tokenised dataset from {len(x):,} texts')
            dataset = self.tokenize(texts=x, labels=np.array(y) if y is not None else None)
            # Set class weights if we have y values
            if y is not None:
                model_params['class_weights'] = torch.tensor(compute_class_weights(y), device=device, dtype=torch.float)
                self.classes_ = np.unique(y)
        elif dataset is not None:
            # We need to get classes from the dataset if available
            if 'labels' in dataset.features:
                self.classes_ = np.unique(dataset['labels'])
            else:
                logger.warning('Dataset does not contain labels')

        train_args = CustomTrainingArguments(**model_params)

        logger.debug(f'Training fresh transformer model using "{train_args.model_name}"')
        model = AutoModelForSequenceClassification.from_pretrained(
            train_args.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
            cache_dir=settings.OFFLINE_MODELS_DIR,
        )

        self.model_ = CustomTrainer(model=model, args=train_args, train_dataset=dataset)
        result = self.model_.train(resume_from_checkpoint=None)

        logger.debug(f'Time: {result.metrics["train_runtime"]:.2f}')
        logger.debug(f'Samples/second: {result.metrics["train_samples_per_second"]:.2f}')

    def tokenize(self, texts: list[str], labels: np.ndarray | None) -> Dataset:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from torch import tensor, long

        if self.tokenizer_ is None:
            try:
                self.tokenizer_ = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
                    self.model_name,
                    model_max_length=self.model_max_length,
                    cache_dir=settings.OFFLINE_MODELS_DIR,
                )
            except ValueError:
                # from transformers import BertTokenizerFast
                # tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=settings.OFFLINE_MODELS_DIR)
                self.tokenizer_ = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
                    self.model_name,
                    model_max_length=self.model_max_length,
                    cache_dir=settings.OFFLINE_MODELS_DIR,
                    use_fast=False,
                )

        params: dict[str, Any] = {'text': texts}
        if labels is not None:
            params['labels'] = tensor(labels, dtype=long)
        dataset = Dataset.from_dict(params)

        dataset = dataset.map(lambda x: self.tokenizer_(x['text'], padding='max_length', truncation=True), batched=True)  # type: ignore[misc]
        dataset.set_format('torch')

        return dataset.remove_columns('text')

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        if self.model_ is None or self.classes_ is None:  # type: ignore[unreachable]
            raise RuntimeError('Model must be trained before dumping non-preview params!')

        return {  # type: ignore[unreachable]
            'model_params': self.model_.args.to_dict(),
            'model_name': self.model_name,
            'model_max_length': self.model_max_length,
            'classes_': self.classes_,
        }

    def predict_proba(self, X: list[str]) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError('Model must be trained before predicting!')

        logger.debug(f'Tokenising {len(X):,} texts')

        dataset = self.tokenize(texts=X, labels=None)
        logger.debug('Predicting on texts')
        # self.model_.eval()
        with torch.no_grad():
            y_pred: np.ndarray = self.model_.predict_proba(dataset).numpy()[:, 1]  # type: ignore[attr-defined]
        logger.debug(f'  > Predictions include {(y_pred > 0.5).sum():,} records at threshold >0.5')
        return y_pred

    def predict(self, X: list[str]) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError('Model must be trained before predicting!')
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path: Path) -> None:
        target = str(path.resolve())
        logger.info(f'Saving trained "{self.model_name}" model to {target}')
        if self.model_ is None:
            raise RuntimeError('Model must be trained before it can be saved!')
        self.model_.save_model(target)  # type: ignore[attr-defined]
        with open(path / 'model_info.json', 'w') as fp:
            json.dump(self.get_params(), fp=fp, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'HuggingfaceClassifier':
        from transformers import AutoModelForSequenceClassification

        source = str(path.resolve())
        logger.info(f'Loading trained model from {source}')
        with open(path / 'model_info.json', 'r') as fp:
            info = json.load(fp)
        classes = info.pop('classes_')
        model = cls(**info)
        model.classes_ = classes
        model.model_ = AutoModelForSequenceClassification.from_pretrained(source)
        return model

    # def best_from_study(self, study: Study) -> 'TransformerClassifier':
    #     params = self._get_params(preview=False)
    #     params['model_params'] |= study.best_trial.user_attrs['model_params']
    #     return TransformerClassifier(**params)
    #
    # def _run_trial(self, trial: Trial, x: list[str], y: list[int]) -> float:
    #     logger.debug(f'Running tuning trial {trial.number}')
    #     training_args = self._train_args(trial=trial, weights=compute_class_weights(np.array(y)))
    #     dataset = tokenize(
    #         texts=x,
    #         labels=np.array(y),
    #         model=training_args.model_name,
    #         cache_dir=settings.OFFLINE_MODELS_DIR,
    #     )
    #     dataset = dataset.shuffle()
    #     train_dataset = dataset.take(n=int(self.tuning_split * len(dataset)))
    #     test_dataset = dataset.skip(n=int(self.tuning_split * len(dataset)))
    #
    #     self._train(args=training_args, dataset=train_dataset)
    #     predictions = self.trainer.predict(test_dataset)
    #     results = evaluate_trainer(predictions=predictions)
    #
    #     logger.info(f'Performance: {results}')
    #
    #     return results['F1']
