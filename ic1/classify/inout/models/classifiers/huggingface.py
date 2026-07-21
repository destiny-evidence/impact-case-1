"""
A list of transformer Classifiers and parameter spaces to validate.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import json
import numpy as np


import torch
from torch import tensor, nn

from sklearn.exceptions import UndefinedMetricWarning

from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, TokenizersBackend
from transformers.trainer_utils import PredictionOutput
from transformers.utils.logging import disable_progress_bar


from ic1.core.config import settings
from ._abc import ClassifierBase
from ic1.classify.inout.utils import compute_class_weights, evaluate

logger = logging.getLogger('classify.inout.transformer')
logging.getLogger('urllib3').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
disable_progress_bar()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_trainer(predictions: PredictionOutput):
    with torch.no_grad():
        return evaluate(y_true=tensor(predictions.label_ids), y_pred=torch.softmax(tensor(predictions.predictions), dim=1)[:, 1])


@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_class_weights: bool | int = field(default=False, metadata={'help': 'Whether to use class weights in loss function'})
    class_weights: list[float] | np.ndarray | None = field(default=None, metadata={'help': 'The weights for each class to be passed to the loss function'})
    model_name: str = field(default='prajjwal1/bert-tiny', metadata={'help': 'Name of the huggingface model'})


class CustomTrainer(Trainer):
    args: CustomTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        y_true = inputs.pop('labels')
        outputs = model(**inputs)
        y_pred = self.activation(outputs.logits)

        criterion = self.loss(weight=self.args.class_weights if self.args.use_class_weights else None)
        loss = criterion(y_pred, y_true)

        return (loss, outputs) if return_outputs else loss

    def predict_proba(self, test_dataset: Dataset) -> np.array:
        predictions = self.predict(test_dataset).predictions
        logits = predictions if torch.is_tensor(predictions) else tensor(predictions)
        # return self.activation(logits).numpy()
        return logits.numpy()  # FIXME: does this still work? returning unscaled logits should be better for ranking


class HuggingfaceClassifier(ClassifierBase):
    def __init__(
        self,
        model_name: str,
        model_max_length: int = 512,
        model_params: dict[str, Any] | None = None,
    ):
        super().__init__(model_params=model_params)
        self.model_name = model_name
        self.model_max_length = model_max_length

        self.model_: CustomTrainer | None = None
        self.tokenizer_: TokenizersBackend | None = None
        self.classes_: np.ndarray | None = None

    @property
    def num_labels(self) -> int:
        if not self.classes_:
            raise RuntimeError('Model not initialised')
        return len(self.classes_)

    def fit(self, X, y):
        self.train(x=X, y=y)
        return self

    def train(
        self,
        x: list[str] | None = None,
        y: list[int] | None = None,
        dataset: Dataset | None = None,
    ):
        if dataset is None and x is None:
            raise RuntimeError('Must provide dataset or list of texts')

        model_params = self.model_params_
        if dataset is not None:
            model_params['class_weights'] = torch.tensor(compute_class_weights(dataset['labels']), device=device, dtype=torch.float)
            self.classes_ = np.unique(dataset['labels'])
        elif y is not None:
            model_params['class_weights'] = torch.tensor(compute_class_weights(y), device=device, dtype=torch.float)
            self.classes_ = np.unique(y)

        train_args = CustomTrainingArguments(**model_params)

        if dataset is None and x is not None:
            logger.info(f'Preparing tokenised dataset from {len(x):,} texts')
            dataset = self.tokenize(texts=x, labels=np.array(y) if y is not None else None)

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
        """
        Returns tokenised dataset from texts and labels using the given model name or filepath
        If using direct path, don't forget to use `--tokenizer` postfix: `{/path/to/model}--tokenizer`
        """
        if not self.tokenizer_:
            self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.model_max_length, cache_dir=settings.OFFLINE_MODELS_DIR)

        dataset = Dataset.from_dict(
            {
                'text': texts,
                'labels': labels,
            },
        )

        dataset = dataset.map(lambda x: self.tokenizer_(x['text'], padding='max_length', truncation=True), batched=True)
        dataset.set_format('torch')

        return dataset.remove_columns('text')

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        if self.model_ is None:
            raise RuntimeError('Model must be trained before dumping non-preview params!')

        return {
            'model_params': self.model_.args.to_dict(),
            'model_name': self.model_name,
            'model_max_length': self.model_max_length,
            'classes_': self.classes_,
        }

    def predict_proba(self, X: list[str]):
        logger.debug(f'Tokenising {len(X):,} texts')

        dataset = self.tokenize(texts=X, labels=None)
        logger.debug('Predicting on texts')
        # self.model_.eval()
        with torch.no_grad():
            y_pred = self.model_.predict_proba(dataset).numpy()[:, 1]
        logger.debug(f'  > Predictions include {(y_pred > 0.5).sum():,} records at threshold >0.5')
        return y_pred

    def predict(self, X: list[str]):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path: Path) -> None:
        target = str(path.resolve())
        logger.info(f'Saving trained "{self.model_name}" model to {target}')
        if not self.model_:
            raise RuntimeError('Model must be trained before it can be saved!')
        self.model_.save_model(target)
        with open(path / 'model_info.json', 'w') as fp:
            json.dump(self.get_params(), fp=fp, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'HuggingfaceClassifier':
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
