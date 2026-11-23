"""
A list of transformer Classifiers and parameter spaces to validate.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path
import json
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning

from ic1.core.config import settings
from ic1.classify.inout.utils import compute_class_weights, evaluate
from ._abc import ClassifierBase

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import SentencePieceBackend, TokenizersBackend
    from transformers.trainer_utils import PredictionOutput


logger = logging.getLogger('classify.inout.transformer')
logging.getLogger('urllib3').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

CHARS_PER_TOKEN_CAP = 32  # texts are cut to this many characters per token of the length limit before tokenising

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

            # class_weights arrives as a Tensor from fit and as a list after get_params round-trip; CrossEntropyLoss accepts both paths via tensor coercion
            criterion = self.loss(weight=cast('torch.Tensor | None', self.args.class_weights if self.args.use_class_weights else None))
            loss = criterion(outputs.logits.float(), y_true)  # CrossEntropyLoss applies the log-softmax itself, so it takes raw logits

            return (loss, outputs) if return_outputs else loss

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
        self.tokenizer_: TokenizersBackend | SentencePieceBackend | None = None
        self.classes_: np.ndarray | None = None

    @property
    def num_labels(self) -> int:
        if self.classes_ is None:
            raise RuntimeError('Model not initialised')
        return len(self.classes_)

    def fit(self, X: list[str], y: np.ndarray) -> 'HuggingfaceClassifier':
        self.train(x=X, y=y)
        return self

    def train(
        self,
        x: list[str] | None = None,
        y: np.ndarray | None = None,
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
            dtype=torch.float32,  # checkpoints stored in bfloat16 would otherwise train in bfloat16
        )

        from transformers import DataCollatorWithPadding

        self.model_ = CustomTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=DataCollatorWithPadding(tokenizer=self._get_tokenizer()))
        result = self.model_.train(resume_from_checkpoint=None)

        logger.debug(f'Time: {result.metrics["train_runtime"]:.2f}')
        logger.debug(f'Samples/second: {result.metrics["train_samples_per_second"]:.2f}')

    def _get_tokenizer(self) -> TokenizersBackend | SentencePieceBackend:
        if self.tokenizer_ is None:
            from transformers import AutoTokenizer

            self.tokenizer_ = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.model_max_length,
                cache_dir=settings.OFFLINE_MODELS_DIR,
            )
        tokenizer = self.tokenizer_
        assert tokenizer is not None  # set above; re-narrows the attribute after the call
        return tokenizer

    def _encode(self, texts: list[str]) -> Any:
        """Tokenise without padding. Every text is truncated to `model_max_length` tokens, however long it is.

        The text is cut to a multiple of the token limit in characters first, which bounds the tokeniser's work on pathological inputs.
        """
        cap = self.model_max_length * CHARS_PER_TOKEN_CAP
        cleaned = [t[:cap] if isinstance(t, str) else '' for t in texts]
        return self._get_tokenizer()(cleaned, truncation=True, max_length=self.model_max_length)

    def tokenize(self, texts: list[str], labels: np.ndarray | None) -> Dataset:
        from datasets import Dataset

        encoded = self._encode(texts)
        data: dict[str, Any] = {key: encoded[key] for key in encoded.keys()}
        if labels is not None:
            data['labels'] = np.asarray(labels).astype(int).tolist()
        return Dataset.from_dict(data)  # padding happens per batch, in the data collator

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        if self.model_ is None or self.classes_ is None:  # type: ignore[unreachable]
            raise RuntimeError('Model must be trained before dumping non-preview params!')

        return {  # type: ignore[unreachable]
            'model_params': self.model_.args.to_dict(),
            'model_name': self.model_name,
            'model_max_length': self.model_max_length,
            'classes_': self.classes_.tolist(),
        }

    def predict_proba(self, X: list[str], batch_size: int | None = None) -> np.ndarray:
        """Probability of the positive class. Works on a freshly trained and on a loaded model, for texts of any length."""
        import torch

        if self.model_ is None:
            raise RuntimeError('Model must be trained before predicting!')

        # A fresh model is wrapped in a Trainer; a loaded one is the bare model
        model = self.model_.model if hasattr(self.model_, 'args') else self.model_
        device = next(model.parameters()).device
        if device.type == 'cpu' and torch.cuda.is_available():
            model.to('cuda')
            device = torch.device('cuda')
        model.eval()

        batch_size = batch_size or int(self.model_params_.get('per_device_eval_batch_size', 16))
        tokenizer = self._get_tokenizer()
        encoded = self._encode(X)
        n = len(encoded['input_ids'])
        order = np.argsort([len(ids) for ids in encoded['input_ids']])  # similar lengths per batch keeps padding small
        probs = np.zeros(n)
        logger.debug(f'Predicting on {n:,} texts')
        with torch.no_grad():
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                features = [{key: encoded[key][i] for key in encoded.keys()} for i in idx]
                batch = {key: value.to(device) for key, value in tokenizer.pad(features, return_tensors='pt').items()}
                logits = model(**batch).logits
                probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        logger.debug(f'  > Predictions include {(probs > 0.5).sum():,} records at threshold >0.5')
        return probs

    def predict(self, X: list[str]) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError('Model must be trained before predicting!')
        return self.classes_[(self.predict_proba(X) > 0.5).astype(int)]

    def save(self, path: Path) -> None:
        target = str(path.resolve())
        logger.info(f'Saving trained "{self.model_name}" model to {target}')
        if self.model_ is None:
            raise RuntimeError('Model must be trained before it can be saved!')
        # Save the model weights+config directly (safetensors) rather than via Trainer.save_model,
        # which also torch.saves self.args -- an instance of the locally-defined CustomTrainingArguments
        # that pickle cannot serialise. load() only needs the model + model_info.json, not training args.
        self.model_.model.save_pretrained(target)  # type: ignore[attr-defined]
        self._get_tokenizer().save_pretrained(target)  # so the saved model needs no hub access to run
        with open(path / 'model_info.json', 'w') as fp:
            # default=str guards against any residual non-JSON values (e.g. a torch tensor
            # class_weights in model_params); those fields are inert on reload.
            json.dump(self.get_params(), fp=fp, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'HuggingfaceClassifier':
        import torch
        from transformers import AutoModelForSequenceClassification

        source = str(path.resolve())
        logger.info(f'Loading trained model from {source}')
        with open(path / 'model_info.json', 'r') as fp:
            info = json.load(fp)
        classes = info.pop('classes_')
        model = cls(**info)
        model.classes_ = np.asarray(classes)
        model.model_ = AutoModelForSequenceClassification.from_pretrained(source, dtype=torch.float32)
        if (path / 'tokenizer_config.json').exists():
            from transformers import AutoTokenizer

            model.tokenizer_ = AutoTokenizer.from_pretrained(source, model_max_length=model.model_max_length)
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
