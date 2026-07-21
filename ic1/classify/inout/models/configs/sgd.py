from typing import Any, TYPE_CHECKING

from ._abc import _SklearnClassifierConfig

if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline


class SGDClassifierConfig(_SklearnClassifierConfig):
    name = 'sgd'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from sklearn.linear_model import SGDClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', SGDClassifier()),
            ],
        )
        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'cls__class_weight': 'balanced',
            'cls__loss': 'log_loss',
            'cls__max_iter': 1000,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }

        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'cls__alpha': trial.suggest_float('cls__alpha', 0.0001, 100000, log=True),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
