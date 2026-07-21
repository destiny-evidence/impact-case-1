from typing import Any, TYPE_CHECKING

from ._abc import _SklearnClassifierConfig

if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline


class RegressionClassifierConfig(_SklearnClassifierConfig):
    name = 'logreg'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', LogisticRegression()),
            ],
        )
        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'cls__class_weight': 'balanced',
            'cls__tol': 0.0001,
            'cls__C': 1.0,
            'cls__solver': 'lbfgs',
            'cls__max_iter': 100,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }
        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'cls__C': trial.suggest_float('cls__C', low=0.01, high=10, log=True),
                'cls__solver': trial.suggest_categorical('cls__solver', ['saga', 'liblinear', 'lbfgs']),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
