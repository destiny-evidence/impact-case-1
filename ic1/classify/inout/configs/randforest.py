from typing import Any, TYPE_CHECKING

from ._abc import _SklearnClassifierConfig

if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline


class RandomForestClassifierConfig(_SklearnClassifierConfig):
    name = 'rand_forest'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', RandomForestClassifier()),
            ],
        )
        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            'clf__n_estimators': 1000,
            'clf__verbose': 0,
            'clf__random_state': None,
            'clf__max_features': 'sqrt',
            'clf__min_samples_split': 2,
            'clf__max_depth': None,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }

        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'clf__n_estimators': trial.suggest_int('clf__n_estimators', low=100, high=5000),
                'clf__max_features': trial.suggest_categorical('clf__max_features', ['sqrt', 'log2']),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
