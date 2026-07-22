from typing import Any, TYPE_CHECKING

from ._abc import _SklearnClassifierConfig

if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline


class IsolationForestClassifierConfig(_SklearnClassifierConfig):
    name = 'iso_forest'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from sklearn.ensemble import IsolationForest
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', IsolationForest()),
            ],
        )
        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'clf__n_estimators': 100,
            'clf__max_samples': 'auto',
            'clf__contamination': 'auto',
            'clf__max_features': 1.0,
            'clf__bootstrap': False,
            'clf__n_jobs': None,
            'clf__random_state': None,
            'clf__verbose': 0,
            'clf__warm_start': False,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }
        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'clf__n_estimators': trial.suggest_int('clf__n_estimators', low=20, high=250),
                'clf__max_features': trial.suggest_float('clf__max_features', low=0.2, high=1.0),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
