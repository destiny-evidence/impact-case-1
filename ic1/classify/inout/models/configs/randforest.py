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
        params = {
            'cls__n_estimators': 1000,
            'cls__verbose': 0,
            'cls__random_state': None,
            'cls__max_features': 'sqrt',
            'cls__min_samples_split': 2,
            'cls__max_depth': None,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }

        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'cls__n_estimators': trial.suggest_int('cls__n_estimators', low=100, high=5000),
                'cls__max_features': trial.suggest_categorical('cls__max_features', ['sqrt', 'log2']),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
