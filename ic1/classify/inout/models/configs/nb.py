from typing import Any, TYPE_CHECKING

from ._abc import _SklearnClassifierConfig

if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline


class NaiveBayesClassifierConfig(_SklearnClassifierConfig):
    name = 'nb'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', MultinomialNB()),
            ],
        )
        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'cls__force_alpha': True,
            'cls__alpha': 1.0,
            'cls__fit_prior': True,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }
        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'cls__alpha': trial.suggest_float('cls__alpha', low=0.0, high=1.0),
                'cls__fit_prior': trial.suggest_categorical('cls__fit_prior', [True, False]),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
