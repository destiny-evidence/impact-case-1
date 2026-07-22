from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline

from ._abc import _SklearnClassifierConfig


class SVMClassifierConfig(_SklearnClassifierConfig):
    name = 'svm'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('embed', TruncatedSVD(n_components=250)),
                ('scale', StandardScaler(with_std=True, with_mean=False)),
                ('clf', CalibratedClassifierCV(SVC(), ensemble=False)),
            ],
        )

        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'clf__estimator__kernel': 'linear',
            'clf__estimator__class_weight': 'balanced',
            'clf__estimator__degree': 3,
            'clf__estimator__gamma': 'auto',
            # 'clf__estimator__probability': True, -> FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
            'clf__estimator__C': 1.0,
            'clf__estimator__max_iter': 1000,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }
        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                'clf__estimator__C': trial.suggest_float('clf__estimator__C', low=0.001, high=1000, log=True),
                'clf__estimator__gamma': trial.suggest_float('clf__estimator__gamma', 0.001, 1.0, log=True),
                'clf__estimator__kernel': trial.suggest_categorical('clf__estimator__kernel', ['linear', 'rbf']),  # , 'poly', 'sigmoid'
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
