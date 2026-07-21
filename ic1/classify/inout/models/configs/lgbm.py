from typing import Any, TYPE_CHECKING

from ._abc import _SklearnClassifierConfig

if TYPE_CHECKING:
    from optuna import Trial
    from sklearn.pipeline import Pipeline


class LightGBMClassifierConfig(_SklearnClassifierConfig):
    name = 'lgbm'

    @classmethod
    def get_pipeline(cls, **kwargs: Any) -> 'Pipeline':
        from lightgbm import LGBMClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', LGBMClassifier()),
            ],
        )
        pipeline.set_params(**kwargs)
        return pipeline

    @classmethod
    def params_default(cls, trial: 'Trial | None' = None) -> dict[str, Any]:
        params = {
            'cls__objective': 'binary',  # (not multiclass))
            'cls__learning_rate': 0.1,
            'cls__n_estimators': 100,  # Number of boosting rounds
            'cls__num_leaves': 31,  # Number of leaves in each tree
            'cls__random_state': None,  # For reproducibility
            'cls__verbose': -1,
            'vect__max_df': 0.8,
            'vect__min_df': 10,
        }

        if trial is not None:
            vect__ngram_range = trial.suggest_categorical('vect__ngram_range_max', [1, 2])
            params |= {
                # Controls step size in boosting
                'cls__learning_rate': trial.suggest_float('cls__learning_rate', 0.01, 0.2, log=True),
                # Number of boosting rounds (discrete float, behaves like int)
                'cls__n_estimators': trial.suggest_int('cls__n_estimators', 50, 500, log=True),
                # Number of leaves in each tree (higher = more complex)
                'cls__num_leaves': trial.suggest_int('cls__num_leaves', 10, 50, log=True),
                # Depth of trees (-1 means no limit)
                'cls__max_depth': trial.suggest_int('cls__max_depth', -1, 20),
                # Minimum data points in a leaf
                'cls__min_child_samples': trial.suggest_int('cls__min_child_samples', 5, 20, log=True),
                # Fraction of samples used in each boosting iteration
                'cls__subsample': trial.suggest_float('cls__subsample', 0.5, 1.0),
                # Fraction of features used per tree
                'cls__colsample_bytree': trial.suggest_float('cls__colsample_bytree', 0.5, 1.0),
                # L1 regularization
                'cls__reg_alpha': trial.suggest_float('cls__reg_alpha', 0.0, 1.0),
                # L2 regularization
                'cls__reg_lambda': trial.suggest_float('cls__reg_alpha', 0.0, 1.0),
                'vect__max_df': trial.suggest_float('vect__max_df', 0.5, 0.8),
                'vect__min_df': trial.suggest_int('vect__min_df', 5, 15),
                'vect__ngram_range': (1, vect__ngram_range),
            }
        return params
