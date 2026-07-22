import logging
import time
from pathlib import Path
from typing import Annotated, Type

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import StratifiedKFold, train_test_split

from .configs import MODEL_CONFIGS, ClassifierConfig
from .utils import load_data, TuningFold, hash_ids, Result, TASK, ClassifierHelper

logger = logging.getLogger(__name__)


def hyperparameter_tuning(
    models: Annotated[list[str], typer.Option(help='List of models to tune', default_factory=lambda: list(MODEL_CONFIGS.keys()))],
    dev_mode: Annotated[bool, typer.Option(help='Run in development mode')] = False,
    num_folds: Annotated[int, typer.Option(help='Number of folds for cross-validation')] = 3,
    random_seed: Annotated[int | None, typer.Option(help='Random seed for cross-validation')] = None,
    num_trials: Annotated[int | None, typer.Option(help='Number of trials for hyperparameter tuning')] = None,
    num_jobs: Annotated[int, typer.Option(help='Number of tuning jobs for parallel processing')] = 1,
    scoring: Annotated[str, typer.Option(help='Scoring metric for hyperparameter tuning')] = 'F1',
    decision_threshold: Annotated[float, typer.Option(help='Decision threshold for classification')] = 0.5,
    result_dir: Annotated[Path, typer.Option(help='Directory to write tuning results to')] = TASK.classifier_results_path,
) -> None:
    """Run all models and find the best hyperparameter setting for each and store results"""
    logger.info(f'Going to write tuning results to {result_dir}')
    result_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_data(dev=dev_mode)
    data = pd.concat([train, val, test] if dev_mode else [train, val])
    x: list[str] = data['text'].tolist()
    y: np.ndarray = data['label'].to_numpy()
    seed = random_seed
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(folds.split(x, y)):
        logger.info(f'Running fold {fold + 1}')
        if seed is not None:
            seed += 1
        for name in models:
            config: Type[ClassifierConfig] = MODEL_CONFIGS[name.upper()]
            logger.info(f'Tuning model {name.upper()} for fold {fold + 1}')
            if seed is not None:
                seed += 1

            tune_idx, test_idx = train_test_split(train_idx, test_size=0.33, random_state=seed)
            train_hash = hash_ids(train_idx)
            tune_hash = hash_ids(tune_idx)
            test_hash = hash_ids(test_idx)
            val_hash = hash_ids(val_idx)

            key = f'{name}_{fold:02}_{train_hash}_{tune_hash}_{test_hash}_{val_hash}'
            result_file = result_dir / f'{key}.json'
            if result_file.exists():
                logger.warning(f'Fold {fold + 1} of model {name.upper()} already exists, skipping (key: {key})')
                continue

            helper = ClassifierHelper(
                config=config,
                tuning_jobs=num_jobs,
                tuning_trials=num_trials,
            )
            logger.info(f'Running tuning for fold {fold + 1} of model {name.upper()}')
            start_time = time.time()
            study = helper.tune(
                X_train=[x[i] for i in tune_idx],
                y_train=y[tune_idx],
                X_test=[x[i] for i in test_idx],
                y_test=y[test_idx],
                scoring=scoring,
                threshold=decision_threshold,
            )
            tune_time = time.time() - start_time
            best_trial = study.best_trial

            logger.info(f'Training model with best parameters for fold {fold + 1} of model {name.upper()}')
            start_time = time.time()
            model = helper.best_from_study(study, X=[x[i] for i in train_idx], y=y[train_idx])
            fit_time = time.time() - start_time

            logger.info(f'Testing model trained with best parameters for fold {fold + 1} of model {name.upper()}')
            scores_val = helper.test(model=model, X=[x[i] for i in val_idx], y=y[val_idx])

            logger.info(f'Storing results to {result_file}')
            with open(result_file, 'w') as fp:
                fp.write(
                    TuningFold(
                        model=name,
                        params=best_trial.user_attrs['model_params'],
                        scores_self=Result.model_validate(best_trial.user_attrs['scores_self']),
                        scores_test=Result.model_validate(best_trial.user_attrs['scores_test']),
                        scores_val=[Result.model_validate(result) for result in scores_val],
                        train_hash=train_hash,
                        tune_hash=tune_hash,
                        test_hash=test_hash,
                        val_hash=val_hash,
                        tune_time=tune_time,
                        fit_time=fit_time,
                    ).model_dump_json(indent=2)
                )
