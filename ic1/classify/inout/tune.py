import logging
import os
import time
from pathlib import Path
from typing import Annotated, Type

import numpy as np
import typer

from .configs import MODEL_CONFIGS, ClassifierConfig
from .utils import load_data, TuningFold, TrialRecord, hash_ids, Result, TASK
from .classifiers import ClassifierHelper

logger = logging.getLogger(__name__)


def hyperparameter_tuning(
    models: Annotated[list[str], typer.Option(help='List of models to tune', default_factory=lambda: list(MODEL_CONFIGS.keys()))],
    dev_mode: Annotated[bool, typer.Option(help='Run in development mode')] = False,
    num_folds: Annotated[int, typer.Option(help='Number of folds for cross-validation')] = 3,
    random_seed: Annotated[int | None, typer.Option(help='Random seed for cross-validation')] = None,
    num_trials: Annotated[int | None, typer.Option(help='Number of trials for hyperparameter tuning')] = None,
    num_jobs: Annotated[int, typer.Option(help='Number of tuning jobs for parallel processing')] = 1,
    scoring: Annotated[str, typer.Option(help='Scoring metric for hyperparameter tuning')] = 'AveragePrecision',
    decision_threshold: Annotated[float, typer.Option(help='Decision threshold for classification')] = 0.5,
    result_dir: Annotated[Path | None, typer.Option(help='Directory to write tuning results to')] = None,
) -> None:
    """Run all models and find the best hyperparameter setting for each and store results"""
    TASK.dev_mode = dev_mode
    result_dir = result_dir or TASK.tuning_results_path
    logger.info(f'Going to write tuning results to {result_dir}')
    result_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_data(dev=dev_mode)

    x_train: list[str] = train['title'].str.cat(train['text'], sep=' ', na_rep='').tolist()
    y_train: np.ndarray = train['label'].to_numpy()
    x_val: list[str] = val['title'].str.cat(val['text'], sep=' ', na_rep='').tolist()
    y_val: np.ndarray = val['label'].to_numpy()

    train_hash = hash_ids(train['item_id'].tolist())
    val_hash = hash_ids(val['item_id'].tolist())

    for name in models:
        config: Type[ClassifierConfig] = MODEL_CONFIGS[name.upper()]
        key = f'{name}_{train_hash}_{val_hash}'
        result_file = result_dir / f'{key}.json'
        if result_file.exists():
            logger.warning(f'Model {name.upper()} already exists, skipping (key: {key})')
            continue

        logger.info(f'Tuning model {name.upper()}')

        helper = ClassifierHelper(
            config=config,
            tuning_jobs=num_jobs,
            tuning_trials=num_trials,
        )
        start_time = time.time()
        study = helper.tune(
            X_train=x_train, y_train=y_train,
            X_test=x_val, y_test=y_val,
            scoring=scoring,
        )
        tune_time = time.time() - start_time
        best_trial = study.best_trial

        # Refit best params on the full train pool (reproduces the winning trial's fit).
        start_time = time.time()
        model = helper.best_from_study(study, X=x_train, y=y_train.tolist())
        fit_time = time.time() - start_time

        slurm_info = None
        if os.getenv('SLURM_JOB_ID') is not None:
            slurm_info = {
                'job_id': os.getenv('SLURM_JOB_ID'),
                'job_name': os.getenv('SLURM_JOB_NAME'),
                'job_array_task_id': os.getenv('SLURM_ARRAY_TASK_ID'),
                'job_array_task_count': os.getenv('SLURM_ARRAY_TASK_COUNT'),
                'job_nodelist': os.getenv('SLURM_JOB_NODELIST'),
            }

        val_probs = model.predict_proba(x_val)
        trials = [TrialRecord(number=t.number, value=t.value, state=t.state.name) for t in study.trials]

        with open(result_file, 'w') as fp:
            fp.write(
                TuningFold(
                    model=name,
                    params=best_trial.user_attrs['model_params'],
                    scores_self=Result.model_validate(best_trial.user_attrs['scores_self']),
                    val_ids=val['item_id'].tolist(),
                    val_labels=y_val.tolist(),
                    val_probs=val_probs.tolist(),
                    train_hash=train_hash,
                    val_hash=val_hash,
                    trials=trials,
                    tune_time=tune_time,
                    fit_time=fit_time,
                    slurm_info=slurm_info,
                ).model_dump_json(indent=2)
            )
