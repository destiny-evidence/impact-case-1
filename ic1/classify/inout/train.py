import json
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .models import ClassifierHelper
from .utils import load_data, logger, TuningFold, hash_ids, TASK


def read_tuning_results(source_dir: Path) -> list[TuningFold]:
    logger.info(f'Reading tuning results from {source_dir}')
    results = []
    for file in source_dir.glob('*.json'):
        with open(file) as fp:
            results.append(TuningFold.model_validate_json(fp.read()))
    return results


def results_to_pd(results: list[TuningFold]) -> pd.DataFrame:
    rows = []
    for ri, result in enumerate(results):
        base = result.model_dump()
        base.pop('params')
        scores_self = base.pop('scores_self')
        scores_test = base.pop('scores_test')
        scores_val = base.pop('scores_val')
        rows.append(base | scores_self | {'scores': 'self'})
        rows.append(base | scores_test | {'scores': 'test'})
        for score in scores_val:
            rows.append(base | score | {'scores': 'val', 'result': ri})

    return pd.DataFrame(rows)


def train_model(
    dev_mode: Annotated[bool, typer.Option(help='Run in development mode')] = False,
    tuning_dir: Annotated[Path, typer.Option(help='Directory to write tuning results to')] = TASK.tuning_results_path,
    target_dir: Annotated[Path, typer.Option(help='Directory to write tuning results to')] = TASK.ml_model_path,
):
    """Run all models and find the best hyperparameter setting for each and store results"""
    results = read_tuning_results(tuning_dir)
    df_results = results_to_pd(results)
    logger.info(f'Found {df_results.shape} tuning results')
    df_best = df_results[df_results['scores'] == 'val'].sort_values(by='F1', ascending=False)
    logger.info(f'Checking {df_best.shape[0]:,} for top results')
    threshold = df_best.iloc[0]['threshold']
    best_result = results[df_best.iloc[0]['result']]
    logger.info(f'Best model was {best_result.model} at threshold {threshold} with parameters {best_result.params}')

    train, val, test = load_data(dev=dev_mode)
    helper = ClassifierHelper.from_run(best_result)

    logger.info(f'Training model with {train.shape[0]:,} samples of which {train["label"].sum():,} are includes...')
    start_time = time.time()
    model = helper.train(X=train['text'].tolist(), y=train['label'].tolist())
    train_time = time.time() - start_time
    logger.info(f'Trained model in {train_time:.2f} seconds...')

    logger.info(f'Writing trained model to {target_dir}')
    model.save(target_dir)
    with open(target_dir / 'train_info.json', 'w') as fp:
        json.dump(
            {
                'info': best_result,
                'threshold': threshold,
                'train_hash': hash_ids(train['item_id'].tolist()),
                'val_hash': hash_ids(val['item_id'].tolist()),
                'test_hash': hash_ids(test['item_id'].tolist()),
            },
            fp=fp,
            indent=2,
        )
