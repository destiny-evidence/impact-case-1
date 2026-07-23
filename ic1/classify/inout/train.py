import json
import logging
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from .utils import load_data, hash_ids, TASK, read_tuning_results, results_to_pd
from .classifiers import ClassifierHelper

logger = logging.getLogger(__name__)


def train_model(
    dev_mode: Annotated[bool, typer.Option(help='Run in development mode')] = False,
    use_val: Annotated[bool, typer.Option(help='Use validation set for training')] = False,
    use_test: Annotated[bool, typer.Option(help='Use test set for training')] = False,
    tuning_dir: Annotated[Path, typer.Option(help='Directory to write tuning results to')] = TASK.tuning_results_path,
    target_dir: Annotated[Path, typer.Option(help='Directory to write tuning results to')] = TASK.ml_model_path,
) -> None:
    """Run all models and find the best hyperparameter setting for each and store results"""
    results = read_tuning_results(tuning_dir)
    df_results = results_to_pd(results)
    logger.info(f'Found {df_results.shape} tuning results')
    df_best = df_results[df_results['scores'] == 'val'].sort_values(by='F1', ascending=False)
    logger.info(f'Checking {df_best.shape[0]:,} for top results')
    threshold = df_best.iloc[0]['threshold']
    best_result = results[df_best.iloc[0]['result']]
    logger.info(f'Best model was {best_result.model} at threshold {threshold} with parameters {best_result.params}')

    helper = ClassifierHelper.from_run(best_result)

    train, val, test = load_data(dev=dev_mode)
    parts = [train]
    if use_val:
        parts.append(val)
    if use_test:
        parts.append(test)
    data = pd.concat(parts)

    logger.info(f'Training model with {data.shape[0]:,} samples of which {data["label"].sum():,} are includes...')
    start_time = time.time()
    model = helper.train(
        X=data['title'].str.cat(data['text'], sep=' ', na_rep='').tolist(),
        y=data['label'].tolist(),
    )
    train_time = time.time() - start_time
    logger.info(f'Trained model in {train_time:.2f} seconds...')

    logger.info(f'Writing trained model to {target_dir}')
    model.save(target_dir)
    with open(target_dir / 'train_info.json', 'w') as fp:
        json.dump(
            {
                'info': best_result,
                'threshold': threshold,
                'data_hash': hash_ids(data['item_id'].tolist()),
                'train_hash': hash_ids(train['item_id'].tolist()),
                'val_hash': hash_ids(val['item_id'].tolist()),
                'test_hash': hash_ids(test['item_id'].tolist()),
            },
            fp=fp,
            indent=2,
        )
