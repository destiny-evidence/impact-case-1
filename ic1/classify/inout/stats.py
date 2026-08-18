import logging
from pathlib import Path
from typing import Annotated

import typer

from .utils import TASK, read_tuning_results, results_to_pd, threshold_scores

logger = logging.getLogger(__name__)


def stats(
    tuning_dir: Annotated[Path, typer.Option(help='Directory to write tuning results to')] = TASK.tuning_results_path,
):
    results = read_tuning_results(tuning_dir)
    df_results = results_to_pd(results)
    logger.info(f'Found {df_results.shape} tuning results')
    df_best = df_results[df_results['scores'] == 'val'].sort_values(by='F1', ascending=False)
    logger.info(f'Checking {df_best.shape[0]:,} for top results')

    threshold = df_best.iloc[0]['threshold']
    best_result = results[df_best.iloc[0]['result']]
    logger.info(f'Best model run was {best_result.model} at threshold {threshold} with parameters {best_result.params}')
    logger.info(f'  > Self: {best_result.scores_self}')
    for score in threshold_scores(best_result.val_labels, best_result.val_probs):
        logger.info(f'  > val: {score}')

    logger.info('Results on [val] by model:')
    print(
        df_results[df_results['scores'] == 'val'].groupby(['model', 'threshold'])[['Precision', 'Recall', 'F1', 'Accuracy', 'ROC_AUC']].describe()
        # .sort_values(by=('F1', 'mean'), ascending=False)
    )

    # >>> df_results.columns
    # Index(['tune_time', 'fit_time', 'model', 'train_hash', 'val_hash',
    #        'slurm_info', 'timestamp', 'threshold', 'n_samples', 'Precision',
    #        'Recall', 'F1', 'Accuracy', 'Fbeta', 'prop_included', 'ROC_AUC',
    #        'AveragePrecision', 'scores', 'result'],
    # >>> df_results['scores'].unique()
    # ['self', 'val']


if __name__ == '__main__':
    typer.run(stats)
