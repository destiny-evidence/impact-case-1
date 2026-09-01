"""Learning curves: has performance plateaued at the current training-set size?

For each tuned model (its winning hyperparameters), retrain on increasing *stratified* fractions of
the train pool and evaluate on the fixed validation pool. Averages over several random subsamples
per fraction to get error bands. The headline metric is threshold-free (AveragePrecision / ROC-AUC)
so the curve reflects ranking quality vs data size, not a moving threshold.

Read the slope at 100%: flat -> more data won't help much (representation-limited); still rising ->
the incoming crowd data should help. Cheap for the conventional models; costly for transformers
(one fine-tune per fraction x seed), so pick `--models` accordingly.
"""

import logging
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from ic1.classify.inout.utils import load_data, read_tuning_results, TASK
from ic1.classify.inout.utils.metrics import evaluate
from ic1.classify.inout.classifiers import ClassifierHelper

logger = logging.getLogger(__name__)


def plot_learning_curves(df: pd.DataFrame, metric: str = 'AveragePrecision'):
    """One axes, one line (+/- std band over seeds) per model. Returns the figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in df.groupby('model'):
        s = g.groupby('n_train')[metric].agg(['mean', 'std']).reset_index()
        ax.plot(s['n_train'], s['mean'], '-o', ms=4, label=name)
        ax.fill_between(s['n_train'], s['mean'] - s['std'].fillna(0), s['mean'] + s['std'].fillna(0), alpha=0.15)
    ax.set_xlabel('training examples')
    ax.set_ylabel(f'{metric} (validation)')
    ax.set_title('Learning curves')
    ax.legend(fontsize=7, ncol=2)
    return fig


def _stratified_subsample(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Indices for a class-stratified fraction of y (preserves prevalence)."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in (0, 1):
        pool = np.flatnonzero(y == cls)
        k = max(1, int(round(len(pool) * frac)))
        idx.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def learning_curve(
    dev_mode: Annotated[bool, typer.Option(help='Use the testing dir / fabricated splits')] = False,
    models: Annotated[list[str], typer.Option(help='Models to curve (default: all tuned)')] = None,
    fractions: Annotated[list[float], typer.Option(help='Train fractions to evaluate')] = None,
    n_seeds: Annotated[int, typer.Option(help='Random subsamples per fraction (error bands)')] = 3,
    beta: Annotated[float, typer.Option()] = 1.0,
    tuning_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    TASK.dev_mode = dev_mode
    tuning_dir = tuning_dir or TASK.tuning_results_path
    folds = {f.model: f for f in read_tuning_results(tuning_dir)}
    models = models or sorted(folds)
    fractions = fractions or [0.1, 0.25, 0.5, 0.75, 1.0]

    train, val, _ = load_data(dev=dev_mode)
    X_train = train['title'].str.cat(train['text'], sep=' ', na_rep='').tolist()
    y_train = train['label'].to_numpy()
    X_val = val['title'].str.cat(val['text'], sep=' ', na_rep='').tolist()
    y_val = val['label'].to_numpy()

    rows = []
    for name in models:
        if name not in folds:
            logger.warning(f'No tuning result for {name}, skipping')
            continue
        for frac in fractions:
            for seed in range(n_seeds):
                idx = _stratified_subsample(y_train, frac, seed=1000 * seed + int(frac * 1000))
                helper = ClassifierHelper.from_run(folds[name])
                model = helper.train(X=[X_train[i] for i in idx], y=y_train[idx].tolist())
                m = evaluate(y_val, model.predict_proba(X_val), beta=beta)
                rows.append({
                    'model': name, 'frac': frac, 'n_train': len(idx), 'seed': seed,
                    'AveragePrecision': m.get('AveragePrecision'), 'ROC_AUC': m.get('ROC_AUC'),
                })
                logger.info(f'{name} frac={frac} seed={seed} n={len(idx)} AP={m.get("AveragePrecision"):.3f}')

    df = pd.DataFrame(rows)
    agg = df.groupby(['model', 'frac', 'n_train'])[['AveragePrecision', 'ROC_AUC']].agg(['mean', 'std'])
    print(agg.to_string())

    # Tag outputs by model so parallel (per-model) array tasks don't overwrite each other.
    tag = models[0] if len(models) == 1 else 'all'
    out_dir = TASK.classifier_results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f'learning_curve_{tag}.csv', index=False)

    fig = plot_learning_curves(df)
    fig.savefig(out_dir / f'learning_curve_{tag}.png', dpi=120, bbox_inches='tight')
    logger.info(f'Wrote learning_curve_{tag}.csv + .png to {out_dir}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    typer.run(learning_curve)
