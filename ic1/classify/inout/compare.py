"""Paired comparison of the baseline and crowd-trained ML models on the shared test set.

For each selected model (filtering, ml_only) both versions are scored at their own validation-selected
threshold on the same test items. Differences (crowd - baseline) come with percentile intervals from a
bootstrap that resamples test items once and scores both models on the same resample.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import TASK

logger = logging.getLogger(__name__)

CRITERIA = ('filtering', 'ml_only')
METRICS = ('Precision', 'Recall', 'Fbeta', 'prop_included', 'AveragePrecision', 'ROC_AUC')


def _load(model_dir: Path, criterion: str) -> tuple[pd.DataFrame, dict]:
    run_dir = model_dir / criterion
    preds = pd.read_csv(run_dir / 'test_predictions.csv')
    info = json.loads((run_dir / 'train_info.json').read_text())
    return preds, info


def _scores(y: np.ndarray, prob: np.ndarray, threshold: float, beta: float) -> dict[str, float]:
    pred = prob > threshold
    tp = float(np.sum(pred & (y == 1)))
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / (y == 1).sum()
    denom = beta**2 * precision + recall
    return {
        'Precision': precision,
        'Recall': recall,
        'Fbeta': (1 + beta**2) * precision * recall / denom if denom else 0.0,
        'prop_included': float(pred.mean()),
        'AveragePrecision': float(average_precision_score(y, prob)),
        'ROC_AUC': float(roc_auc_score(y, prob)),
    }


def compare_criterion(base: tuple[pd.DataFrame, dict], crowd: tuple[pd.DataFrame, dict], n_boot: int, seed: int) -> dict:
    (b, b_info), (c, c_info) = base, crowd
    if b_info['test_hash'] != c_info['test_hash']:
        raise ValueError('Baseline and crowd models were evaluated on different test sets')
    b, c = b.set_index('item_id'), c.set_index('item_id')
    if set(b.index) != set(c.index):
        raise ValueError('Test predictions cover different items')
    c = c.loc[b.index]
    if not (b['y_true'].to_numpy() == c['y_true'].to_numpy()).all():
        raise ValueError('Test labels differ between the two prediction files')
    beta = float(b_info['beta'])
    if beta != float(c_info['beta']):
        raise ValueError('Models were selected with different beta')

    y = b['y_true'].to_numpy().astype(int)
    pb, pc = b['y_prob'].to_numpy(), c['y_prob'].to_numpy()
    tb, tc = float(b_info['threshold']), float(c_info['threshold'])
    point_b, point_c = _scores(y, pb, tb, beta), _scores(y, pc, tc, beta)

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        sb, sc = _scores(y[idx], pb[idx], tb, beta), _scores(y[idx], pc[idx], tc, beta)
        for m in METRICS:
            deltas[m].append(sc[m] - sb[m])

    out = {'n_test': n, 'n_positive': int(y.sum()), 'beta': beta, 'threshold_base': tb, 'threshold_crowd': tc, 'metrics': {}}
    for m in METRICS:
        d = np.asarray(deltas[m])
        out['metrics'][m] = {
            'base': point_b[m],
            'crowd': point_c[m],
            'delta': point_c[m] - point_b[m],
            'delta_lo': float(np.percentile(d, 2.5)),
            'delta_hi': float(np.percentile(d, 97.5)),
            'n_boot': len(d),
        }
    return out


def compare(
    base_dir: Annotated[Path | None, typer.Option(help='Baseline model directory (default: the non-crowd production models)')] = None,
    crowd_dir: Annotated[Path | None, typer.Option(help='Crowd-trained model directory (default: the crowd production models)')] = None,
    out: Annotated[Path | None, typer.Option(help='Where to write the comparison JSON (default: next to the crowd models)')] = None,
    n_boot: Annotated[int, typer.Option(help='Bootstrap resamples of the test set')] = 5000,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Compare baseline and crowd-trained selected models on the shared test set."""
    base_task, crowd_task = replace(TASK, crowd=False, dev_mode=False), replace(TASK, crowd=True, dev_mode=False)
    base_dir = base_dir or base_task.ml_model_path
    crowd_dir = crowd_dir or crowd_task.ml_model_path
    out = out or crowd_dir / 'comparison_vs_baseline.json'

    result = {}
    for criterion in CRITERIA:
        result[criterion] = compare_criterion(_load(base_dir, criterion), _load(crowd_dir, criterion), n_boot=n_boot, seed=seed)
        r = result[criterion]
        print(
            f'\n== {criterion} (n={r["n_test"]}, positives={r["n_positive"]}, beta={r["beta"]:g}; thresholds {r["threshold_base"]:.2f} -> {r["threshold_crowd"]:.2f}) =='
        )
        print(f'{"metric":18s}{"baseline":>10s}{"crowd":>10s}{"delta":>10s}   95% interval of delta')
        for m, v in r['metrics'].items():
            print(f'{m:18s}{v["base"]:10.3f}{v["crowd"]:10.3f}{v["delta"]:+10.3f}   [{v["delta_lo"]:+.3f}, {v["delta_hi"]:+.3f}]')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info(f'Wrote {out}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    typer.run(compare)
