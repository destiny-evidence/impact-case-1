import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from torch import Tensor


def evaluate(
    # expecting 1D array for y_true and y_pred
    y_true: 'np.ndarray | Tensor',
    y_pred: 'np.ndarray | Tensor',
    threshold: float = 0.5,
    beta: float = 0.5
) -> dict[str, Any]:
    y_pred_binary = np.where(y_pred > threshold, 1, 0)

    results = {
        'F1': f1_score(y_true, y_pred_binary, zero_division=0),
        'Fbeta': fbeta_score(y_true, y_pred_binary, beta=beta, zero_division=0),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred_binary),
        'prop_included': float(y_pred_binary.mean()), # share passed on to LLM
        'threshold': threshold,
        'n_samples': y_true.shape[0],
    }

    try:
        results['ROC_AUC'] = roc_auc_score(y_true, y_pred)
        results['AveragePrecision'] = average_precision_score(y_true, y_pred)
    except:  # noqa: E722
        pass

    logger.debug(' | '.join([f'{key}: {score:.1%}' for key, score in results.items()]))
    return results

def posterior_metric_summaries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    beta: float = 1.0,
    num_samples: int = 10000,
    ci_probability: float = 0.95,
    seed: int = 0,
    confusion_prior: float = 0.5,
    prevalence_prior: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Bayesian credible intervals for fixed-threshold binary metrics, via prob_conf_mat.

    Fits a Dirichlet-multinomial posterior over the confusion matrix and summarises the
    positive-class Precision, Recall and F-beta as posterior median + HDI. Complements the
    score-based ranking metrics (ROC-AUC / average precision), which are NOT confusion-matrix
    functionals and cannot be obtained here.

    Scope: test-set sampling uncertainty for this fixed model only -- not model-selection
    uncertainty, nor distribution shift to the full corpus. A mild proper prior (Jeffreys, 0.5)
    is used instead of the library default Haldane (0.0) so the posterior stays well-defined when
    confusion cells are zero at small positive counts.

    Returns {metric: {'median': float, 'hdi_lo': float, 'hdi_hi': float}}.
    """
    from prob_conf_mat import Study
    from prob_conf_mat.stats.summary import summarize_posterior

    study = Study(seed=seed, num_samples=num_samples, ci_probability=ci_probability)
    study.add_experiment(
        experiment_name='final/model',
        y_true=np.asarray(y_true).astype(int),
        y_pred=np.asarray(y_pred).astype(int),
        confusion_prior=confusion_prior,
        prevalence_prior=prevalence_prior,
    )

    metrics = {'Precision': 'precision', 'Recall': 'recall', 'Fbeta': f'fbeta+beta={beta:g}'}
    for metric in metrics.values():
        study.add_metric(metric)

    out: dict[str, dict[str, float]] = {}
    for label, metric in metrics.items():
        res = study.get_metric_samples(metric=metric, experiment_name='final/model', sampling_method='posterior')
        samples = res.values[:, 1] if res.values.ndim > 1 else res.values  # positive class = column 1
        summ = summarize_posterior(samples, ci_probability=ci_probability)
        out[label] = {'median': float(summ.median), 'hdi_lo': float(summ.hdi[0]), 'hdi_hi': float(summ.hdi[1])}
    return out
