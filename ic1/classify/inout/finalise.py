import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from .utils import load_data, hash_ids, TASK, read_tuning_results
from .classifiers import ClassifierHelper
from ic1.classify.inout.utils.metrics import evaluate, posterior_metric_summaries

logger = logging.getLogger(__name__)


def _threshold_grid(step: float) -> np.ndarray:
    """Candidate decision thresholds: a linear grid from 0.02, preceded by a geometric range down to 1e-5.

    Fine-tuned transformers put a tenth or more of the positives at probabilities below 0.02, so a linear grid
    alone cannot reach a high recall floor for them.
    """
    low = np.geomspace(1e-5, 0.02, 31)[:-1]
    return np.unique(np.concatenate([low, np.round(np.arange(0.02, 0.99, step), 2)]))


def _select_filtering(val: pd.DataFrame, recall_floor: float) -> pd.Series:
    """Pick the filtering model: the config that forwards the fewest documents to the LLM
    (lowest `onward_fraction`) while still catching at least `recall_floor` of the includes
    on validation. Ties broken by recall, so among equally cheap configs we keep the safest.
    """
    eligible = val[val['Recall'] >= recall_floor]
    if eligible.empty:
        raise ValueError(f'No tuned config reaches recall >= {recall_floor} on validation; lower --recall-floor or tune more models.')
    return eligible.sort_values(['prop_included', 'Recall'], ascending=[True, False]).iloc[0]


def finalise_models(
    dev_mode: Annotated[bool, typer.Option(help='Fabricate splits from train; never read real test')] = False,
    crowd: Annotated[bool, typer.Option(help='Also train on the crowd-labelled items (train split only); outputs go to the *_crowd directories')] = False,
    beta: Annotated[float, typer.Option(help='Beta for F-beta selection of the ML-only model')] = 1.0,
    recall_floor: Annotated[float, typer.Option(help='Min validation recall the filtering model must clear')] = 0.95,
    n_boot: Annotated[int, typer.Option(help='Bootstrap resamples for test-set CIs')] = 1000,
    tuning_dir: Annotated[Path | None, typer.Option()] = None,
    target_dir: Annotated[Path | None, typer.Option()] = None,
    threshold_step: Annotated[float, typer.Option()] = 0.01,
) -> None:
    """Select the ML-only and filtering models on validation, fit each on train+val,
    then evaluate both once on the sealed test set with bootstrap CIs and cost proxy."""
    TASK.dev_mode = dev_mode
    TASK.crowd = crowd
    tuning_dir = tuning_dir or TASK.tuning_results_path
    target_dir = target_dir or TASK.ml_model_path

    results = read_tuning_results(tuning_dir)
    grid = _threshold_grid(threshold_step)
    rows = []
    for i, r in enumerate(results):
        yv, pv = np.asarray(r.val_labels), np.asarray(r.val_probs)
        for t in grid:
            m = evaluate(yv, pv, threshold=float(t), beta=beta)
            rows.append({'result': i, 'model': r.model, 'threshold': float(t), 'Fbeta': m['Fbeta'], 'Recall': m['Recall'], 'prop_included': m['prop_included']})
    cand = pd.DataFrame(rows)

    selections = {
        'ml_only': cand.sort_values('Fbeta', ascending=False).iloc[0],
        'filtering': _select_filtering(cand, recall_floor),
    }

    train, val_df, test = load_data(dev=dev_mode)
    fit_df = pd.concat([train, val_df])  # lock config, then use all non-test data
    X_fit = fit_df['title'].str.cat(fit_df['text'], sep=' ', na_rep='').tolist()
    X_test = test['title'].str.cat(test['text'], sep=' ', na_rep='').tolist()
    y_test = test['label'].to_numpy()

    for name, sel in selections.items():
        try:
            helper = ClassifierHelper.from_run(results[int(sel['result'])])
            model = helper.train(X=X_fit, y=fit_df['label'].tolist())
            y_prob = model.predict_proba(X_test)

            threshold = float(sel['threshold'])
            y_pred = (y_prob > threshold).astype(int)
            point = evaluate(y_test, y_prob, threshold=threshold, beta=beta)
            hdi = posterior_metric_summaries(y_test, y_pred, beta=beta, seed=42)

            run = results[int(sel['result'])]
            out = target_dir / name
            out.mkdir(parents=True, exist_ok=True)
            # Write metrics/provenance before the (large, failure-prone) model save, so a save
            # failure doesn't lose the scores.
            (out / 'test.json').write_text(json.dumps({'point': point, 'hdi': hdi, 'selection': sel.to_dict()}, indent=2, default=str))
            (out / 'train_info.json').write_text(
                json.dumps(
                    {
                        'criterion': name,
                        'model': run.model,
                        'params': run.params,
                        'threshold': threshold,
                        'beta': beta,
                        'crowd': crowd,
                        'n_fit': len(fit_df),
                        'fit_hash': hash_ids(fit_df['item_id'].tolist()),
                        'train_hash': hash_ids(train['item_id'].tolist()),
                        'val_hash': hash_ids(val_df['item_id'].tolist()),
                        'test_hash': hash_ids(test['item_id'].tolist()),
                    },
                    indent=2,
                    default=str,
                )
            )
            pd.DataFrame({'item_id': test['item_id'], 'y_true': y_test, 'y_prob': y_prob}).to_csv(out / 'test_predictions.csv', index=False)
            model.save(out / 'weights')
            logger.info(f'Wrote {name} model + results to {out}')
        except Exception:
            logger.exception(f'Finalising {name} model failed; continuing with remaining models')
