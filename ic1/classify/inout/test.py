"""Tests the best model on a test dataset. Retrains on all data and persists."""

import typer

from ic1.classify.inout.utils import load_data, get_classifier
from ic1.classify.base import ModelRun, score, TestResult, hash_ids
from ic1.core.config import TASKS, TaskName
from pathlib import Path
import pandas as pd

INOUT = TASKS[TaskName.INOUT]

def selection_score(run: ModelRun) -> float:
    # TODO: Define selection formula, using time and cost too
    return run.f1

def find_best_model(path: Path) -> ModelRun:
    """Read all runs and return the one with the best overall score"""
    runs = [
        ModelRun.model_validate_json(l)
        for l in path.read_text().splitlines()
        if l.strip()
    ]
    if not runs:
        raise FileNotFoundError(f'No runs found in {path} — run train_val.py first')
    return max(runs, key=selection_score)

def main():
    """Find the best model."""
    best_model = find_best_model(INOUT.ml_model_runs_path)
    clf = get_classifier(best_model)
    train, val, test = load_data()

    train_x = train['text'].to_list() + val['text'].to_list()
    train_y = train['label'].to_list() + val['label'].to_list()

    clf.fit(train_x, train_y)

    y_pred_proba = clf.predict_proba(test['text'].to_list())

    scores = score(test['label'], y_pred_proba, best_model.threshold)

    test_result = TestResult(
        model=best_model.model,
        config=best_model.config,
        precision=scores['precision'],
        recall=scores['recall'],
        f1 = scores['f1'],
        threshold=best_model.threshold,
        train_hash=hash_ids(train_x),
        val_hash=hash_ids(test['text'].to_list()),
        fit_time=clf.fit_time,
        selected_run=best_model
    )
    INOUT.ml_test_result_path.write_text(
        test_result.model_dump_json(indent=2)
    )

    clf.save(INOUT.ml_model_path)

    predictions = pd.DataFrame({
        'item_id': test['item_id'].to_list(),
        'score': y_pred_proba,
        'predicted_label': (y_pred_proba >= best_model.threshold).astype(int)
    })
    predictions.to_csv(INOUT.ml_model_predictions_path, index=False)





if __name__ == '__main__':
    typer.run(main)