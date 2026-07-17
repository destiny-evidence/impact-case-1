"""Train and validate a list of models and model configurations."""

import typer
from ic1.classify.inout.sklearn_configs import CONFIGS as SKLEARN_CONFIGS
from ic1.classify.inout.transformer_configs import CONFIGS as TRANSFORMER_CONFIGS
from ic1.classify.inout.utils import load_data
from ic1.classify.base import ModelRun
from ic1.core.config import TASKS, TaskName
from rich import print

INOUT = TASKS[TaskName.INOUT]

ALL_CONFIGS = SKLEARN_CONFIGS + TRANSFORMER_CONFIGS


def append_run(run: ModelRun, dev_mode: bool) -> None:
    INOUT.dev_mode = dev_mode
    INOUT.classifier_results_path.mkdir(parents=True, exist_ok=True)
    with INOUT.ml_model_runs_path.open('a', encoding='utf-8') as f:
        f.write(run.model_dump_json() + '\n')


def main(dev_mode=True) -> None:

    train, val, _ = load_data()
    if val.empty:
        print('val split empty!')
        return
    for clf in ALL_CONFIGS:
        print(f'[bold]tuning[/bold] {clf.name}...')
        model_runs = clf.tune(x_train=train['text'].tolist(), y_train=train['label'].tolist(), x_val=val['text'].tolist(), y_val=val['label'].tolist())
        for model_run in model_runs:
            append_run(model_run, dev_mode=dev_mode)


if __name__ == '__main__':
    typer.run(main)
