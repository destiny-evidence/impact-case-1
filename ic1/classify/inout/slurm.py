import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from ic1.core.config import settings
from .configs import MODEL_CONFIGS
from .configs._abc import _SklearnClassifierConfig, _HuggingfaceClassifierConfig
from .utils import TASK, ensure_offline_models

logger = logging.getLogger(__name__)


def _write_file(
    target: Path,
    models: list[str],
    sbatch_args: dict[str, Any],
    script_args: dict[str, Any],
    venv_path: Path,
    command: str = 'ic1 classify-inout tune',
) -> None:
    # Write slurm batch file (an array job, one task per model, running `command`)
    # For information on array jobs, see: https://hpcdocs.hpc.arizona.edu/running_jobs/batch_jobs/array_jobs/

    sbatch = [f'#SBATCH --{key}={value}' for key, value in sbatch_args.items()]
    script_params = []
    for key, value in script_args.items():
        if value is None:
            continue
        if type(value) is str:
            script_params.append(f'--{key}="{value}"')
        elif type(value) is bool and value:
            script_params.append(f'--{key}')
        elif type(value) is bool and not value:
            script_params.append(f'--no-{key}')  # FIXME: Might fail if there's a custom renaming of bool options
        elif type(value) is list:
            for vi in value:
                script_params.append(f'--{key}="{vi}"')  # FIXME: This just assumes that list values are always strings
        else:
            script_params.append(f'--{key}={value}')

    with open(target, 'w') as slurm_file:
        slurm_file.write(f"""#!/bin/bash

{'\n'.join(sbatch)}
#SBATCH --oversubscribe  # use non-utilized GPUs on busy nodes
#SBATCH --mail-type=END,FAIL  # 'NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'
#SBATCH --array=1-{(len(models))}

# Set this to exit the script when an error occurs
set -e
# Set this to print commands before executing
set -o xtrace

# Set up python environment
module load anaconda
module load cuda

# Python env vars
export PYTHONPATH=$PYTHONPATH:{os.getcwd()}
export PYTHONUNBUFFERED=1

# Environment variables for script
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT={venv_path}

echo "Using python from $(which python)"
echo "Python version is $(python --version)"

MODELS=("{'" "'.join(models)}")

job=$(($SLURM_ARRAY_TASK_ID - 1))
model_idx=$(($job % {len(models)}))

echo "array_task_id" $SLURM_ARRAY_TASK_ID " --> job" $job
echo "model_idx" $model_idx
echo "model" "${{MODELS[$model_idx]}}"

uv run --extra classify --link-mode=copy {command} --models="${{MODELS[$model_idx]}}" {' '.join(script_params)}

echo "Job done."
    """)


def _write_finalise_file(
    target: Path,
    sbatch_args: dict[str, Any],
    script_args: dict[str, Any],
    venv_path: Path,
) -> None:
    """Write a single (non-array) slurm job that runs `finalise` after tuning completes."""
    sbatch = [f'#SBATCH --{key}={value}' for key, value in sbatch_args.items()]
    script_params = []
    for key, value in script_args.items():
        if value is None:
            continue
        if type(value) is bool:
            script_params.append(f'--{key}' if value else f'--no-{key}')
        elif type(value) is str:
            script_params.append(f'--{key}="{value}"')
        else:
            script_params.append(f'--{key}={value}')

    with open(target, 'w') as slurm_file:
        slurm_file.write(f"""#!/bin/bash

{'\n'.join(sbatch)}
#SBATCH --mail-type=END,FAIL

# Set this to exit the script when an error occurs
set -e
# Set this to print commands before executing
set -o xtrace

# Set up python environment
module load anaconda
module load cuda

# Python env vars
export PYTHONPATH=$PYTHONPATH:{os.getcwd()}
export PYTHONUNBUFFERED=1

# Environment variables for script
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT={venv_path}

uv run --extra classify --link-mode=copy ic1 classify-inout finalise {' '.join(script_params)}

echo "Finalise done."
    """)


def main(
    slurm_user: Annotated[str, typer.Option(help='email address to notify when done')],
    models: Annotated[list[str], typer.Option(help='List of models to tune', default_factory=lambda: list(MODEL_CONFIGS.keys()))],
    schedule_jobs: Annotated[bool, typer.Option('--submit/--script-only', help='Schedule jobs on slurm')] = False,
    slurm_log: Annotated[Path, typer.Option(help='Directory to write slurm logs to')] = settings.LOGGING_DIR,
    slurm_hours: Annotated[int, typer.Option(help='Number of hours to allocate per slurm job in array')] = 5,
    slurm_gpu_qos: Annotated[
        str,
        typer.Option(
            help='Slurm partition QoS to use for GPU tasks [gpushort, gpumedium, ...]\n'
            'GPU QoS has different MaxJobsPU with different priority, see `$ sacctmgr show qos`'
        ),
    ] = 'gpumedium',
    slurm_cpu_qos: Annotated[str, typer.Option(help='Slurm partition QoS to use for GPU tasks [short, ...]')] = 'short',
    dev_mode: Annotated[bool, typer.Option(help='Run in development mode')] = False,
    num_folds: Annotated[int, typer.Option(help='Number of folds for cross-validation')] = 3,
    random_seed: Annotated[int | None, typer.Option(help='Random seed for cross-validation')] = None,
    num_trials_gpu: Annotated[int | None, typer.Option(help='Number of trials for GPU hyperparameter tuning')] = None,
    num_trials_cpu: Annotated[int | None, typer.Option(help='Number of trials for CPU hyperparameter tuning')] = None,
    num_jobs: Annotated[int, typer.Option(help='Number of tuning jobs for parallel processing')] = 1,
    scoring: Annotated[str, typer.Option(help='Scoring metric for hyperparameter tuning')] = 'F1',
    decision_threshold: Annotated[float, typer.Option(help='Decision threshold for classification')] = 0.5,
    result_dir: Annotated[Path | None, typer.Option(help='Directory to write tuning results to')] = None,
    run_finalise: Annotated[bool, typer.Option('--finalise/--no-finalise', help='Append a finalise job that runs after the tuning arrays complete')] = True,
    recall_floor: Annotated[float, typer.Option(help='Min validation recall the filtering model must clear (finalise)')] = 0.95,
    beta: Annotated[float, typer.Option(help='Beta for F-beta selection of the ML-only model (finalise)')] = 1.0,
    run_learning_curve: Annotated[bool, typer.Option('--learning-curve/--no-learning-curve', help='Append learning-curve jobs (per model) after tuning')] = False,
    lc_n_seeds: Annotated[int, typer.Option(help='Learning-curve subsamples per fraction')] = 3,
    lc_fractions: Annotated[list[float], typer.Option(help='Learning-curve train fractions (default: 0.1..1.0)')] = None,
) -> None:
    logger.info('Preparing slurm script and submitting job!')

    # Route outputs to the testing/ dir in dev mode, production dir otherwise.
    TASK.dev_mode = dev_mode
    result_dir = result_dir or TASK.tuning_results_path

    # Ensure directories are ready
    venv_path = Path(sys.executable).parent.parent.resolve()
    slurm_log.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    configs_cpu = [MODEL_CONFIGS[model.upper()] for model in models if issubclass(MODEL_CONFIGS[model.upper()], _SklearnClassifierConfig)]
    configs_gpu = [MODEL_CONFIGS[model.upper()] for model in models if issubclass(MODEL_CONFIGS[model.upper()], _HuggingfaceClassifierConfig)]
    logger.info(f'Found {len(configs_cpu)} CPU model configs and {len(configs_gpu)} GPU model configs')

    # Ensure models are downloaded
    logger.info('Making sure all models are available offline!')
    ensure_offline_models([config.model_name for config in configs_gpu])  # type: ignore[union-attr]

    script_args = {
        'dev-mode': dev_mode,
        'result-dir': result_dir,
        'num-folds': num_folds,
        'random-seed': random_seed,
        'num-jobs': num_jobs,
        'scoring': scoring,
        'decision-threshold': decision_threshold,
    }

    sbatch_args = {
        'time': f'{slurm_hours:0>2}:00:00',
        'nodes': '1',
        'mem': '8G',
        'mail-user': f'"{slurm_user}"',
        'output': f'{slurm_log}/%A_%a.out',
        'error': f'{slurm_log}/%A_%a.err',
        'chdir': os.getcwd(),
    }

    logger.info(f'Writing CPU slurm file for {[config.name for config in configs_cpu]}')
    _write_file(
        target=Path('classify.cpu.slurm'),
        models=[config.name for config in configs_cpu],
        script_args=script_args | {'num-trials': num_trials_cpu},
        venv_path=venv_path,
        sbatch_args=sbatch_args
        | {
            'cpus-per-task': 12,
            'partition': 'standard',
            'qos': slurm_cpu_qos,
        },
    )

    logger.info(f'Writing GPU slurm file for {[config.name for config in configs_gpu]}')
    _write_file(
        target=Path('classify.gpu.slurm'),
        models=[config.name for config in configs_gpu],
        script_args=script_args | {'num-trials': num_trials_gpu},
        venv_path=venv_path,
        sbatch_args=sbatch_args
        | {
            'gres': 'gpu:1',  # number of GPUs
            'partition': 'gpu',
            'qos': slurm_gpu_qos,
            'cpus-per-task': 5,
        },
    )

    if run_finalise:
        # If any transformer models are in play, the winning model might be a transformer, so
        # finalise needs a GPU to refit it; otherwise a CPU node suffices.
        finalise_sbatch = sbatch_args | {
            'output': f'{slurm_log}/finalise_%j.out',
            'error': f'{slurm_log}/finalise_%j.err',
        } | (
            {'gres': 'gpu:1', 'partition': 'gpu', 'qos': slurm_gpu_qos, 'cpus-per-task': 5}
            if configs_gpu
            else {'cpus-per-task': 12, 'partition': 'standard', 'qos': slurm_cpu_qos}
        )
        logger.info('Writing finalise slurm file')
        _write_finalise_file(
            target=Path('classify.finalise.slurm'),
            sbatch_args=finalise_sbatch,
            script_args={
                'dev-mode': dev_mode,
                'tuning-dir': result_dir,
                'recall-floor': recall_floor,
                'beta': beta,
            },
            venv_path=venv_path,
        )

    if run_learning_curve:
        lc_args = {'dev-mode': dev_mode, 'tuning-dir': result_dir, 'n-seeds': lc_n_seeds, 'fractions': lc_fractions}
        lc_command = 'python ic1/classify/inout/learning_curve.py'
        lc_log = {'output': f'{slurm_log}/lc_%A_%a.out', 'error': f'{slurm_log}/lc_%A_%a.err'}
        if configs_cpu:
            logger.info('Writing learning-curve CPU slurm file')
            _write_file(
                target=Path('classify.lc.cpu.slurm'),
                models=[config.name for config in configs_cpu],
                script_args=lc_args,
                venv_path=venv_path,
                command=lc_command,
                sbatch_args=sbatch_args | lc_log | {'cpus-per-task': 12, 'partition': 'standard', 'qos': slurm_cpu_qos},
            )
        if configs_gpu:
            logger.info('Writing learning-curve GPU slurm file')
            _write_file(
                target=Path('classify.lc.gpu.slurm'),
                models=[config.name for config in configs_gpu],
                script_args=lc_args,
                venv_path=venv_path,
                command=lc_command,
                sbatch_args=sbatch_args | lc_log | {'gres': 'gpu:1', 'partition': 'gpu', 'qos': slurm_gpu_qos, 'cpus-per-task': 5},
            )

    if schedule_jobs:
        logger.info('Scheduling jobs to slurm queue')
        dep_ids: list[str] = []
        for target, configs in [('classify.gpu.slurm', configs_gpu), ('classify.cpu.slurm', configs_cpu)]:
            if not configs:
                continue
            res = subprocess.run(['sbatch', '--parsable', target], capture_output=True, text=True, check=True)
            job_id = res.stdout.strip().split(';')[0]
            dep_ids.append(job_id)
            logger.info(f'Submitted {target} as job {job_id}')

        # Both finalise and the learning curve need the tuned params, so they depend on the
        # tuning arrays (afterany: run once tuning finishes, regardless of per-model failures).
        dep = ['--dependency=afterany:' + ':'.join(dep_ids)] if dep_ids else []

        if run_finalise:
            subprocess.run(['sbatch', *dep, 'classify.finalise.slurm'], check=True)
            logger.info(f'Submitted finalise{" depending on " + ":".join(dep_ids) if dep_ids else ""}')

        if run_learning_curve:
            for target, configs in [('classify.lc.gpu.slurm', configs_gpu), ('classify.lc.cpu.slurm', configs_cpu)]:
                if not configs:
                    continue
                subprocess.run(['sbatch', *dep, target], check=True)
                logger.info(f'Submitted {target}{" depending on tuning" if dep_ids else ""}')
