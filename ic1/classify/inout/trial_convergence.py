"""Did the Optuna search converge, or would more trials help?

Reads the per-model trial trajectories now stored in the tuning JSONs, plots best-so-far vs trial
number, and summarises where each model's best was found. If the best objective is reached late
(high `frac_to_best`) and the objective is still climbing over the final trials (`gain_last_25%` > 0),
more trials would likely help; if it was found early with a long flat tail, the search has converged.
"""

import logging
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from ic1.classify.inout.utils import TASK, read_tuning_results

logger = logging.getLogger(__name__)


def _best_so_far(fold) -> tuple[np.ndarray, np.ndarray] | None:
    vals = sorted((t.number, t.value) for t in fold.trials if t.value is not None)
    if not vals:
        return None
    numbers = np.array([n for n, _ in vals])
    best = np.maximum.accumulate([v for _, v in vals])
    return numbers, best


def trial_convergence(
    dev_mode: Annotated[bool, typer.Option()] = False,
    tuning_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    TASK.dev_mode = dev_mode
    tuning_dir = tuning_dir or TASK.tuning_results_path
    folds = read_tuning_results(tuning_dir)

    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for fold in folds:
        curve = _best_so_far(fold)
        if curve is None:
            logger.warning(f'{fold.model}: no trials stored — re-tune to populate the trajectory')
            continue
        numbers, best = curve
        n = len(numbers)
        final = float(best[-1])
        at_75 = float(best[max(0, int(0.75 * n) - 1)])
        first_best = int(numbers[np.argmax(best >= final)])  # trial that first reached the best
        n_failed = sum(1 for t in fold.trials if t.value is None)
        rows.append({
            'model': fold.model,
            'n_trials': n,
            'n_failed': n_failed,
            'best': round(final, 3),
            'best_at_trial': first_best,
            'frac_to_best': round(first_best / max(1, int(numbers[-1])), 2),
            'gain_last_25%': round(final - at_75, 3),
        })
        ax.plot(numbers, best, label=fold.model)

    if not rows:
        logger.warning('No trajectories found. The current JSONs predate the `trials` field — re-run tuning.')
        return

    ax.set_xlabel('trial')
    ax.set_ylabel('best-so-far objective')
    ax.set_title('Optuna convergence (best-so-far)')
    ax.legend(fontsize=7, ncol=2)
    out = tuning_dir / 'trial_convergence.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')

    print(pd.DataFrame(rows).sort_values('best', ascending=False).to_string(index=False))
    logger.info(f'Wrote {out}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    typer.run(trial_convergence)
