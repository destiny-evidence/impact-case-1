"""Combine the per-model `learning_curve_<model>.csv` files into one plot + summary table.

Read-only: run locally after the (per-model, possibly cluster-run) curves land. Reuses the plotting
function from `learning_curve` so the combined figure matches the per-model ones.
"""

import glob
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from ic1.classify.inout.utils import TASK
from ic1.classify.inout.learning_curve import plot_learning_curves

logger = logging.getLogger(__name__)


def combine(
    dev_mode: Annotated[bool, typer.Option(help='Read from the testing results dir')] = False,
    metric: Annotated[str, typer.Option(help='Metric to plot')] = 'AveragePrecision',
    results_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    TASK.dev_mode = dev_mode
    results_dir = results_dir or TASK.classifier_results_path

    files = sorted(glob.glob(str(results_dir / 'learning_curve_*.csv')))
    if not files:
        raise FileNotFoundError(f'No learning_curve_*.csv under {results_dir} — run the curves first.')
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    fig = plot_learning_curves(df, metric=metric)
    out = results_dir / 'learning_curve_combined.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')

    # Summary: full-data value + slope over the top two fractions (the "still rising?" read).
    means = df.groupby(['model', 'frac'])[metric].mean().unstack()
    fracs = sorted(df['frac'].unique())
    summary = pd.DataFrame({
        f'{metric}@full': means[fracs[-1]].round(3),
        f'slope_{fracs[-2]}->{fracs[-1]}': (means[fracs[-1]] - means[fracs[-2]]).round(3),
    }).sort_values(f'{metric}@full', ascending=False)
    print(summary.to_string())
    logger.info(f'Wrote {out} ({len(files)} models)')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    typer.run(combine)
