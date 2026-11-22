import typer

from .tune import hyperparameter_tuning
from .finalise import finalise_models
from .slurm import main as slurm_runner
from .stats import stats
from .compare import compare

app = typer.Typer()
app.command('tune', help='Run hyperparameter tuning and store results')(hyperparameter_tuning)
app.command('finalise', help='Using best setup from tuning trials, train, store, and evaluate the final model')(finalise_models)
app.command('slurm', help='Prepare a slurm script to run the hyper-parameter tuning as a job array')(slurm_runner)
app.command('stats', help='Print statistics about the tuning results')(stats)
app.command('compare', help='Paired comparison of the baseline and crowd-trained models on the shared test set')(compare)

if __name__ == '__main__':
    app()
