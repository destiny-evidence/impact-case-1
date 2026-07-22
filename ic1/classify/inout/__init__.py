import typer

from .tune import hyperparameter_tuning
from .train import train_model
from .slurm import main as slurm_runner

app = typer.Typer()
app.command('tune', help='Run hyperparameter tuning and store results')(hyperparameter_tuning)
app.command('train', help='Using best setup from tuning trials, train and store a model')(train_model)
app.command('slurm', help='Prepare a slurm script to run the hyper-parameter tuning as a job array')(slurm_runner)

if __name__ == '__main__':
    app()
