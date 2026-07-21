import typer

from .tune import hyperparameter_tuning
from .train import train_model

app = typer.Typer()
app.command('tune', help='Run hyperparameter tuning and store results')(hyperparameter_tuning)
app.command('train', help='Select best setup from tuning trials and train and store a model')(train_model)

if __name__ == '__main__':
    app()
