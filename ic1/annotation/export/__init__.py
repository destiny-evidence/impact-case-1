import typer

from .crowd_ris import main as crowd_ris
from .export_annotations import main as export_annotations
from .resolve_annotations import main as resolve_annotations

app = typer.Typer()
app.command('resolve', help='Run hyperparameter tuning and store results')(resolve_annotations)
app.command('csv', help='Using best setup from tuning trials, train and store a model')(export_annotations)
app.command('ris', help='Prepare a slurm script to run the hyper-parameter tuning as a job array')(crowd_ris)


if __name__ == '__main__':
    app()
