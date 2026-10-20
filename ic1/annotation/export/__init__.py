import typer

from .crowd_ris import main as crowd_ris
from .export_annotations import main as export_annotations

app = typer.Typer()
app.command(
    'csv',
    help='Export the annotations of a scheme (in/out or taxonomy) as long-format CSVs, one row per item and annotator, '
    'with sensitive and pseudonymised shareable versions, each with a resolved-only subset.',
)(export_annotations)
app.command(
    'ris',
    help='Write a random sample of in/out records to RIS: either resolved annotated items or unseen items '
    'without an in/out assignment.',
)(crowd_ris)


if __name__ == '__main__':
    app()
