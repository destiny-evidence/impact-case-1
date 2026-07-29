import typer

from ic1.core.config import TaskName
from .inout import app as inout_app

app = typer.Typer()

app.add_typer(inout_app, name=TaskName.INOUT)


if __name__ == '__main__':
    app()
