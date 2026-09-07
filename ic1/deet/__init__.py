import typer
from ic1.deet import create_deet_project

app = typer.Typer()

app.command('create-project')(create_deet_project.main)