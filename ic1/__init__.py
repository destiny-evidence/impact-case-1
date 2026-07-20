import typer

from ic1.annotation.export.export_annotations import main as export_annotations
from ic1.annotation.scheme.import_taxonomy import main as import_taxonomy


def main():
    app = typer.Typer()

    app.command('export-labels', help='Export annotations and resolutions for in/out and taxonomy schemes')(export_annotations)
    app.command('import-taxonomy', help='Import *.ttl as annotation scheme into NACSOS')(import_taxonomy)

    app()


if __name__ == '__main__':
    main()
