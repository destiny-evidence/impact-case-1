def main():
    import typer

    import logging
    from rich.logging import RichHandler

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=logging.DEBUG,
        handlers=[RichHandler()],
    )

    from ic1.annotation.export.export_annotations import main as export_annotations
    from ic1.annotation.scheme.import_taxonomy import main as import_taxonomy
    from ic1.classify.inout import app as inout_app

    app = typer.Typer()

    app.command('export-labels', help='Export annotations and resolutions for in/out and taxonomy schemes')(export_annotations)
    app.command('import-taxonomy', help='Import *.ttl as annotation scheme into NACSOS')(import_taxonomy)
    app.add_typer(inout_app, name='classify-inout', help='Inclusion classification model tuning and training')
    app()


if __name__ == '__main__':
    main()
