def main():
    import logging

    import typer
    from typer._click import Context
    from typer.core import TyperGroup
    from typer.main import get_command
    from rich.logging import RichHandler

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=logging.DEBUG,
        handlers=[RichHandler()],
    )
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    from ic1.annotation.export.export_annotations import main as export_annotations
    from ic1.annotation.scheme.import_taxonomy import main as import_taxonomy
    from ic1.classify.inout import app as inout_app
    from ic1.evaluation_splits import create_split

    app = typer.Typer()

    app.command('export', help='Export annotations and resolutions for in/out and taxonomy schemes')(export_annotations)
    app.command('split-data', help='Split data into train, validation, and test sets')(create_split)
    app.command('import-taxonomy', help='Import *.ttl as annotation scheme into NACSOS')(import_taxonomy)
    app.add_typer(inout_app, name='classify-inout', help='Inclusion classification model tuning and training')

    def tree_command(ctx: typer.Context):
        """Show a tree view of all commands and sub-apps."""
        click_root = get_command(app)  # resolve Typer -> Click command tree
        print('Tree of available commands:')
        root_context = Context(click_root, parent=ctx.parent)
        _print_tree(click_root, root_context)

    def _print_tree(group: TyperGroup, context: Context, prefix: str = ''):
        # Typer groups added with add_typer() resolve their commands lazily, so
        # their ``commands`` mapping may be empty.  Use Click's public lookup
        # API to include those commands as well as directly registered ones.
        names = group.list_commands(context)
        for i, name in enumerate(names):
            cmd = group.get_command(context, name)
            if cmd is None:
                continue

            is_last = i == len(names) - 1
            connector = '└── ' if is_last else '├── '
            child_prefix = prefix + ('    ' if is_last else '│   ')

            print(f'{prefix}{connector}{name}')

            help_text = cmd.get_short_help_str() or (cmd.help or '')
            if help_text:
                print(f'{child_prefix}({help_text})')

            if isinstance(cmd, TyperGroup):
                child_context = Context(cmd, info_name=name, parent=context)
                _print_tree(cmd, child_context, child_prefix)

    app.command('help', help='Show tree view of all commands and sub-apps')(tree_command)

    app()


if __name__ == '__main__':
    main()
