"""Fail if any DVC-tracked file is missing from any configured remote."""

import json
import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-m', 'dvc', *args], capture_output=True, text=True)


def _remotes() -> list[str]:
    # `dvc remote list` prints "name\turl" per remote, with a "(default)" marker line after the default one.
    result = _run('remote', 'list')
    if result.returncode != 0:
        return []
    return [fields[0] for line in result.stdout.splitlines() if (fields := line.split()) and fields[0] != '(default)']


def main() -> int:
    remotes = _remotes()
    if not remotes:
        print('No DVC remotes configured; nothing to check.')
        return 0

    failed = False
    for remote in remotes:
        result = _run('data', 'status', '--not-in-remote', '--json', '-r', remote)
        if result.returncode != 0:
            print(f'Could not reach DVC remote {remote!r}, so cannot verify data is pushed:')
            print(result.stderr.strip())
            failed = True
            continue
        try:
            missing: list[str] = json.loads(result.stdout or '{}').get('not_in_remote', [])
        except json.JSONDecodeError:
            print(f'Unexpected output from `dvc data status` for remote {remote!r}: {result.stdout!r}')
            failed = True
            continue
        if missing:
            failed = True
            print(f'DVC data missing from remote {remote!r}:')
            for path in missing:
                print(f'  {path}')
            print(f'Run `uv run dvc push -r {remote}`, then commit again.')

    if failed:
        print('If a remote is genuinely unreachable and you know this is safe:')
        print('  SKIP=dvc-pushed git commit ...')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
