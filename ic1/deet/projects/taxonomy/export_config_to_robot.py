"""Freeze the winning taxonomy extraction bundle into a taxonomy robot's .configs."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import yaml

PROJECT = Path(__file__).resolve().parent
EXP_DIR = PROJECT / "data-extraction-experiments"

# Reuse the scoreable-concept F1 logic used by the matrix report.
sys.path.insert(0, str(PROJECT))
from matrix_scores import _scores  # noqa: E402


class Metric(str, Enum):
    """Selection metric for picking the winning experiment."""

    micro = "micro"
    macro = "macro"


def _resolve_vocab(raw_path: str, exp: Path) -> Path:
    """Resolve a config vocabulary path (written relative to the project cwd)."""
    for base in (PROJECT, exp):
        candidate = (base / raw_path).resolve()
        if candidate.is_file():
            return candidate
    msg = f"Could not resolve vocabulary path {raw_path!r} against {PROJECT} or {exp}"
    raise typer.BadParameter(msg)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(
    robot: Annotated[Path, typer.Argument(help="Path to the taxonomy robot repo")],
    experiment: Annotated[
        str, typer.Argument(help="Experiment dir name to freeze (default: best-scoring)")
    ]
) -> None:
    """Freeze the winning taxonomy extraction bundle into a robot's .configs/taxonomy/."""
    exp = EXP_DIR / experiment

    config_text = (exp / "config.yaml").read_text()
    config = yaml.safe_load(config_text)

    ttl_src = _resolve_vocab(config["vocabulary_path"], exp)
    map_src = _resolve_vocab(config["vocabulary_mapping_path"], exp)
    csv_src = exp / "prompts_used.csv"


    new_text = re.sub(r"^(vocabulary_path:).*$", rf"\1 {ttl_src.name}", config_text, flags=re.M)
    new_text = re.sub(
        r"^(vocabulary_mapping_path:).*$", rf"\1 {map_src.name}", new_text, flags=re.M
    )

    out_dir = robot.resolve() / ".configs" / "taxonomy"
    provenance = {
        "source_experiment": exp.name,
        "method": config.get("method"),
        "model": config.get("model"),
        "prompts_csv": csv_src.name,
        "votes": config.get("votes"),
        "vocabulary_ttl": ttl_src.name,
        "vocabulary_mapping": map_src.name,
        "impact_case_1_git_sha": _git_sha(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }

    typer.echo(f"Freeze plan -> {out_dir}")
    typer.echo(f"  extraction_config.yaml   (from {exp.name}/config.yaml)")
    typer.echo(f"  {ttl_src.name}   (from {ttl_src})")
    typer.echo(f"  {map_src.name}   (from {map_src})")
    typer.echo(f"  {csv_src.name}   (from {csv_src})")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extraction_config.yaml").write_text(new_text)
    shutil.copy2(ttl_src, out_dir / ttl_src.name)
    shutil.copy2(map_src, out_dir / map_src.name)
    shutil.copy2(csv_src, out_dir / csv_src.name)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    typer.secho("\nDone. Bundle written and provenance recorded.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(main)
