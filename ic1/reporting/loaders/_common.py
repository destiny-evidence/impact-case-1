"""Helpers shared by the per-domain loaders."""

from __future__ import annotations

import difflib
import re

# (.*) not (.+): deet appends "_<label>" with an empty label for base runs, so
# the folder ends in a bare "_" — match that as an empty suffix, not a miss.
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.*)$")


def parse_run_name(name: str) -> tuple[str, str]:
    """Return (sort_key, human label) for a run folder name.

    Dated runs (``YYYY-MM-DD_HH-MM-SS_suffix``) sort by their timestamp; other
    names keep their (number-stripped) name and sort first.
    """
    m = _TS_RE.match(name)
    if m:
        return m.group(1), m.group(2)
    return name, re.sub(r"^\d+_", "", name)


def model_short(model: str) -> str:
    m = model or ""
    if "luna" in m:
        return "luna"
    if "sol" in m:
        return "sol"
    if "kimi" in m.lower() or "k26" in m or "k2.6" in m.lower():
        return "kimi"
    if "deepseek" in m.lower():
        return "deepseek"
    if "opus" in m.lower() or "claude" in m.lower():
        return "opus"
    if "terra" in m:
        return "terra"
    if "4o-mini" in m or "4o_mini" in m:
        return "4o-mini"
    if "MiniLM" in m or "sentence-transformers" in m:
        return "MiniLM"
    return m


def word_churn(before: str, after: str) -> tuple[int, int]:
    """Word-level (added, deleted) between two strings via difflib opcodes."""
    sm = difflib.SequenceMatcher(
        a=(before or "").split(), b=(after or "").split(), autojunk=False
    )
    added = deleted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            deleted += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, deleted
