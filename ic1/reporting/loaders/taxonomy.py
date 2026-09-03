"""Load taxonomy data-extraction experiments into a tidy run-level DataFrame.

Each experiment folder holds a snapshotted ``config.yaml`` (its method/model/
votes/vocabulary identity), a ``goldstandard_llm_comparison.csv`` (per
document x concept human-vs-LLM grid), and ``extraction_metadata.json`` (tokens
and cost). We reduce each to a single row.

Scoring mirrors ``matrix_scores.py``: scores are over SCOREABLE concepts only
(>=1 gold positive). Zero-gold concepts are unevaluable and would distort both
micro (extra FP) and macro (forced zeros), so they are excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

from ic1.reporting.loaders._common import (
    model_short as _model_short,
)
from ic1.reporting.loaders._common import (
    parse_run_name as _parse_run_name,
)
from ic1.reporting.loaders._common import (
    word_churn as _word_churn,
)

# ic1/reporting/loaders/taxonomy.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP_DIR = (
    _REPO_ROOT / "ic1" / "deet" / "projects" / "taxonomy" / "data-extraction-experiments"
)

METHOD_LABELS = {
    "llm": "Flat",
    "hierarchical_top_down": "Top-down",
    "semantic": "Semantic",
}


def _prompt_family(method: str) -> str:
    """Churn lineage a run belongs to.

    Flat and top-down share the concept definition/scope-note prompts, so they
    diff against each other. Semantic and keyword runs use wholly different
    "prompt" content (embedding descriptions, keyword lists), so each diffs only
    within its own family — else the cross-family transition dwarfs everything.
    """
    return "llm" if method in ("llm", "hierarchical_top_down") else method


def _scoreable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only concepts with >=1 gold positive (evaluable concepts)."""
    gold = df.groupby("attribute_label").human_extraction.apply(
        lambda s: s.fillna(False).astype(bool).sum()
    )
    return df[df.attribute_label.isin(gold[gold >= 1].index)]


def _scores(comparison: Path) -> dict:
    df = _scoreable(pd.read_csv(comparison))
    yt = df.human_extraction.fillna(False).astype(int)
    yp = df.llm_extraction.fillna(False).astype(int)
    macro = (
        df.groupby("attribute_label")
        .apply(
            lambda a: f1_score(
                a.human_extraction.fillna(False).astype(int),
                a.llm_extraction.fillna(False).astype(int),
                zero_division=0,
            ),
            include_groups=False,
        )
        .mean()
    )
    return {
        "micro": f1_score(yt, yp, zero_division=0),
        "macro": float(macro),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "tp": int(((yt == 1) & (yp == 1)).sum()),
        "fp": int(((yt == 0) & (yp == 1)).sum()),
        "fn": int(((yt == 1) & (yp == 0)).sum()),
        "n_concepts_scoreable": int(df.attribute_label.nunique()),
        "n_docs": int(df.document_id.nunique()),
    }


def _cost(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {"cost_usd": None, "input_tokens": None, "output_tokens": None}
    meta = json.loads(meta_path.read_text())
    return {
        "cost_usd": meta.get("total_cost_usd"),
        "input_tokens": meta.get("total_input_tokens"),
        "output_tokens": meta.get("total_output_tokens"),
    }


def load_taxonomy_runs(exp_dir: Path | str | None = None) -> pd.DataFrame:
    """Walk experiment folders into one row per completed run.

    Runs missing a config or comparison CSV (incomplete/aborted) are skipped.
    Returned columns include: run, ts, label, method, method_label, model,
    model_short, votes, vocab, micro, macro, precision, recall, tp/fp/fn,
    n_docs, n_concepts_scoreable, cost_usd, input/output_tokens.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    rows = []
    for d in sorted(root.iterdir()):
        comp = d / "goldstandard_llm_comparison.csv"
        cfg = d / "config.yaml"
        if not (comp.exists() and cfg.exists()):
            continue
        c = yaml.safe_load(cfg.read_text()) or {}
        method = c.get("method", "llm")
        model = c.get("model", "?")
        vocab = "edited" if "edited" in str(c.get("vocabulary_path", "")) else "pruned"
        ts, label = _parse_run_name(d.name)
        rows.append({
            "run": d.name,
            "ts": ts,
            "label": label,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "model": model,
            "model_short": _model_short(model),
            "votes": c.get("votes", 1),
            "vocab": vocab,
            **_scores(comp),
            **_cost(d / "extraction_metadata.json"),
        })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


def load_taxonomy_prompt_churn(exp_dir: Path | str | None = None) -> pd.DataFrame:
    """Per-step word churn in the taxonomy prompts and the system prompt.

    Each row diffs a run against the chronologically previous run OF THE SAME
    prompt family (see ``_prompt_family``): taxonomy churn sums word additions/
    deletions across all concept prompts (joined on concept_id); system churn
    diffs the ``system_prompt`` string. Aligned to the same runs as
    ``load_taxonomy_runs``. The first run of each family has all-zero churn.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    runs = []
    for d in sorted(root.iterdir()):
        comp = d / "goldstandard_llm_comparison.csv"
        cfg = d / "config.yaml"
        prompts = d / "prompts_used.csv"
        if not (comp.exists() and cfg.exists() and prompts.exists()):
            continue
        c = yaml.safe_load(cfg.read_text()) or {}
        ts, label = _parse_run_name(d.name)
        p = pd.read_csv(prompts)
        runs.append({
            "run": d.name, "ts": ts, "label": label,
            "family": _prompt_family(c.get("method", "llm")),
            "system_prompt": (c.get("prompt_config") or {}).get("system_prompt", ""),
            "prompts": dict(zip(p.concept_id, p.prompt.fillna(""), strict=True)),
        })
    runs.sort(key=lambda r: r["ts"])

    rows = []
    prev_by_family: dict[str, dict] = {}
    for r in runs:
        prev = prev_by_family.get(r["family"])
        if prev is None:
            tax_add = tax_del = sys_add = sys_del = 0
        else:
            tax_add = tax_del = 0
            for cid in set(r["prompts"]) & set(prev["prompts"]):
                a, dl = _word_churn(prev["prompts"][cid], r["prompts"][cid])
                tax_add += a
                tax_del += dl
            sys_add, sys_del = _word_churn(prev["system_prompt"], r["system_prompt"])
        rows.append({
            "run": r["run"], "ts": r["ts"], "label": r["label"],
            "tax_add": tax_add, "tax_del": tax_del,
            "sys_add": sys_add, "sys_del": sys_del,
        })
        prev_by_family[r["family"]] = r
    return pd.DataFrame(rows).reset_index(drop=True)
