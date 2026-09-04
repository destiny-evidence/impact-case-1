"""Load in/out (include/exclude) screening experiments into tidy frames.

Same deet on-disk layout as taxonomy, but the story differs: each run's
``attribute_label`` is an *operating point* (high recall / best balance / high
precision), not a taxonomy concept — so we score precision/recall per mode, not
micro/macro-F1. Runs are folded by identical prompt-content (resample siblings
collapse to one point, mean P/R). Doc counts trace the dev/validation cycles
(dev 65 -> val 65 -> dev 130 -> val 70 -> dev 200).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from ic1.reporting.loaders._common import model_short as _model_short
from ic1.reporting.loaders._common import parse_run_name as _parse_run_name
from ic1.reporting.loaders._common import word_churn as _word_churn

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP_DIR = (
    _REPO_ROOT / "ic1" / "deet" / "projects" / "inout" / "data-extraction-experiments"
)

# Operating points kept (short label + colour key). "max precision" was dropped
# (collapsed) in the final design, so it is excluded here.
MODE_LABELS = {
    "include - high recall": "high recall",
    "include - best balance": "best balance",
    "include - high precision": "high precision",
}

_RESAMPLE_SUFFIX = re.compile(r"(_resample_\d+|_s\d+|_\d+)$")


def _clean_label(label: str) -> str:
    """Strip resample-sibling suffixes so a folded state reads cleanly."""
    return _RESAMPLE_SUFFIX.sub("", label) or "base"


def _phase(run_name: str) -> str:
    """Dev / validation / test cycle from the run folder name (deet's markers)."""
    if "TEST" in run_name:
        return "test"
    if "VALIDATION" in run_name:
        return "validation"
    return "dev"


def _mode_scores(mdf: pd.DataFrame) -> dict:
    yt = mdf.human_extraction.fillna(False).astype(int)
    yp = mdf.llm_extraction.fillna(False).astype(int)
    return {
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
    }


def _collect_states(root: Path, models: tuple[str, ...]) -> list[dict]:
    """Per-run records, folded by identical prompt-content (mean P/R per mode).

    A "state" is a distinct prompt configuration; resample runs of the same
    prompts/model/phase/docs collapse into one, so the timeline shows prompt
    states rather than repeated draws. Returned in chronological order.
    """
    runs = []
    for d in sorted(root.iterdir()):
        comp = d / "goldstandard_llm_comparison.csv"
        cfg = d / "config.yaml"
        prompts = d / "prompts_used.csv"
        if not (comp.exists() and cfg.exists() and prompts.exists()):
            continue
        c = yaml.safe_load(cfg.read_text()) or {}
        ms = _model_short(c.get("model", "?"))
        if ms not in models:
            continue
        df = pd.read_csv(comp)
        ts, label = _parse_run_name(d.name)
        pdf = pd.read_csv(prompts)
        modes = {
            MODE_LABELS[m]: _mode_scores(g)
            for m, g in df.groupby("attribute_label")
            if m in MODE_LABELS
        }
        runs.append({
            "run": d.name, "ts": ts, "label": _clean_label(label),
            "model": c.get("model", "?"), "model_short": ms,
            "votes": c.get("votes") or 1,
            "phase": _phase(d.name),
            "n_docs": int(df.document_id.nunique()),
            "system_prompt": (c.get("prompt_config") or {}).get("system_prompt", ""),
            "prompts": dict(zip(pdf.attribute_id, pdf.prompt.fillna(""), strict=True)),
            "modes": modes,
        })

    # Fold resample siblings: same model / phase / docs / prompts -> one state.
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for r in runs:
        # Fold only true resample siblings: same cleaned label AND same config/
        # content. Without the label, two distinct experiments whose stored
        # prompts happen to be identical (e.g. a change that lives in code, not
        # the saved prompt text) would wrongly collapse into one point.
        sig = (
            r["model_short"], r["phase"], r["n_docs"], r["label"],
            r["system_prompt"], tuple(sorted(r["prompts"].items())),
        )
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append(r)

    states = []
    for sig in order:
        grp = sorted(groups[sig], key=lambda r: r["ts"])
        rep = grp[0]
        mode_names = {m for r in grp for m in r["modes"]}
        folded_modes = {
            m: {
                k: float(pd.Series([r["modes"][m][k] for r in grp if m in r["modes"]]).mean())
                for k in ("precision", "recall", "f1")
            }
            for m in mode_names
        }
        states.append({**rep, "modes": folded_modes, "n_resamples": len(grp)})
    states.sort(key=lambda s: s["ts"])
    return states


def load_inout_runs(
    exp_dir: Path | str | None = None, models: tuple[str, ...] = ("sol", "luna", "terra")
) -> pd.DataFrame:
    """One row per (prompt-state, operating mode) with precision/recall/f1.

    ``method_label`` carries the operating mode (for colour) and ``model_short``
    the model (for marker), matching the taxonomy schema so the shared timeline
    plot works unchanged.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    rows = []
    for s in _collect_states(root, models):
        for mode, sc in s["modes"].items():
            rows.append({
                "run": s["run"], "ts": s["ts"], "label": s["label"],
                "method_label": mode, "model": s["model"],
                "model_short": s["model_short"], "votes": s["votes"],
                "phase": s["phase"], "n_docs": s["n_docs"],
                "n_resamples": s["n_resamples"], "vocab": "n/a",
                **sc,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_inout_prompt_churn(
    exp_dir: Path | str | None = None, models: tuple[str, ...] = ("sol", "luna", "terra")
) -> pd.DataFrame:
    """Per-state word churn in the scope prompts and the system prompt.

    Diffs each folded state against the chronologically previous one, aligned to
    the same states (``run``) as ``load_inout_runs``.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    states = _collect_states(root, models)
    rows = []
    prev = None
    for s in states:
        if prev is None:
            tax_add = tax_del = sys_add = sys_del = 0
        else:
            tax_add = tax_del = 0
            for cid in set(s["prompts"]) & set(prev["prompts"]):
                before, after = prev["prompts"][cid], s["prompts"][cid]
                # Skip modes left empty in either state (not under test that run)
                # — else the blank counts as a huge spurious delete/re-add.
                if not before.strip() or not after.strip():
                    continue
                a, dl = _word_churn(before, after)
                tax_add += a
                tax_del += dl
            sys_add, sys_del = _word_churn(prev["system_prompt"], s["system_prompt"])
        rows.append({
            "run": s["run"], "ts": s["ts"], "label": s["label"],
            "tax_add": tax_add, "tax_del": tax_del,
            "sys_add": sys_add, "sys_del": sys_del,
        })
        prev = s
    return pd.DataFrame(rows).reset_index(drop=True)


# Published list prices (USD per 1M tokens, input/output) for models deet's own
# cost table doesn't cover (accessed via Azure Foundry). DeepSeek V4 Pro off-peak
# and Kimi K2.6 official rates as of 2026-09; actual Foundry cost may differ, so
# these points are flagged as estimates. See figure caption.
_PRICING_USD_PER_MTOK = {
    "DeepSeek-V4-Pro": (0.66, 1.98),
    "Kimi-K2.6": (0.95, 4.00),
}

# Representative run(s) per model for the cost/performance comparison. luna,
# opus, deepseek and kimi come from the 2026-09-01 bakeoff — identical prompts,
# 130 docs, one vote — so their F2 is a controlled comparison. terra and sol are
# each at their own representative config (the figure notes this).
_MODEL_COMPARISON_RUNS = {
    "luna": ["_08-25-12_luna", "_luna_s1", "_luna_s2"],
    "opus": ["_opus"],
    "deepseek": ["_deepseek"],
    "kimi": ["_kimi26"],
    "terra": ["_20-22-45_terra"],
    "sol": ["_all_modes_resample"],
}


def load_inout_model_costs(exp_dir: Path | str | None = None) -> pd.DataFrame:
    """Per-model best-balance F2 vs cost per document for model selection.

    Cost is deet's recorded ``total_cost_usd`` where available; for models deet
    does not price (DeepSeek, Kimi) it is estimated from recorded token counts x
    published list prices (``_PRICING_USD_PER_MTOK``), flagged via ``estimated``.
    F2 (recall-weighted) is over the best-balance operating point. Resample
    siblings are averaged. One row per model.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    per_run = []
    for d in sorted(root.iterdir()):
        comp = d / "goldstandard_llm_comparison.csv"
        cfg = d / "config.yaml"
        meta = d / "extraction_metadata.json"
        if not (comp.exists() and cfg.exists() and meta.exists()):
            continue
        model_short = next(
            (m for m, pats in _MODEL_COMPARISON_RUNS.items()
             if any(p in d.name for p in pats)),
            None,
        )
        if model_short is None:
            continue
        c = yaml.safe_load(cfg.read_text()) or {}
        m = json.loads(meta.read_text())
        df = pd.read_csv(comp)
        bb = df[df.attribute_label == "include - best balance"]
        if bb.empty:
            continue
        f2 = fbeta_score(
            bb.human_extraction.fillna(False).astype(int),
            bb.llm_extraction.fillna(False).astype(int),
            beta=2, zero_division=0,
        )
        cost = m.get("total_cost_usd")
        estimated = False
        model = c.get("model", "?")
        if cost is None and model in _PRICING_USD_PER_MTOK:
            pin, pout = _PRICING_USD_PER_MTOK[model]
            cost = (m["total_input_tokens"] * pin + m["total_output_tokens"] * pout) / 1e6
            estimated = True
        if cost is None:
            continue
        n_docs = df.document_id.nunique()
        per_run.append({
            "model_short": model_short, "model": model,
            "cost_per_doc": cost / n_docs, "f2": float(f2),
            "estimated": estimated,
        })

    df = pd.DataFrame(per_run)
    return (
        df.groupby("model_short", as_index=False)
        .agg(model=("model", "first"), cost_per_doc=("cost_per_doc", "mean"),
             f2=("f2", "mean"), estimated=("estimated", "first"))
    )


def inout_cycle_spans(runs: pd.DataFrame) -> list[dict]:
    """Shaded spans for the dev/validation cycles, labelled with doc counts.

    Groups consecutive prompt-states (in x-order) sharing (phase, n_docs) into
    one span. Returns dicts with x0/x1 (in step index space), label and kind.
    """
    steps = (
        runs[["run", "ts", "phase", "n_docs"]]
        .drop_duplicates("run").sort_values("ts").reset_index(drop=True)
    )
    spans = []
    start = 0
    for i in range(1, len(steps) + 1):
        prev = steps.iloc[i - 1]
        boundary = i == len(steps) or (
            steps.iloc[i].phase != prev.phase or steps.iloc[i].n_docs != prev.n_docs
        )
        if boundary:
            spans.append({
                "x0": start - 0.5, "x1": i - 0.5,
                "label": f"{prev.phase} · {int(prev.n_docs)} docs",
                "kind": prev.phase,
            })
            start = i
    return spans
