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

import numpy as np
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
DEFAULT_SPLITS_JSON = _REPO_ROOT / "data" / "exports" / "inout_splits.json"
DEFAULT_DEET_SPLITS_JSON = (
    _REPO_ROOT / "ic1" / "deet" / "projects" / "inout" / "evaluation_splits.json"
)


def load_inout_splits(
    splits_json: Path | str | None = None,
    deet_splits_json: Path | str | None = None,
) -> dict:
    """Sizes of the evaluation splits and how many docs each stage drew.

    Reads the canonical ``inout_splits.json`` (the deterministic, recorded
    ``item_id -> split`` assignment) for the disjoint train/validation/test
    partition, and the deet project's ``evaluation_splits.json`` for the LLM's
    prompt-development subset (``development_ids``, a subset of validation).

    Returns ``{"train", "validation", "test", "total", "llm_dev"}`` (counts).
    """
    sp = Path(splits_json) if splits_json is not None else DEFAULT_SPLITS_JSON
    s = json.loads(sp.read_text())
    train, val, test = len(s["train"]), len(s["validation"]), len(s["test"])

    dp = Path(deet_splits_json) if deet_splits_json is not None else DEFAULT_DEET_SPLITS_JSON
    llm_dev = 0
    if dp.exists():
        llm_dev = len(json.loads(dp.read_text()).get("development_ids", []))
    return {
        "train": train, "validation": val, "test": test,
        "total": train + val + test, "llm_dev": llm_dev,
    }

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
        if "MODELCMP" in d.name:
            continue  # model-selection bake-off, not a prompt iteration
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

# Scope prompts in cascade order. They are near-identical in length (prompt and
# reasoning tokens within ~5%), so per-call cost is split evenly across them.
_CASCADE = ["include - high recall", "include - best balance", "include - high precision"]


def load_inout_model_costs(exp_dir: Path | str | None = None) -> pd.DataFrame:
    """Per-model best-balance F2 vs realistic cascade cost, for model selection.

    Reads the ``MODELCMP_<model>`` runs — each a single-pass (no voting), same-
    splits validation run of all three scope prompts, so models are directly
    comparable.

    **Cost** models the production cascade: high recall runs on every document,
    best balance only on high-recall includes, high precision only on those.
    Measuring the pass-through fractions p1 = P(HR⁺) and p2 = P(bal⁺ | HR⁺), the
    expected calls per document are ``1 + p1 + p1·p2`` instead of 3. The three
    prompts cost the same per call, so per-doc cost = (run cost / docs / 3) ×
    (1 + p1 + p1·p2). deet's ``total_cost_usd`` is used where available; DeepSeek/
    Kimi are estimated from tokens × published prices (``estimated`` flag).

    **F2** is the cascade's best-balance output — the gated decision HR⁺ ∩ bal⁺
    (a doc high recall drops never reaches the balance prompt). One row per model.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    rows = []
    for d in sorted(root.iterdir()):
        if "MODELCMP_" not in d.name:
            continue
        comp = d / "goldstandard_llm_comparison.csv"
        cfg = d / "config.yaml"
        meta = d / "extraction_metadata.json"
        if not (comp.exists() and cfg.exists() and meta.exists()):
            continue
        c = yaml.safe_load(cfg.read_text()) or {}
        model = c.get("model", "?")
        df = pd.read_csv(comp)
        n_docs = df.document_id.nunique()

        # Per-scope decisions and the (mode-invariant) gold.
        wide = df.pivot_table(index="document_id", columns="attribute_label",
                              values="llm_extraction", aggfunc="first")
        gold = df.groupby("document_id").human_extraction.first().fillna(False).astype(bool)
        hr = wide[_CASCADE[0]].reindex(gold.index).fillna(False).astype(bool)
        bal = wide[_CASCADE[1]].reindex(gold.index).fillna(False).astype(bool)

        # Cascade routing fractions and the gated best-balance decision.
        p1 = float(hr.mean())
        p2 = float((bal & hr).sum() / hr.sum()) if hr.sum() else 0.0
        cascade_calls = 1 + p1 + p1 * p2
        f2 = fbeta_score(gold.astype(int), (hr & bal).astype(int),
                         beta=2, zero_division=0)

        # Total run cost (all three prompts, one pass); split evenly per prompt.
        m = json.loads(meta.read_text())
        total_cost = m.get("total_cost_usd")
        estimated = False
        if total_cost is None and model in _PRICING_USD_PER_MTOK:
            pin, pout = _PRICING_USD_PER_MTOK[model]
            total_cost = (m["total_input_tokens"] * pin
                          + m["total_output_tokens"] * pout) / 1e6
            estimated = True
        if total_cost is None:
            continue
        cost_per_doc = (total_cost / n_docs / len(_CASCADE)) * cascade_calls

        rows.append({
            "model_short": _model_short(model), "model": model,
            "cost_per_doc": cost_per_doc, "f2": float(f2),
            "estimated": estimated, "p1": p1, "p2": p2,
        })
    return pd.DataFrame(rows).sort_values("cost_per_doc").reset_index(drop=True)


def load_inout_test_metrics(exp_dir: Path | str | None = None) -> pd.DataFrame:
    """Per-operating-point metrics on the latest held-out TEST run.

    Columns: Operating point, Precision, Recall, F0.5, F1, F2. One row per mode,
    in high recall -> best balance -> high precision order.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    tests = [
        d for d in sorted(root.iterdir())
        if (d / "goldstandard_llm_comparison.csv").exists() and _phase(d.name) == "test"
    ]
    if not tests:
        return pd.DataFrame()
    df = pd.read_csv(tests[-1] / "goldstandard_llm_comparison.csv")
    rows = []
    for mode_label, short in MODE_LABELS.items():
        g = df[df.attribute_label == mode_label]
        if g.empty:
            continue
        yt = g.human_extraction.fillna(False).astype(int)
        yp = g.llm_extraction.fillna(False).astype(int)
        rows.append({
            "Operating point": short,
            "Precision": precision_score(yt, yp, zero_division=0),
            "Recall": recall_score(yt, yp, zero_division=0),
            "F0.5": fbeta_score(yt, yp, beta=0.5, zero_division=0),
            "F1": f1_score(yt, yp, zero_division=0),
            "F2": fbeta_score(yt, yp, beta=2, zero_division=0),
        })
    return pd.DataFrame(rows)


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


# Real (non-dev) finalised ML models. Dev-mode runs land under models/testing
# with a test set fabricated from train — unusable for the head-to-head.
_ML_MODEL_DIR = _REPO_ROOT / "data" / "models" / "inout" / "results" / "model"


def _latest_test_run(root: Path | None = None) -> Path:
    """Most recent held-out TEST deet run (folder name marks the phase)."""
    root = root or DEFAULT_EXP_DIR
    tests = [
        d for d in sorted(root.iterdir())
        if (d / "goldstandard_llm_comparison.csv").exists() and _phase(d.name) == "test"
    ]
    if not tests:
        raise FileNotFoundError(f"No TEST run under {root}")
    return tests[-1]


def _point(yt: np.ndarray, yp: np.ndarray) -> dict:
    return {
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "f2": fbeta_score(yt, yp, beta=2, zero_division=0),
    }


def _ci(yt: np.ndarray, yp: np.ndarray, *, seed: int) -> dict:
    """Bayesian HDIs via prob_conf_mat — the same method as classify's finalise
    (``posterior_metric_summaries``). Two calls (beta=1, beta=2) cover F1 and F2;
    Precision/Recall are shared. Returns ``{metric}_lo``/``{metric}_hi``."""
    from ic1.classify.inout.utils.metrics import posterior_metric_summaries

    s1 = posterior_metric_summaries(yt, yp, beta=1, seed=seed)
    s2 = posterior_metric_summaries(yt, yp, beta=2, seed=seed)
    pairs = (("precision", s1["Precision"]), ("recall", s1["Recall"]),
             ("f1", s1["Fbeta"]), ("f2", s2["Fbeta"]))
    out = {}
    for key, m in pairs:
        out[f"{key}_lo"], out[f"{key}_hi"] = m["hdi_lo"], m["hdi_hi"]
    return out


# Operating points reported for the LLM-only and cascade families, in
# recall -> balance -> precision order (matches the timeline figure).
_MODE_ORDER = ("high recall", "best balance", "high precision")


def load_inout_prompts(
    llm_run: Path | str | None = None, headline: str = "best balance"
) -> dict:
    """The exact prompts from the final held-out TEST run, for the docs.

    Reads the run's ``config.yaml`` (system prompt) and ``prompts_used.csv``
    (per-operating-point scope prompts). The three scope prompts share their
    system prompt and CLIMATE/HEALTH definitions verbatim and differ only in
    wording, so the renderer shows the ``headline`` prompt in full and the others
    as a diff against it.

    Returns ``{"run", "system", "headline", "modes": {mode: full_prompt}}`` with
    modes in ``_MODE_ORDER``.
    """
    root = Path(llm_run) if llm_run is not None else _latest_test_run()
    cfg = yaml.safe_load((root / "config.yaml").read_text()) or {}
    system = (cfg.get("prompt_config") or {}).get("system_prompt", "").strip()

    pdf = pd.read_csv(root / "prompts_used.csv")
    pdf["mode"] = pdf.attribute_label.map(MODE_LABELS)
    modes = {
        m: pdf.loc[pdf["mode"] == m, "prompt"].iloc[0].strip()
        for m in _MODE_ORDER if (pdf["mode"] == m).any()
    }
    if headline not in modes:
        headline = next(iter(modes))
    return {"run": root.name, "system": system, "headline": headline, "modes": modes}


def load_inout_comparison(
    llm_run: Path | str | None = None,
    ml_model_dir: Path | str | None = None,
    corpus_size: int = 6_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Head-to-head on the *identical* held-out test set, with Bayesian 95% HDIs
    and a cost proxy (issue #1).

    Join key: the deet ``external_id`` IS the nacsos ``item_id`` (== splits.test),
    so the LLM comparison rows and the finalised ML predictions align 1:1.

    Systems — one row per operating point for the two LLM-using families, plus a
    single ML-only reference:
      - LLM only (mode) : that operating-point decision on every doc.
      - ML only         : the F-beta-selected model, thresholded ("Optimal ML-only").
      - ML -> LLM (mode): the high-recall *filtering* model gates the corpus; docs
                          it passes take the LLM's ``mode`` decision, docs it drops
                          are excluded — simulated by reusing the LLM's decisions
                          on the passed docs.

    Intervals use the same Dirichlet-multinomial confusion-matrix posterior as
    ``classify``'s finalise (``posterior_metric_summaries``). Cost is the LLM
    run's recorded USD/doc extrapolated to ``corpus_size``; the cascade pays it
    only on the fraction the filter forwards, ML-only is ~free. Columns include
    ``system``, ``family``, ``mode``, the four point metrics, ``*_lo``/``*_hi``
    HDIs, ``prop_to_llm`` and ``corpus_cost``.
    """
    root = Path(llm_run) if llm_run is not None else _latest_test_run()
    mld = Path(ml_model_dir) if ml_model_dir is not None else _ML_MODEL_DIR

    comp = pd.read_csv(root / "goldstandard_llm_comparison.csv")
    comp = comp[comp.attribute_label.isin(MODE_LABELS)].copy()
    comp["mode"] = comp.attribute_label.map(MODE_LABELS)
    # Gold is mode-invariant; one LLM decision column per operating point.
    hum = comp.groupby("external_id").human_extraction.first()
    llm_wide = comp.pivot_table(
        index="external_id", columns="mode", values="llm_extraction", aggfunc="first"
    )
    base = pd.DataFrame({"human_extraction": hum}).join(llm_wide).reset_index()

    ml_only = pd.read_csv(mld / "ml_only" / "test_predictions.csv")
    filt = pd.read_csv(mld / "filtering" / "test_predictions.csv")
    t_ml = json.loads((mld / "ml_only" / "train_info.json").read_text())["threshold"]
    t_filt = json.loads((mld / "filtering" / "train_info.json").read_text())["threshold"]

    df = (
        base.merge(ml_only.rename(columns={"y_prob": "ml_prob"}),
                   left_on="external_id", right_on="item_id")
            .merge(filt[["item_id", "y_prob"]].rename(columns={"y_prob": "filt_prob"}),
                   on="item_id")
    )
    if df.empty:
        raise ValueError(
            "No overlap between the LLM test docs and the ML predictions. The "
            "committed ML run is dev-mode (data/models/testing/...), whose test "
            "set is fabricated from train. Re-run `finalise_models` with "
            "dev_mode=False so predictions cover splits.test."
        )

    yt = df.human_extraction.fillna(False).astype(int).to_numpy()
    filt_pass = (df.filt_prob.to_numpy() > t_filt).astype(int)
    ml_pred = (df.ml_prob.to_numpy() > t_ml).astype(int)
    llm_inc = {
        m: df[m].fillna(False).astype(int).to_numpy()
        for m in _MODE_ORDER if m in df.columns
    }

    meta = json.loads((root / "extraction_metadata.json").read_text())
    llm_cost_doc = meta["total_cost_usd"] / comp.external_id.nunique()
    prop_pass = float(filt_pass.mean())        # cascade: LLM runs on forwarded docs
    prop_reject = 1.0 - float(ml_pred.mean())  # OR: LLM runs on ML's rejects only

    # (system label, family, mode, y_pred, fraction hitting the LLM)
    systems: list[tuple[str, str, str | None, np.ndarray, float]] = []
    for m in llm_inc:
        systems.append((f"LLM only ({m})", "LLM only", m, llm_inc[m], 1.0))
    systems.append(("ML only", "ML only", None, ml_pred, 0.0))
    for m in llm_inc:
        systems.append(
            (f"ML → LLM ({m})", "ML → LLM", m, filt_pass & llm_inc[m], prop_pass)
        )
    # OR-ensemble: include if ML *or* the LLM includes. Uses the balanced ml_only
    # model (OR-ing with the high-recall filter would collapse precision). The
    # prediction equals a full OR, but you only need the LLM on docs ML rejected —
    # ML's accepts are already positive — so it costs prop_reject, not the full run.
    for m in llm_inc:
        systems.append(
            (f"ML or LLM ({m})", "ML or LLM", m, ml_pred | llm_inc[m], prop_reject)
        )

    rows = []
    for name, family, mode, yp, to_llm in systems:
        rows.append({
            "system": name, "family": family, "mode": mode, "n": int(df.shape[0]),
            **_point(yt, yp),
            **_ci(yt, yp, seed=seed),
            "prop_to_llm": to_llm,
            "cost_per_doc": llm_cost_doc * to_llm,
            "corpus_cost": llm_cost_doc * to_llm * corpus_size,
        })
    return pd.DataFrame(rows)
