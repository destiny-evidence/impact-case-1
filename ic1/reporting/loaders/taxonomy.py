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
import rdflib
import yaml
from rdflib.namespace import DCTERMS, SKOS
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


def _score_df(df: pd.DataFrame) -> dict:
    """Micro/macro F1 + P/R over the SCOREABLE concepts in df (>=1 gold pos).

    Returns {} when no concept in df is scoreable (e.g. a scheme with no gold
    positive in the sample), so callers can drop it.
    """
    df = _scoreable(df)
    if df.empty:
        return {}
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


def _scores(comparison: Path) -> dict:
    return _score_df(pd.read_csv(comparison))


def _scheme_map(vocab_path: Path) -> pd.DataFrame:
    """prefLabel -> (scheme key, scheme title), parsed from a taxonomy TTL.

    ``attribute_label`` in the comparison grid is the concept skos:prefLabel,
    so this is the join from a scored concept to its SKOS ConceptScheme.
    """
    g = rdflib.Graph()
    g.parse(vocab_path, format="turtle")
    titles = {str(s): str(t) for s, t in g.subject_objects(DCTERMS.title)}
    rows = []
    for concept, scheme in g.subject_objects(SKOS.inScheme):
        pref = g.value(concept, SKOS.prefLabel)
        if pref is None:
            continue
        uri = str(scheme)
        rows.append({
            "attribute_label": str(pref),
            "scheme": uri.rsplit("/", 1)[-1],
            "scheme_title": titles.get(uri, uri.rsplit("/", 1)[-1]),
        })
    return pd.DataFrame(rows)


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


def load_taxonomy_scheme_scores(
    run: str, exp_dir: Path | str | None = None
) -> pd.DataFrame:
    """Per-scheme scores for a SINGLE run.

    Joins each scored concept to its SKOS ConceptScheme (via the run's own
    vocabulary TTL) and scores each scheme independently over that scheme's
    scoreable concepts. One row per scheme with >=1 scoreable concept, sorted by
    micro-F1 descending. Columns: run, scheme, scheme_title, plus every metric
    from ``_score_df`` (micro/macro/precision/recall/tp/fp/fn/n_*).
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    d = root / run
    c = yaml.safe_load((d / "config.yaml").read_text()) or {}
    # vocabulary_path is stored relative to the deet project dir (the parent of
    # the data-extraction-experiments folder), not the run snapshot dir.
    smap = _scheme_map((root.parent / c.get("vocabulary_path", "")).resolve())
    df = pd.read_csv(d / "goldstandard_llm_comparison.csv").merge(
        smap, on="attribute_label", how="left"
    )
    rows = []
    for (scheme, title), grp in df.groupby(["scheme", "scheme_title"]):
        s = _score_df(grp)
        if s:
            rows.append({"run": run, "scheme": scheme, "scheme_title": title, **s})
    return (
        pd.DataFrame(rows).sort_values("micro", ascending=False).reset_index(drop=True)
    )


def _resolve_vocab(run_dir: Path, exp_root: Path) -> Path:
    """The vocabulary TTL a run used (path is relative to the deet project dir)."""
    c = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    return (exp_root.parent / c.get("vocabulary_path", "")).resolve()


def load_taxonomy_level_scores(
    run: str, exp_dir: Path | str | None = None
) -> pd.DataFrame:
    """Hierarchy-respecting scores for a SINGLE run: one row per parent node.

    Every parent (each SKOS ConceptScheme for its top-level concepts, and every
    concept that has narrower concepts) becomes a group scored ONLY over its
    DIRECT children — scores never roll up across levels. Each concept is a
    member of exactly one group (its parent's) and, if it has children, also
    heads its own group. Rows carry a depth-first ``order`` and a ``level`` (0 =
    a scheme's top-level concepts) so the plot can render the group tree.

    Columns: run, scheme, scheme_title, group_key, group_label, level, order,
    parent_group, plus every metric from ``_score_df``. Sorted by ``order``.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    d = root / run
    g = rdflib.Graph()
    g.parse(_resolve_vocab(d, root), format="turtle")
    titles = {str(s): str(t) for s, t in g.subject_objects(DCTERMS.title)}
    pref = {str(c): str(p) for c, p in g.subject_objects(SKOS.prefLabel)}
    broader = {str(c): str(p) for c, p in g.subject_objects(SKOS.broader)}
    scheme = {str(c): str(s) for c, s in g.subject_objects(SKOS.inScheme)}

    def gkey(concept: str) -> str:
        """The group a concept belongs to: its parent, or its scheme if top-level."""
        return broader.get(concept) or f"scheme:{scheme[concept]}"

    def depth(concept: str) -> int:
        n, seen = 0, set()
        while concept in broader and concept not in seen:
            seen.add(concept)
            concept = broader[concept]
            n += 1
        return n

    # Score each group over the DIRECT-child concept rows it owns.
    label_group = {pref[c]: gkey(c) for c in pref if c in scheme}
    df = pd.read_csv(d / "goldstandard_llm_comparison.csv")
    df["group_key"] = df.attribute_label.map(label_group)
    scored = {k: s for k, grp in df.groupby("group_key") if (s := _score_df(grp))}

    def is_scheme(k: str) -> bool:
        return k.startswith("scheme:")

    def group_scheme_uri(k: str) -> str:
        return k[len("scheme:"):] if is_scheme(k) else scheme[k]

    def group_meta(k: str) -> dict:
        s_uri = group_scheme_uri(k)
        s_key = s_uri.rsplit("/", 1)[-1]
        s_title = titles.get(s_uri, s_key)
        return {
            "scheme": s_key,
            "scheme_title": s_title,
            "group_label": s_title if is_scheme(k) else pref.get(k, k),
            "level": 0 if is_scheme(k) else depth(k) + 1,
            "parent_group": None if is_scheme(k) else gkey(k),
        }

    # Depth-first pre-order so a parent group is listed directly above its
    # children's groups; siblings keep taxonomy (prefLabel) order.
    children: dict[str | None, list[str]] = {}
    for k in scored:
        pg = group_meta(k)["parent_group"]
        pg = pg if pg in scored else None  # orphaned groups become roots
        children.setdefault(pg, []).append(k)
    for kids in children.values():
        kids.sort(key=lambda k: group_meta(k)["group_label"])

    order: list[str] = []

    def walk(k: str) -> None:
        order.append(k)
        for kid in children.get(k, []):
            walk(kid)

    roots = sorted(children.get(None, []), key=lambda k: group_meta(k)["scheme_title"])
    for r in roots:
        walk(r)

    rows = [
        {"run": run, "group_key": k, "order": i, **group_meta(k), **scored[k]}
        for i, k in enumerate(order)
    ]
    return pd.DataFrame(rows)


def load_taxonomy_concept_tree(run: str, exp_dir: Path | str | None = None) -> list:
    """Full concept tree for a SINGLE run, for the drill-down table.

    Returns a list of scheme dicts (sorted worst direct-children F1 first), each
    ``{scheme_title, children_f1, n_children_scoreable, children:[node...]}``.
    A node is ``{label, own, children_f1, n_children_scoreable, children}`` where
    ``own`` is the concept's OWN document-level scores + the specific human/LLM
    disagreements (the failure evidence), and ``children_f1`` is the pooled F1
    over its DIRECT children only (matching the lane plot). ``own`` is None for a
    concept absent from the comparison grid.
    """
    root = Path(exp_dir) if exp_dir is not None else DEFAULT_EXP_DIR
    d = root / run
    g = rdflib.Graph()
    g.parse(_resolve_vocab(d, root), format="turtle")
    titles = {str(s): str(t) for s, t in g.subject_objects(DCTERMS.title)}
    pref = {str(c): str(p) for c, p in g.subject_objects(SKOS.prefLabel)}
    broader = {str(c): str(p) for c, p in g.subject_objects(SKOS.broader)}
    scheme = {str(c): str(s) for c, s in g.subject_objects(SKOS.inScheme)}

    kids: dict[str, list[str]] = {}
    top: dict[str, list[str]] = {}
    for c in pref:
        if c not in scheme:
            continue
        parent = broader.get(c)
        (kids.setdefault(parent, []) if parent else top.setdefault(scheme[c], [])).append(c)

    df = pd.read_csv(d / "goldstandard_llm_comparison.csv")
    by_label = {lab: sub for lab, sub in df.groupby("attribute_label")}

    def confusion(label: str) -> dict | None:
        sub = by_label.get(label)
        if sub is None:
            return None
        yt = sub.human_extraction.fillna(False).astype(int)
        yp = sub.llm_extraction.fillna(False).astype(int)
        gold = int(yt.sum())
        disagree = sub[yt.to_numpy() != yp.to_numpy()].sort_values(
            "human_extraction", ascending=False
        )
        return {
            "f1": float(f1_score(yt, yp, zero_division=0)) if gold else None,
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)) if gold else None,
            "tp": int(((yt == 1) & (yp == 1)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
            "gold": gold,
            "disagreements": [
                {
                    "document": "(untitled)" if pd.isna(r.document_name) else str(r.document_name),
                    "human": bool(r.human_extraction),
                    "llm": bool(r.llm_extraction),
                    "reasoning": "" if pd.isna(r.llm_reasoning) else str(r.llm_reasoning),
                    "verbatim": "" if pd.isna(r.llm_verbatim_text) else str(r.llm_verbatim_text),
                }
                for r in disagree.itertuples()
            ],
        }

    def children_score(labels: list[str]) -> dict:
        subs = [by_label[label] for label in labels if label in by_label]
        return _score_df(pd.concat(subs)) if subs else {}

    def build(uri: str) -> dict:
        child_uris = sorted(kids.get(uri, []), key=lambda u: pref[u])
        cs = children_score([pref[u] for u in child_uris])
        return {
            "label": pref[uri],
            "own": confusion(pref[uri]),
            "children_f1": cs.get("micro"),
            "n_children_scoreable": cs.get("n_concepts_scoreable", 0),
            "children": [build(u) for u in child_uris],
        }

    schemes = []
    for s_uri, tops in top.items():
        cs = children_score([pref[u] for u in tops])
        schemes.append({
            "scheme_title": titles.get(s_uri, s_uri.rsplit("/", 1)[-1]),
            "children_f1": cs.get("micro"),
            "n_children_scoreable": cs.get("n_concepts_scoreable", 0),
            "children": [build(u) for u in sorted(tops, key=lambda u: pref[u])],
        })
    schemes.sort(key=lambda s: (s["children_f1"] is None, s["children_f1"] or 0.0))
    return schemes


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
