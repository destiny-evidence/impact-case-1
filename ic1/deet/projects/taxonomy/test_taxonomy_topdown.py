"""Top-down spot-test harness for iterating on one scheme's hierarchy.

The flat harness (test_taxonomy_prompt.py) runs the flat extractor and so cannot
see top-down's central dynamic: a missed *root* prunes its whole subtree. This
harness runs the real TopDownLLMExtractor but restricted to a single scheme, so
we can iterate on root/branch definitions cheaply and watch both recall and the
pruning cascade — without paying to walk all 19 schemes.

It restricts the extractor to one scheme by filtering `load_schemes`, runs N
samples on the active-split docs, votes, and reports per-concept tp/fp/fn plus a
cascade breakdown of the false negatives:
  - root-level FN  (a depth-1 root itself missed)
  - pruned FN      (an ancestor was negative, so the subtree was cut)
  - leaf-miss FN   (parent positive, but the concept still missed)

Usage:
    python test_taxonomy_topdown.py --scheme "Interventions" -n 3
    python test_taxonomy_topdown.py --scheme "Interventions" -n 3 --focus 321 319
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.project import DeetProject
from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
from deet.extractors.base_extractor import DataExtractionConfig
from deet.extractors.cli_helpers import prepare_documents
from deet.extractors.hierarchical.top_down_llm_extractor import TopDownLLMExtractor

MAPPING = Path("../../../../data/scheme/destiny_taxonomy_nacsos_mapping.json")


def _hierarchy():
    """Return (concept_id_by_label, parent_label_by_label, descendants_by_label)."""
    m = json.loads(MAPPING.read_text())
    by_uri = {x["concept_uri"]: x for x in m}
    kids: dict[str, list[str]] = defaultdict(list)
    for x in m:
        if x["broader_uri"]:
            kids[x["broader_uri"]].append(x["concept_uri"])

    def descendants(uri):
        out, stack = [], list(kids.get(uri, []))
        while stack:
            u = stack.pop()
            out.append(u)
            stack += kids.get(u, [])
        return out

    label_uri = {x["pref_label"]: x["concept_uri"] for x in m}
    parent = {
        x["pref_label"]: (by_uri.get(x["broader_uri"], {}).get("pref_label"))
        for x in m
    }
    desc_labels = {
        x["pref_label"]: [by_uri[u]["pref_label"] for u in descendants(x["concept_uri"])]
        for x in m
    }
    return label_uri, parent, desc_labels


def run_topdown(scheme_query: str, config_path: Path, n_samples: int) -> pd.DataFrame:
    project = DeetProject.load()
    config = DataExtractionConfig.from_yaml(config_path)

    processed = project.process_data()
    processed.populate_custom_prompts(
        method=CustomPromptPopulationMethod.TAXONOMY, config=config
    )
    strategy = project.load_evaluation_strategy()
    processed.filter_documents_by_ids(strategy.get_active_ids(project))
    documents, _ = prepare_documents(
        processed.documents, config,
        linked_document_path=project.linked_documents_path,
        pdf_dir=project.pdf_dir_abspath, link_map_path=project.link_map_path,
    )

    extractor = TopDownLLMExtractor(config=config)
    _orig_load = extractor.load_schemes
    extractor.load_schemes = lambda: [
        s for s in _orig_load() if scheme_query.lower() in s.title.lower()
    ]
    schemes = extractor.load_schemes()
    if not schemes:
        raise SystemExit(f"No scheme matching '{scheme_query}'")
    print(f"Top-down over scheme: {schemes[0].title!r} "
          f"({len(list(documents))} docs, {n_samples} samples)")

    votes: dict[tuple[int, int], list[bool]] = defaultdict(list)
    gold: dict[tuple[int, int], bool] = {}
    id2label: dict[int, str] = {}

    for s in range(n_samples):
        run_output = extractor.extract_from_documents(
            attributes=processed.attributes,
            documents=documents,
            context_type=config.default_context_type,
        )
        scheme_attrs = [c.attribute for c in extractor.mapped_schemes[0].concepts.values()]
        for a in scheme_attrs:
            id2label[a.attribute_id] = a.attribute_label
        with tempfile.TemporaryDirectory() as tmp:
            ev = GoldStandardLLMEvaluator(
                gold_standard_annotated_documents=processed.annotated_documents,
                llm_annotated_documents=run_output.annotated_documents,
                attributes=scheme_attrs,
                extraction_run_id=f"td_{s}",
            )
            comp = Path(tmp) / "c.csv"
            ev.export_llm_comparison(comp)
            df = pd.read_csv(comp)
        for _, r in df.iterrows():
            key = (int(r.attribute_id), int(r.document_id))
            votes[key].append(bool(r.llm_extraction))
            gold[key] = bool(r.human_extraction)
        print(f"  sample {s + 1}/{n_samples} done")

    rows = []
    for (aid, doc), vlist in votes.items():
        verdict = sum(vlist) > len(vlist) / 2
        g = gold[(aid, doc)]
        rows.append({
            "attribute_id": aid, "attribute_label": id2label.get(aid, str(aid)),
            "document_id": doc, "gold": int(g), "verdict": int(verdict),
            "tp": int(g and verdict), "fp": int((not g) and verdict),
            "fn": int(g and not verdict),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", required=True, help="Scheme title substring, e.g. 'Interventions'")
    ap.add_argument("--config", type=Path, default=Path("configs/top_down_luna.yaml"))
    ap.add_argument("-n", "--samples", type=int, default=3)
    ap.add_argument("--focus", type=int, nargs="*", default=None,
                    help="attribute ids to print per-doc detail for")
    args = ap.parse_args()

    df = run_topdown(args.scheme, args.config, args.samples)
    _, parent, desc_labels = _hierarchy()

    # Per-concept tally (only concepts with >=1 gold are scoreable).
    agg = df.groupby(["attribute_id", "attribute_label"])[["tp", "fp", "fn"]].sum()
    agg["gold"] = agg.tp + agg.fn
    scoreable = agg[agg.gold > 0].copy()
    scoreable["depth_note"] = [
        "ROOT" if parent.get(lbl) is None else "" for _, lbl in scoreable.index
    ]
    print("\n=== scoreable concepts (>=1 gold) ===")
    print(scoreable.sort_values("gold", ascending=False)[
        ["gold", "tp", "fp", "fn", "depth_note"]].to_string())

    tp, fp, fn = scoreable.tp.sum(), scoreable.fp.sum(), scoreable.fn.sum()
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    print(f"\nscheme micro: P {P:.3f}  R {R:.3f}  F1 "
          f"{2 * tp / (2 * tp + fp + fn):.3f}  (tp{tp} fp{fp} fn{fn})")

    # Cascade breakdown of FN.
    gold_pos = {(r.attribute_label, r.document_id)
                for r in df.itertuples() if r.gold}
    verdict_pos = {(r.attribute_label, r.document_id)
                   for r in df.itertuples() if r.verdict}
    root_fn = pruned = leaf = 0
    for r in df.itertuples():
        if not (r.gold and not r.verdict):
            continue
        par = parent.get(r.attribute_label)
        if par is None:
            root_fn += 1
        elif (par, r.document_id) not in verdict_pos:
            pruned += 1
        else:
            leaf += 1
    print(f"\nFN breakdown: root {root_fn} | pruned (ancestor cut) {pruned} | leaf-miss {leaf}")

    if args.focus:
        for aid in args.focus:
            sub = df[df.attribute_id == aid]
            if not len(sub):
                continue
            lbl = sub.attribute_label.iloc[0]
            print(f"\n[{aid}] {lbl}")
            for r in sub.sort_values("document_id").itertuples():
                flag = "OK " if r.gold == r.verdict else "XX "
                print(f"  {flag} doc {r.document_id:>12}  gold={r.gold} verdict={r.verdict}")


if __name__ == "__main__":
    main()
