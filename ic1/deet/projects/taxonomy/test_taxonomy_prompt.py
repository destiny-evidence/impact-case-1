"""Spot-test harness for taxonomy prompt (scope-note) changes.

Runs the flat LLM extractor on the ACTIVE-split documents but only for a
handful of target attributes (`filter_attribute_ids`), so we can iterate on a
concept's definition/scope-note in the edited TTL without paying to prompt for
all ~486 attributes.

Prompts are populated exactly as production does it: `--prompt-population
taxonomy`, i.e. built from the `definition` + `scope_note` fields of the TTL
pointed at by the config's `vocabulary_path`. So editing the edited TTL and
re-running this script is a faithful preview of the real run for those concepts.

Caveat: the flat extractor sends the *selected* attributes in one prompt, so
here the target concepts are prompted in isolation (without the other 484 in
context). That validates the prompt-text change; the full run may differ
slightly because of the surrounding attribute list.

Usage:
    python test_taxonomy_prompt.py --attr-ids 454 441 --samples 3
    python test_taxonomy_prompt.py --config configs/flat_llm_luna.yaml -n 5 -v
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.project import DeetProject
from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
from deet.extractors.base_extractor import DataExtractionConfig
from deet.extractors.cli_helpers import prepare_documents
from deet.extractors.extractor_registry import get_data_extractor


def run_spot_test(
    attr_ids: list[int],
    config_path: Path,
    n_samples: int,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the target attributes N times on the active split and tally votes."""
    project = DeetProject.load()
    config = DataExtractionConfig.from_yaml(config_path)

    processed = project.process_data()
    processed.populate_custom_prompts(
        method=CustomPromptPopulationMethod.TAXONOMY, config=config
    )

    # Restrict to the active evaluation split (e.g. the 25 development docs).
    strategy = project.load_evaluation_strategy()
    active_ids = strategy.get_active_ids(project)
    processed.filter_documents_by_ids(active_ids)

    documents, _stats = prepare_documents(
        processed.documents,
        config,
        linked_document_path=project.linked_documents_path,
        pdf_dir=project.pdf_dir_abspath,
        link_map_path=project.link_map_path,
    )

    id2label = {a.attribute_id: a.attribute_label for a in processed.attributes}
    target_attrs = [a for a in processed.attributes if a.attribute_id in attr_ids]
    print(f"Spot-testing {len(target_attrs)} attribute(s) on "
          f"{len(list(documents))} document(s), {n_samples} sample(s) each:")
    for a in target_attrs:
        print(f"  [{a.attribute_id}] {a.attribute_label}")
        if verbose:
            print(f"      prompt: {a.prompt}")

    extractor = get_data_extractor(config=config)

    # (attr_id, doc_id) -> list of llm booleans across samples; gold is stable.
    votes: dict[tuple[int, int], list[bool]] = defaultdict(list)
    gold: dict[tuple[int, int], bool] = {}

    for s in range(n_samples):
        run_output = extractor.extract_from_documents(
            attributes=processed.attributes,
            documents=documents,
            filter_attribute_ids=attr_ids,
            context_type=config.default_context_type,
        )
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = GoldStandardLLMEvaluator(
                gold_standard_annotated_documents=processed.annotated_documents,
                llm_annotated_documents=run_output.annotated_documents,
                attributes=target_attrs,
                extraction_run_id=f"spot_{s}",
            )
            comp_path = Path(tmp) / "comparison.csv"
            evaluator.export_llm_comparison(comp_path)
            df = pd.read_csv(comp_path)
        df = df[df.attribute_id.isin(attr_ids)]
        for _, r in df.iterrows():
            key = (int(r.attribute_id), int(r.document_id))
            votes[key].append(bool(r.llm_extraction))
            gold[key] = bool(r.human_extraction)
        print(f"  sample {s + 1}/{n_samples} done")

    rows = []
    for key, vlist in votes.items():
        attr_id, doc_id = key
        n_true = sum(vlist)
        verdict = n_true > len(vlist) / 2
        g = gold[key]
        rows.append({
            "attribute_id": attr_id,
            "attribute_label": id2label.get(attr_id, "?"),
            "document_id": doc_id,
            "gold": int(g),
            "votes": f"{n_true}/{len(vlist)}",
            "verdict": int(verdict),
            "tp": int(g and verdict),
            "fp": int((not g) and verdict),
            "fn": int(g and (not verdict)),
            "tn": int((not g) and (not verdict)),
            "match": g == verdict,
        })
    return pd.DataFrame(rows).sort_values(["attribute_id", "document_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attr-ids", type=int, nargs="+", default=[454, 441],
                    help="Attribute ids to spot-test (default: All genders, All ages)")
    ap.add_argument("--config", type=Path, default=Path("configs/flat_llm_luna.yaml"))
    ap.add_argument("-n", "--samples", type=int, default=3)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    df = run_spot_test(
        attr_ids=args.attr_ids,
        config_path=args.config,
        n_samples=args.samples,
        verbose=args.verbose,
    )

    print("\n=== per-document ===")
    for aid, sub in df.groupby("attribute_id"):
        label = sub.attribute_label.iloc[0]
        print(f"\n[{aid}] {label}")
        for _, r in sub.iterrows():
            flag = "OK " if r.match else "XX "
            print(f"  {flag} doc {r.document_id:>12}  gold={r.gold}  "
                  f"votes={r.votes}  verdict={r.verdict}")

    print("\n=== per-attribute tally ===")
    agg = df.groupby(["attribute_id", "attribute_label"])[
        ["tp", "fp", "fn", "tn"]
    ].sum()
    agg["precision"] = agg.tp / (agg.tp + agg.fp).replace(0, pd.NA)
    agg["recall"] = agg.tp / (agg.tp + agg.fn).replace(0, pd.NA)
    agg["f1"] = 2 * agg.tp / (2 * agg.tp + agg.fp + agg.fn).replace(0, pd.NA)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
