#!/usr/bin/env python3
"""Harness for spot-testing prompt changes before committing to prompts/prompt_definitions.csv.

Usage (Python API):
    from test_prompt import PromptTester
    tester = PromptTester(base_label='include - best balance', config_path='configs/luna.yaml')
    results = tester.test(
        changes=[("old text", "new text")],
        test_cases=[("dairy", "dietary sulphur", "INCL")],
        n_samples=4,
    )

Usage (CLI):
    python test_prompt.py --base-label "include - best balance" \
      --old "old text" --new "new text" \
      --test-case dairy="dietary sulphur":INCL \
      --samples 4 --verbose
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

from deet.extractors.base_extractor import DataExtractionConfig
from deet.extractors.llm_data_extractor import LLMDataExtractor
from deet.data_models.base import Attribute, AttributeType
from deet.data_models.documents import ContextType


class PromptTester:
    """Test prompt modifications before committing."""

    def __init__(
        self,
        base_label: str = "include - best balance",
        config_path: str = "configs/luna.yaml",
        prompt_csv: str = "prompts/prompt_definitions.csv",
        gold_path: str = "../../../../data/private/exports/inout_deet.csv",
    ):
        """Initialize the tester.

        Args:
            base_label: which prompt row to load (e.g. 'include - best balance')
            config_path: path to extractor config YAML
            prompt_csv: path to prompt_definitions.csv
            gold_path: path to gold standard CSV with human labels
        """
        self.base_label = base_label
        self.config_path = Path(config_path)
        self.prompt_csv = Path(prompt_csv)
        self.gold_path = Path(gold_path)

        # Load base prompt and attribute_id
        self.base_prompt = None
        self.aid = None
        for r in csv.DictReader(open(self.prompt_csv)):
            if r["attribute_label"].strip() == base_label:
                self.base_prompt = r["prompt"]
                self.aid = int(r["attribute_id"])
                break
        if self.base_prompt is None:
            raise ValueError(f"Prompt label '{base_label}' not found in {self.prompt_csv}")

        # Load extractor config
        self.config = DataExtractionConfig.from_yaml(self.config_path)
        self.ex = LLMDataExtractor(config=self.config)
        self.ex.config.max_workers = 1  # sequential for testing

        # Load gold labels — index by name AND by document_id
        self.gold = {}
        self.gold_by_id = {}
        for r in csv.DictReader(open(self.gold_path)):
            self.gold[r["name"].strip().lower()] = r
            self.gold_by_id[r["document_id"].strip()] = r

    def test(
        self,
        changes: list[tuple[str, str]],
        test_cases: list[tuple[str, str, str]],
        n_samples: int = 4,
        verbose: bool = False,
    ) -> list[dict]:
        """Test a prompt modification.

        Args:
            changes: list of (old_text, new_text) tuples to apply in sequence
            test_cases: list of (label, substring_match, expected_verdict) tuples
                where expected_verdict is "INCL" or "EXCL"
            n_samples: number of samples per test case (self-consistency)
            verbose: print detailed output

        Returns:
            list of result dicts with keys:
              - label: test case label
              - human: human label (INCL or EXCL)
              - votes: f"{include_count}/{n}" tally
              - verdict: INCL or EXCL (majority vote)
              - match: True if verdict == human
        """
        # Apply changes to base
        modified = self.base_prompt
        for old, new in changes:
            if old not in modified:
                raise ValueError(f"Change anchor not found: {old[:50]!r}")
            modified = modified.replace(old, new)

        # Create attribute with modified prompt
        attr = Attribute(
            prompt=modified,
            output_data_type=AttributeType.BOOL,
            attribute_id=self.aid,
            attribute_label=self.base_label,
        )

        results = []
        for label, substring, expected in test_cases:
            # Find the document: try exact ID first, then name substring
            doc = self.gold_by_id.get(substring.strip())
            if doc is None:
                for d in self.gold.values():
                    if substring.lower() in d["name"].lower():
                        doc = d
                        break
            if doc is None:
                if verbose:
                    print(f"⚠ {label:18} doc not found (substring: {substring[:30]!r})")
                results.append(
                    {
                        "label": label,
                        "human": expected,
                        "votes": "?/?",
                        "verdict": "?",
                        "match": False,
                    }
                )
                continue

            # Sample N times
            include_count = 0
            for _ in range(n_samples):
                result = self.ex.extract_from_document(
                    attributes=[attr],
                    payload=doc["abstract"],
                    context_type=ContextType.ABSTRACT_ONLY,
                )
                if result.annotations[0].output_data:
                    include_count += 1

            verdict = "INCL" if include_count > n_samples / 2 else "EXCL"
            human_verdict = expected
            match = verdict == human_verdict

            if verbose:
                status = "✓" if match else "✗"
                print(
                    f"{status} {label:18} {include_count}/{n_samples:2} "
                    f"human={human_verdict:4} verdict={verdict:4}"
                )

            results.append(
                {
                    "label": label,
                    "human": human_verdict,
                    "votes": f"{include_count}/{n_samples}",
                    "verdict": verdict,
                    "match": match,
                }
            )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Spot-test prompt changes before committing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single change with guards
  python test_prompt.py \\
    --base-label "include - best balance" \\
    --old "old text" \\
    --new "new text" \\
    --test-case "dairy=dietary sulphur:INCL" \\
    --test-case "radar=feature fusion:EXCL" \\
    --samples 4 --verbose

  # Test multiple changes in sequence
  python test_prompt.py \\
    --base-label "include - best balance" \\
    --change "old1" "new1" \\
    --change "old2" "new2" \\
    --test-case "dairy=dietary sulphur:INCL" \\
    --samples 5
        """,
    )
    parser.add_argument(
        "--base-label",
        default="include - best balance",
        help="Prompt label to test (default: 'include - best balance')",
    )
    parser.add_argument(
        "--config",
        default="configs/luna.yaml",
        help="Extractor config YAML (default: configs/luna.yaml)",
    )
    parser.add_argument(
        "--prompt-csv",
        default="prompts/prompt_definitions.csv",
        help="Prompt definitions CSV",
    )
    parser.add_argument(
        "--gold",
        default="../../../../data/private/exports/inout_deet.csv",
        help="Gold standard CSV",
    )
    parser.add_argument(
        "--change",
        nargs=2,
        action="append",
        dest="changes",
        metavar=("OLD", "NEW"),
        help="Change to apply (old_text, new_text); can repeat",
    )
    parser.add_argument(
        "--old",
        help="Old text (shorthand for single --change)",
    )
    parser.add_argument(
        "--new",
        help="New text (shorthand for single --change)",
    )
    parser.add_argument(
        "--test-case",
        action="append",
        dest="test_cases_str",
        metavar="LABEL=SUBSTRING:VERDICT",
        help="Test case: label=substring:INCL or label=substring:EXCL; can repeat",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Samples per test case (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-test output",
    )

    args = parser.parse_args()

    # Build changes list
    changes = args.changes or []
    if args.old and args.new:
        changes.insert(0, (args.old, args.new))
    if not changes:
        parser.error("Must provide --old/--new or --change")

    # Parse test cases
    test_cases = []
    for tc_str in args.test_cases_str or []:
        parts = tc_str.split("=")
        if len(parts) != 2:
            parser.error(f"Invalid test case format: {tc_str!r} (use LABEL=SUBSTRING:VERDICT)")
        label = parts[0]
        sub_verdict = parts[1].split(":")
        if len(sub_verdict) != 2:
            parser.error(f"Invalid test case format: {tc_str!r} (use LABEL=SUBSTRING:VERDICT)")
        substring, verdict = sub_verdict
        test_cases.append((label, substring, verdict))

    if not test_cases:
        parser.error("Must provide --test-case")

    # Run
    tester = PromptTester(
        base_label=args.base_label,
        config_path=args.config,
        prompt_csv=args.prompt_csv,
        gold_path=args.gold,
    )

    results = tester.test(changes=changes, test_cases=test_cases, n_samples=args.samples, verbose=args.verbose)

    # Summary
    passed = sum(1 for r in results if r["match"])
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
