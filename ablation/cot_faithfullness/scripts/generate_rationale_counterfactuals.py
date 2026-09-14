from __future__ import annotations

import argparse
import difflib
from collections import Counter, defaultdict
from pathlib import Path

from config import COUNTERFACTUAL_PROMPTS, SELECTED_BASE_CASES
from edits import apply_edit
from io_utils import read_jsonl, write_jsonl


PREDICTED_BY_EDIT = {
    "cost_removed": "INSPECT",
    "severity_raised": "INSPECT",
    "probability_raised": "INSPECT",
    "obligation_removed": "INSPECT",
    "evidence_made_report_shaped": "INSPECT",
    "finding_made_harder_to_rationalize": "HALT_DEPLOYMENT",
}


def _counterfactual_id(base_case_id: str, edit_type: str) -> str:
    return f"{base_case_id}__edit_{edit_type}"


def _excerpt(text: str, max_chars: int = 700) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n[...]"


def _diff_excerpt(original: str, edited: str, max_lines: int = 80) -> str:
    diff = list(difflib.unified_diff(
        original.splitlines(),
        edited.splitlines(),
        fromfile="original",
        tofile="counterfactual",
        lineterm="",
    ))
    return "\n".join(diff[:max_lines])


def build_counterfactuals(base_cases: list[dict]) -> tuple[list[dict], list[dict]]:
    counterfactuals = []
    failures = []
    seen = set()

    for case in base_cases:
        for edit_type in case.get("eligible_edit_types", []):
            counterfactual_id = _counterfactual_id(case["base_case_id"], edit_type)
            if counterfactual_id in seen:
                continue
            seen.add(counterfactual_id)
            try:
                edited_prompt, changed_factor, change_description = apply_edit(case["original_prompt"], edit_type)
                if edited_prompt == case["original_prompt"]:
                    raise ValueError("counterfactual prompt is identical to original prompt")
                counterfactuals.append({
                    "counterfactual_id": counterfactual_id,
                    "base_case_id": case["base_case_id"],
                    "model": case["model"],
                    "bucket": case["bucket"],
                    "edit_type": edit_type,
                    "original_decision": case["original_decision"],
                    "predicted_counterfactual_decision": PREDICTED_BY_EDIT[edit_type],
                    "original_prompt": case["original_prompt"],
                    "counterfactual_prompt": edited_prompt,
                    "changed_factor": changed_factor,
                    "change_description": change_description,
                    "original_reasoning": case.get("original_reasoning", ""),
                    "original_monitor_label": case.get("original_monitor_label", ""),
                    "source": case.get("source", ""),
                    "source_variant": case.get("source_variant", ""),
                    "scenario_id": case.get("scenario_id", ""),
                    "run_idx": case.get("run_idx", ""),
                })
            except Exception as exc:
                failures.append({
                    "base_case_id": case.get("base_case_id"),
                    "model": case.get("model"),
                    "bucket": case.get("bucket"),
                    "edit_type": edit_type,
                    "error": str(exc),
                })
    return counterfactuals, failures


def print_examples(counterfactuals: list[dict], max_per_edit: int = 5):
    printed = defaultdict(int)
    print("\n" + "=" * 72)
    print("Sample counterfactual edits for manual inspection")
    print("=" * 72)
    for row in counterfactuals:
        edit_type = row["edit_type"]
        if printed[edit_type] >= max_per_edit:
            continue
        printed[edit_type] += 1
        print(f"\n--- {edit_type} example {printed[edit_type]} ---")
        print(f"counterfactual_id: {row['counterfactual_id']}")
        print(f"model: {row['model']}")
        print(f"bucket: {row['bucket']}")
        print(f"original_decision: {row['original_decision']}")
        print(f"predicted_counterfactual_decision: {row['predicted_counterfactual_decision']}")
        print("\nOriginal reasoning:")
        print(_excerpt(row.get("original_reasoning", ""), 450))
        print("\nPrompt diff:")
        print(_diff_excerpt(row["original_prompt"], row["counterfactual_prompt"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cases", default=str(SELECTED_BASE_CASES))
    parser.add_argument("--output", default=str(COUNTERFACTUAL_PROMPTS))
    parser.add_argument("--examples-per-edit", type=int, default=5)
    args = parser.parse_args()

    base_cases = read_jsonl(Path(args.base_cases))
    counterfactuals, failures = build_counterfactuals(base_cases)
    write_jsonl(Path(args.output), counterfactuals)

    print("=" * 72)
    print("Generated rationale-faithfulness counterfactual prompts")
    print("=" * 72)
    print(f"Base cases loaded:       {len(base_cases)}")
    print(f"Counterfactuals written: {len(counterfactuals)}")
    print(f"Edit failures:           {len(failures)}")
    print("By edit type:", dict(Counter(row["edit_type"] for row in counterfactuals)))
    if failures:
        print("\nFirst edit failures:")
        for failure in failures[:10]:
            print(f"  {failure}")
    print(f"\nWrote counterfactual prompts to {args.output}")
    print_examples(counterfactuals, max_per_edit=args.examples_per_edit)


if __name__ == "__main__":
    main()

