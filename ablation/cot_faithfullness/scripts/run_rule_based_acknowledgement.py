from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from acknowledgement import rule_based_acknowledgement
from config import (
    COUNTERFACTUAL_PROMPTS,
    PARSED_OUTPUTS,
    RULE_BASED_ACKNOWLEDGEMENT_OUTPUTS,
)
from io_utils import append_jsonl, read_jsonl


def run_rule_based_acknowledgement(
    prompts_path: Path = COUNTERFACTUAL_PROMPTS,
    parsed_path: Path = PARSED_OUTPUTS,
    output_path: Path = RULE_BASED_ACKNOWLEDGEMENT_OUTPUTS,
    limit: int | None = None,
):
    prompts = {row["counterfactual_id"]: row for row in read_jsonl(prompts_path)}
    parsed_rows = read_jsonl(parsed_path)
    if limit is not None:
        parsed_rows = parsed_rows[:limit]

    existing = read_jsonl(output_path)
    done = {row["counterfactual_id"] for row in existing if row.get("counterfactual_id")}
    rows_to_annotate = [
        row for row in parsed_rows
        if row.get("counterfactual_id") not in done
        and row.get("parse_status") == "ok"
        and row.get("parsed_reasoning")
        and row.get("counterfactual_id") in prompts
    ]

    total = len(parsed_rows)
    remaining = len(rows_to_annotate)
    print("=" * 72)
    print("Running rule-based acknowledgement annotator")
    print("=" * 72)
    print(f"Parsed outputs: {parsed_path}")
    print(f"Prompts:        {prompts_path}")
    print(f"Output:         {output_path}")
    print(f"Total rows:     {total}")
    print(f"Remaining:      {remaining}")
    print("=" * 72)

    for idx, parsed in enumerate(rows_to_annotate, 1):
        counterfactual_id = parsed["counterfactual_id"]
        prompt_row = prompts[counterfactual_id]
        annotation = rule_based_acknowledgement(
            edit_type=prompt_row.get("edit_type", parsed.get("edit_type", "")),
            reasoning=parsed.get("parsed_reasoning", ""),
            changed_factor=prompt_row.get("changed_factor", ""),
            change_description=prompt_row.get("change_description", ""),
        )
        out = {
            "counterfactual_id": counterfactual_id,
            "base_case_id": parsed.get("base_case_id", prompt_row.get("base_case_id")),
            "model": parsed.get("model", prompt_row.get("model")),
            "bucket": parsed.get("bucket", prompt_row.get("bucket")),
            "edit_type": parsed.get("edit_type", prompt_row.get("edit_type")),
            "changed_factor": prompt_row.get("changed_factor", ""),
            "change_description": prompt_row.get("change_description", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **annotation,
        }
        append_jsonl(output_path, out)
        print(
            f"[{idx}/{remaining}] {out.get('model')} {out.get('edit_type')} "
            f"{counterfactual_id} ... {out['rule_based_acknowledgement_label']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(COUNTERFACTUAL_PROMPTS))
    parser.add_argument("--parsed", default=str(PARSED_OUTPUTS))
    parser.add_argument("--output", default=str(RULE_BASED_ACKNOWLEDGEMENT_OUTPUTS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_rule_based_acknowledgement(
        prompts_path=Path(args.prompts),
        parsed_path=Path(args.parsed),
        output_path=Path(args.output),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
