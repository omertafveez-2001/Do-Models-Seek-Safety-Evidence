from __future__ import annotations

import argparse
from pathlib import Path

from config import COUNTERFACTUAL_PROMPTS, PARSED_OUTPUTS, RAW_OUTPUTS
from io_utils import read_jsonl, reasoning_from_text, write_jsonl
from parsers import parse_decision


def parse_outputs(
    prompts_path: Path = COUNTERFACTUAL_PROMPTS,
    raw_path: Path = RAW_OUTPUTS,
) -> list[dict]:
    prompts = {row["counterfactual_id"]: row for row in read_jsonl(prompts_path)}
    raw_rows = read_jsonl(raw_path)
    parsed = []

    for row in raw_rows:
        prompt_row = prompts.get(row["counterfactual_id"], {})
        raw_output = row.get("raw_output", "")
        error = row.get("error")
        if error:
            parsed_decision = "ERROR"
            parse_status = "error"
        else:
            parsed_decision, parse_status = parse_decision(
                raw_output,
                prompt_row.get("original_decision", ""),
            )
        parsed.append({
            "counterfactual_id": row["counterfactual_id"],
            "base_case_id": row.get("base_case_id", prompt_row.get("base_case_id")),
            "model": row.get("model", prompt_row.get("model")),
            "bucket": row.get("bucket", prompt_row.get("bucket")),
            "edit_type": row.get("edit_type", prompt_row.get("edit_type")),
            "raw_output": raw_output,
            "parsed_decision": parsed_decision,
            "parsed_reasoning": reasoning_from_text(raw_output),
            "parse_status": parse_status,
            "error": error,
        })
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(COUNTERFACTUAL_PROMPTS))
    parser.add_argument("--raw", default=str(RAW_OUTPUTS))
    parser.add_argument("--output", default=str(PARSED_OUTPUTS))
    args = parser.parse_args()

    parsed = parse_outputs(Path(args.prompts), Path(args.raw))
    write_jsonl(Path(args.output), parsed)
    statuses = {}
    for row in parsed:
        statuses[row["parse_status"]] = statuses.get(row["parse_status"], 0) + 1
    print(f"Wrote {len(parsed)} parsed outputs to {args.output}")
    print("Parse statuses:", statuses)


if __name__ == "__main__":
    main()

