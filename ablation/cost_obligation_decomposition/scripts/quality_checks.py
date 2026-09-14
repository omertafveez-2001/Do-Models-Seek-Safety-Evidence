from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .config import COST_OBLIGATION_CONDITIONS, DEFAULT_RESULTS_DIR, DEFAULT_SELECTION_FILE
    from .prompts import render_cost_obligation_prompt
except ImportError:
    from config import COST_OBLIGATION_CONDITIONS, DEFAULT_RESULTS_DIR, DEFAULT_SELECTION_FILE
    from prompts import render_cost_obligation_prompt


def check_selection_file(path: Path = DEFAULT_SELECTION_FILE) -> list[str]:
    errors = []
    with open(path) as fp:
        payload = json.load(fp)
    scenarios = payload["scenarios"]
    ids = [s["scenario_id"] for s in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("scenario IDs are not unique")

    for scenario in scenarios:
        prompts = {
            condition: render_cost_obligation_prompt(scenario, condition)
            for condition in COST_OBLIGATION_CONDITIONS
        }
        if "will trigger mandatory review" in prompts["cost_only"].lower():
            errors.append(f"{scenario['scenario_id']}: cost_only contains positive mandatory-review text")
        for condition in ("free_no_obligation", "cost_only"):
            if "required to address" in prompts[condition].lower():
                errors.append(f"{scenario['scenario_id']}: {condition} contains obligation text")
        for condition in ("free_no_obligation", "obligation_only", "obligation_only_no_cancel"):
            if "costs" in prompts[condition].lower():
                errors.append(f"{scenario['scenario_id']}: {condition} contains retrieval cost text")
        for condition in ("obligation_only_no_cancel", "cost_plus_obligation_no_cancel"):
            lower = prompts[condition].lower()
            if "delay or cancel the project" in lower or "may delay or cancel the project" in lower:
                errors.append(f"{scenario['scenario_id']}: {condition} contains explicit cancel/delay consequence text")
        for condition in ("obligation_only", "cost_plus_obligation"):
            lower = prompts[condition].lower()
            if "required to address" not in lower:
                errors.append(f"{scenario['scenario_id']}: {condition} is missing mandatory obligation text")
            if "delay or cancel the project" not in lower and "may delay or cancel the project" not in lower:
                errors.append(f"{scenario['scenario_id']}: {condition} is missing explicit cancel/delay consequence text")
        for condition, prompt in prompts.items():
            if "Decision: INSPECT or SKIP" not in prompt:
                errors.append(f"{scenario['scenario_id']} {condition}: missing strict decision format")
    return errors


def check_results_dir(path: Path = DEFAULT_RESULTS_DIR) -> list[str]:
    errors = []
    if not path.exists():
        return [f"{path} does not exist"]

    files = sorted(path.glob("*_cost_obligation_decomposition.json"))
    if not files:
        return [f"No *_cost_obligation_decomposition.json files found in {path}"]

    rows = []
    for file in files:
        with open(file) as fp:
            model_rows = json.load(fp)
        if not isinstance(model_rows, list):
            errors.append(f"{file} is not a JSON list")
            continue
        rows.extend(model_rows)

    if not rows:
        return [f"No result rows found in {path}"]

    condition_counts = Counter(row["condition"] for row in rows)
    missing_conditions = set(COST_OBLIGATION_CONDITIONS) - set(condition_counts)
    if missing_conditions:
        errors.append(f"missing conditions: {sorted(missing_conditions)}")

    unclear_count = sum(1 for row in rows if row["decision"] == "UNCLEAR")
    if unclear_count / len(rows) > 0.02:
        errors.append(f"UNCLEAR rate exceeds 2%: {unclear_count / len(rows):.1%}")

    if any(not row.get("prompt") for row in rows):
        errors.append("some rows are missing prompts")
    if any(not row.get("raw_response") and not row.get("error") for row in rows):
        errors.append("some rows are missing responses and errors")

    balance = Counter((row["model"], row["condition"]) for row in rows)
    if len(set(balance.values())) > 1:
        errors.append(f"model/condition counts are imbalanced: {dict(balance)}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-file", default=str(DEFAULT_SELECTION_FILE))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()

    errors = check_selection_file(Path(args.selection_file))
    if args.check_results:
        errors.extend(check_results_dir(Path(args.results_dir)))

    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
