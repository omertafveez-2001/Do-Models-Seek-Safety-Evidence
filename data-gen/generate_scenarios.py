"""
Scenario Generator — Strategic Ignorance Experiment
====================================================
Generates all 900 scenarios for the strategic ignorance experiment
(300 offered_report, 300 report_discovery, and 300 evidence_discovery).

Steps:
1. Generates frozen domain contexts using Opus 4.8 (done once, saved to disk)
2. Assembles all 900 scenarios from the parameter matrix
3. Saves scenarios to scenarios.json ready for the experiment runner

Usage:
    export ANTHROPIC_API_KEY=your_key
    python data-gen/generate_scenarios.py
    python data-gen/generate_scenarios.py --scenarios-only   # skip domain context generation
"""
import os
import argparse
import json
from collections import Counter
from pathlib import Path
from scenario_dataset import generate_all_scenarios
from scenario_config import DOMAIN_CONTEXTS_FILE

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).parent.parent / ".env")


def _assert_balanced_counts(scenarios: list[dict], field: str, expected_count: int):
    counts = Counter(s[field] for s in scenarios)
    assert counts, f"No counts found for {field}"
    assert all(count == expected_count for count in counts.values()), (
        f"Unexpected {field} distribution: {counts}"
    )


def validate_scenarios(
    scenarios: list[dict],
    offered_report: list[dict],
    report_discovery: list[dict],
    evidence_discovery: list[dict],
):
    assert len(offered_report) == 300, f"Expected 300 offered_report scenarios, got {len(offered_report)}"
    assert len(report_discovery) == 300, f"Expected 300 report_discovery scenarios, got {len(report_discovery)}"
    assert len(evidence_discovery) == 300, (
        f"Expected 300 evidence_discovery scenarios, got {len(evidence_discovery)}"
    )
    assert len({s["scenario_id"] for s in scenarios}) == len(scenarios), "Scenario IDs must be unique"
    assert all("Decision:" in s["prompt"] for s in scenarios), "Every prompt must include 'Decision:'"
    assert all("Decision: INSPECT or SKIP" in s["prompt"] for s in scenarios), (
        "Every prompt must include the exact response format"
    )
    assert all("post_inspection_prompt" in s for s in scenarios), "Every scenario must include post_inspection_prompt"
    assert all("issue_description" in s for s in scenarios), "Every scenario must include issue_description"

    for variant_name, variant_scenarios in [
        ("offered_report", offered_report),
        ("report_discovery", report_discovery),
        ("evidence_discovery", evidence_discovery),
    ]:
        _assert_balanced_counts(variant_scenarios, "domain", 60)
        _assert_balanced_counts(variant_scenarios, "cost_label", 60)
        _assert_balanced_counts(variant_scenarios, "severity_label", 75)
        _assert_balanced_counts(variant_scenarios, "probability_label", 100)
        print(f"{variant_name.title()} distribution checks passed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios-only",
        action="store_true",
        help="skip domain context generation and load from disk (domain_contexts.json must exist)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Strategic Ignorance Experiment — Scenario Generator")
    print("=" * 60)

    client = None
    contexts = None
    if args.scenarios_only:
        if not DOMAIN_CONTEXTS_FILE.exists():
            raise FileNotFoundError(
                f"--scenarios-only requires {DOMAIN_CONTEXTS_FILE} to exist. "
                "Run without --scenarios-only first."
            )
        print("--scenarios-only: loading domain contexts from disk...")
        with open(DOMAIN_CONTEXTS_FILE) as f:
            contexts = json.load(f)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    scenarios = generate_all_scenarios(client, contexts=contexts)

    # Split scenario variants
    offered_report = [s for s in scenarios if s["variant"] == "offered_report"]
    report_discovery = [s for s in scenarios if s["variant"] == "report_discovery"]
    evidence_discovery = [s for s in scenarios if s["variant"] == "evidence_discovery"]

    validate_scenarios(scenarios, offered_report, report_discovery, evidence_discovery)

    print(f"\nGenerated {len(offered_report)} offered_report scenarios")
    print(f"Generated {len(report_discovery)} report_discovery scenarios")
    print(f"Generated {len(evidence_discovery)} evidence_discovery scenarios")
    print(f"Total: {len(scenarios)} scenarios\n")

    output_path = Path(__file__).parent / "data" / "scenarios.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Saved to {output_path}")
    print("\nSample scenario (offered_report, healthcare, free, high, high):")
    print("-" * 60)
    sample = next(
        s for s in offered_report
        if s["domain"] == "healthcare"
        and s["cost_label"] == "free"
        and s["severity_label"] == "high"
        and s["probability_label"] == "high"
    )
    print(sample["prompt"])
    print("-" * 60)
    print("\nScenario generator complete. Run experiment_runner.py next.")

if __name__ == "__main__":
    main()
