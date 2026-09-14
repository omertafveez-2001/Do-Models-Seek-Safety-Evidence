"""
Experiment Runner — Strategic Ignorance Step 2
==============================================
Runs the full scenario matrix against a subject model.
5 runs per scenario.

Usage (from project root):
    python main_experiment/scripts/experiment_runner.py --model sonnet --variant offered_report
    python main_experiment/scripts/experiment_runner.py --model sonnet --variant report_discovery
    python main_experiment/scripts/experiment_runner.py --model sonnet --variant evidence_discovery
    python main_experiment/scripts/experiment_runner.py --model opus   --variant offered_report

Optional:
    --scenarios <path>   path to scenarios.json  (default: data-gen/data/scenarios.json)
    --runs <n>           runs per scenario        (default: 5)
    --save-path <path>   output file path         (default: main_experiment/runs/{variant}/{model}_{variant}.json)

Runs saved to the default variant path above (or --save-path)
"""

from __future__ import annotations

import os
import json
import time
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_experiment.scripts.parsers import parse_initial_decision, parse_post_decision

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-8",
}

OPENAI_MODELS = {
    "gpt-5.5": "gpt-5.5-2026-04-23",
    "o3":      "o3-2025-04-16",
}

OPENAI_REASONING_MODELS = {"o3-2025-04-16", "o1", "o1-mini"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_RUNS = 5
SLEEP        = 0.5
ANTHROPIC_MAX_TOKENS = 1024
OPENAI_MAX_COMPLETION_TOKENS = 1024
OPENAI_REASONING_MAX_COMPLETION_TOKENS = 2048

# ---------------------------------------------------------------------------
# Per-run result builder
# ---------------------------------------------------------------------------

def _metadata(
    scenario: dict,
    run_idx: int,
    model_id: str,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict:
    return {
        "scenario_id":        scenario["scenario_id"],
        "run_idx":            run_idx,
        "variant":            scenario["variant"],
        "domain":             scenario["domain"],
        "cost_label":         scenario["cost_label"],
        "severity_label":     scenario["severity_label"],
        "probability_label":  scenario["probability_label"],
        "model_id":           model_id,
        "max_tokens":         max_tokens,
        "reasoning_effort":   reasoning_effort,
    }


def _error_result(
    scenario: dict,
    run_idx: int,
    err: Exception,
    model_id: str,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict:
    return {
        **_metadata(scenario, run_idx, model_id, max_tokens, reasoning_effort),
        "decision":                 "ERROR",
        "response":                 str(err),
        "post_inspection_decision": None,
        "post_inspection_response": None,
        "input_tokens":             0,
        "output_tokens":            0,
    }


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def run_anthropic(client, model_id: str, scenario: dict, run_idx: int) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": scenario["prompt"],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    max_tokens = ANTHROPIC_MAX_TOKENS
    reasoning_effort = None

    def _call(msgs):
        return client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=msgs,
        )

    try:
        r1 = _call(messages)
        text1    = r1.content[0].text.strip()
        decision = parse_initial_decision(text1)
        in_tok   = r1.usage.input_tokens
        out_tok  = r1.usage.output_tokens

        post_decision = None
        post_text     = None

        if decision == "INSPECT":
            messages.append({"role": "assistant", "content": text1})
            messages.append({"role": "user", "content": scenario["post_inspection_prompt"]})

            r2 = _call(messages)
            post_text     = r2.content[0].text.strip()
            post_decision = parse_post_decision(post_text)
            in_tok  += r2.usage.input_tokens
            out_tok += r2.usage.output_tokens

        return {
            **_metadata(scenario, run_idx, model_id, max_tokens, reasoning_effort),
            "decision":                 decision,
            "response":                 text1,
            "post_inspection_decision": post_decision,
            "post_inspection_response": post_text,
            "input_tokens":             in_tok,
            "output_tokens":            out_tok,
        }

    except Exception as e:
        return _error_result(scenario, run_idx, e, model_id, max_tokens, reasoning_effort)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def run_openai(client, model_id: str, scenario: dict, run_idx: int) -> dict:
    messages = [{"role": "user", "content": scenario["prompt"]}]
    is_reasoning = model_id in OPENAI_REASONING_MODELS
    max_tokens = (
        OPENAI_REASONING_MAX_COMPLETION_TOKENS
        if is_reasoning
        else OPENAI_MAX_COMPLETION_TOKENS
    )
    reasoning_effort = "low" if is_reasoning else None

    def _call(msgs):
        kwargs = {"model": model_id, "messages": msgs}
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["max_completion_tokens"] = max_tokens
        return client.chat.completions.create(**kwargs)

    try:
        r1    = _call(messages)
        text1 = r1.choices[0].message.content.strip()
        decision = parse_initial_decision(text1)
        in_tok   = r1.usage.prompt_tokens
        out_tok  = r1.usage.completion_tokens

        post_decision = None
        post_text     = None

        if decision == "INSPECT":
            messages.append({"role": "assistant", "content": text1})
            messages.append({"role": "user", "content": scenario["post_inspection_prompt"]})

            r2        = _call(messages)
            post_text = r2.choices[0].message.content.strip()
            post_decision = parse_post_decision(post_text)
            in_tok  += r2.usage.prompt_tokens
            out_tok += r2.usage.completion_tokens

        return {
            **_metadata(scenario, run_idx, model_id, max_tokens, reasoning_effort),
            "decision":                 decision,
            "response":                 text1,
            "post_inspection_decision": post_decision,
            "post_inspection_response": post_text,
            "input_tokens":             in_tok,
            "output_tokens":            out_tok,
        }

    except Exception as e:
        return _error_result(scenario, run_idx, e, model_id, max_tokens, reasoning_effort)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_model_keys = list(ANTHROPIC_MODELS) + list(OPENAI_MODELS)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=all_model_keys)
    parser.add_argument(
        "--variant",
        required=True,
        choices=["offered_report", "report_discovery", "evidence_discovery"],
    )
    parser.add_argument("--scenarios", default=str(Path(__file__).resolve().parents[2] / "data-gen" / "data" / "scenarios.json"))
    parser.add_argument("--runs",      type=int, default=DEFAULT_RUNS)
    parser.add_argument("--save-path", default=None, help="override output file path")
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        raise FileNotFoundError(
            f"{scenarios_path} not found — run data-gen/generate_scenarios.py first"
        )

    with open(scenarios_path) as f:
        all_scenarios = json.load(f)

    scenarios = [s for s in all_scenarios if s["variant"] == args.variant]

    is_openai = args.model in OPENAI_MODELS
    if is_openai:
        from openai import OpenAI
        model_id = OPENAI_MODELS[args.model]
        client   = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        run_fn   = lambda s, i: run_openai(client, model_id, s, i)
    else:
        import anthropic
        model_id = ANTHROPIC_MODELS[args.model]
        client   = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        run_fn   = lambda s, i: run_anthropic(client, model_id, s, i)

    if args.save_path:
        results_file = Path(args.save_path)
        results_dir  = results_file.parent
    else:
        results_dir  = Path(__file__).resolve().parent.parent / "runs" / args.variant
        results_file = results_dir / f"{args.model}_{args.variant}.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-completed (scenario_id, run_idx) pairs
    existing = []
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)
    done_keys = {(r["scenario_id"], r["run_idx"]) for r in existing}
    results   = list(existing)

    total     = len(scenarios) * args.runs
    remaining = total - len(done_keys)

    print("=" * 60)
    print("Strategic Ignorance — Experiment Runner")
    print("=" * 60)
    print(f"Model:      {model_id}")
    print(f"Variant:    {args.variant}")
    print(f"Scenarios:  {len(scenarios)}")
    print(f"Runs each:  {args.runs}")
    print(f"Total:      {total}  (remaining: {remaining})")
    print("=" * 60)

    call_idx = len(done_keys)
    for scenario in scenarios:
        for run_idx in range(args.runs):
            if (scenario["scenario_id"], run_idx) in done_keys:
                continue

            call_idx += 1
            print(f"[{call_idx}/{total}] {scenario['scenario_id']} run={run_idx + 1}", end=" ... ", flush=True)

            result = run_fn(scenario, run_idx)
            results.append(result)

            label = result["decision"]
            if result["post_inspection_decision"]:
                label += f" → {result['post_inspection_decision']}"
            print(label)

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            time.sleep(SLEEP)

    print(f"\nDone. {len(results)} total runs saved to {results_file}")

    decisions = [r["decision"] for r in results]
    n = len(decisions)
    print(f"\nSummary ({args.model}):")
    for label in ["INSPECT", "SKIP", "UNCLEAR", "ERROR"]:
        count = decisions.count(label)
        print(f"  {label:<10} {count:>4}  ({count / n * 100:.1f}%)")


if __name__ == "__main__":
    main()
