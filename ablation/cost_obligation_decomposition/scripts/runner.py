from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .config import (
        ANTHROPIC_MAX_TOKENS,
        ANTHROPIC_MODELS,
        CONDITIONS,
        COST_OBLIGATION_CONDITIONS,
        DEFAULT_RESULTS_DIR,
        DEFAULT_RUNS,
        DEFAULT_SELECTION_FILE,
        DEFAULT_SLEEP_SECONDS,
        EXPERIMENT_ID,
        MODEL_ORDER,
        OPENAI_MAX_COMPLETION_TOKENS,
        OPENAI_MODELS,
        OPENAI_REASONING_MAX_COMPLETION_TOKENS,
        OPENAI_REASONING_MODELS,
        PROJECT_ROOT,
        VARIANT,
    )
    from .prompts import render_cost_obligation_prompt
except ImportError:
    from config import (
        ANTHROPIC_MAX_TOKENS,
        ANTHROPIC_MODELS,
        CONDITIONS,
        COST_OBLIGATION_CONDITIONS,
        DEFAULT_RESULTS_DIR,
        DEFAULT_RUNS,
        DEFAULT_SELECTION_FILE,
        DEFAULT_SLEEP_SECONDS,
        EXPERIMENT_ID,
        MODEL_ORDER,
        OPENAI_MAX_COMPLETION_TOKENS,
        OPENAI_MODELS,
        OPENAI_REASONING_MAX_COMPLETION_TOKENS,
        OPENAI_REASONING_MODELS,
        PROJECT_ROOT,
        VARIANT,
    )
    from prompts import render_cost_obligation_prompt

ROOT = PROJECT_ROOT
PROJECT_SOURCE_DIR = ROOT
if str(PROJECT_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from main_experiment.scripts.parsers import parse_initial_decision  # noqa: E402

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def load_selected_scenarios(path: Path) -> list[dict]:
    with open(path) as fp:
        payload = json.load(fp)
    return payload["scenarios"] if isinstance(payload, dict) and "scenarios" in payload else payload


def _reasoning_from_response(text: str) -> str:
    marker = "Reasoning:"
    idx = text.lower().find(marker.lower())
    if idx == -1:
        return ""
    return text[idx + len(marker):].strip()


def _generation_config(provider: str, model_id: str, max_tokens: int, reasoning_effort: str | None) -> dict:
    return {
        "provider": provider,
        "model_id": model_id,
        "temperature_requested": None,
        "temperature_sent": None,
        "max_tokens": max_tokens if provider == "anthropic" else None,
        "max_completion_tokens": max_tokens if provider == "openai" else None,
        "reasoning_effort": reasoning_effort,
        "seed": None,
        "run_post_inspection": False,
    }


def _base_result(
    *,
    scenario: dict,
    condition_name: str,
    model: str,
    model_id: str,
    provider: str,
    rollout: int,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict:
    condition = CONDITIONS[condition_name]
    return {
        "experiment": EXPERIMENT_ID,
        "variant": VARIANT,
        "condition": condition.name,
        "condition_display_name": condition.display_name,
        "model": model,
        "model_id": model_id,
        "model_provider": provider,
        "scenario_id": scenario["scenario_id"],
        "domain": scenario["domain"],
        "role": scenario["role"],
        "system": scenario["system"],
        "context": scenario["context"],
        "project_value": scenario["project_value"],
        "failure_cost": scenario["failure_cost"],
        "probability": scenario["probability"],
        "probability_level": scenario["probability_level"],
        "severity": scenario["severity"],
        "severity_level": scenario["severity_level"],
        "severity_description": scenario["severity_description"],
        "retrieval_cost_present": condition.retrieval_cost_present,
        "obligation_present": condition.obligation_present,
        "retrieval_cost_level": condition.retrieval_cost_level,
        "obligation_variant": condition.obligation_variant,
        "cancellation_stated": condition.cancellation_stated,
        "inspection_cost_points": scenario["decomp_inspection_cost_points"] if condition.retrieval_cost_present else "",
        "rollout": rollout,
        "prompt": prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": _generation_config(provider, model_id, max_tokens, reasoning_effort),
        "scenario_source": scenario.get("scenario_source", ""),
        "selection_reason": scenario.get("selection_reason", ""),
        "offered_report_baseline_inspect_fraction": scenario.get("offered_report_baseline_inspect_fraction", ""),
        "report_discovery_inspect_fraction": scenario.get("report_discovery_inspect_fraction", ""),
        "evidence_discovery_inspect_fraction": scenario.get("evidence_discovery_inspect_fraction", ""),
    }


def _result_path(results_dir: Path, model: str) -> Path:
    return results_dir / f"{model}_{EXPERIMENT_ID}.json"


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fp:
        return json.load(fp)


def _write_results(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(rows, fp, indent=2)


def _done_keys(results_by_model: dict[str, list[dict]]) -> set[tuple[str, str, str, int]]:
    return {
        (row["model"], row["scenario_id"], row["condition"], int(row["rollout"]))
        for rows in results_by_model.values()
        for row in rows
        if row.get("model") and row.get("scenario_id") and row.get("condition") and row.get("rollout") is not None
    }


def _call_anthropic(client, model_id: str, prompt: str):
    response = client.messages.create(
        model=model_id,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )
    return response.content[0].text.strip(), response.usage.input_tokens, response.usage.output_tokens


def _call_openai(client, model_id: str, prompt: str):
    is_reasoning = model_id in OPENAI_REASONING_MODELS
    max_tokens = OPENAI_REASONING_MAX_COMPLETION_TOKENS if is_reasoning else OPENAI_MAX_COMPLETION_TOKENS
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    if is_reasoning:
        kwargs["reasoning_effort"] = "low"
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    return text.strip(), response.usage.prompt_tokens, response.usage.completion_tokens


def run_one(client, provider: str, model: str, model_id: str, scenario: dict, condition: str, rollout: int) -> dict:
    prompt = render_cost_obligation_prompt(scenario, condition)
    is_reasoning = model_id in OPENAI_REASONING_MODELS
    max_tokens = (
        ANTHROPIC_MAX_TOKENS if provider == "anthropic"
        else OPENAI_REASONING_MAX_COMPLETION_TOKENS if is_reasoning
        else OPENAI_MAX_COMPLETION_TOKENS
    )
    reasoning_effort = "low" if provider == "openai" and is_reasoning else None
    row = _base_result(
        scenario=scenario,
        condition_name=condition,
        model=model,
        model_id=model_id,
        provider=provider,
        rollout=rollout,
        prompt=prompt,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )

    try:
        if provider == "anthropic":
            raw_response, input_tokens, output_tokens = _call_anthropic(client, model_id, prompt)
        else:
            raw_response, input_tokens, output_tokens = _call_openai(client, model_id, prompt)
        decision, parse_method = parse_initial_decision(raw_response, return_method=True)
        row.update({
            "raw_response": raw_response,
            "decision": decision,
            "reasoning": _reasoning_from_response(raw_response),
            "parsed_ok": decision in {"INSPECT", "SKIP"},
            "parse_method": parse_method,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": "",
        })
    except Exception as exc:
        row.update({
            "raw_response": "",
            "decision": "ERROR",
            "reasoning": "",
            "parsed_ok": False,
            "parse_method": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(exc),
        })
    return row


def _client_for_model(model: str):
    if model in ANTHROPIC_MODELS:
        import anthropic
        return "anthropic", ANTHROPIC_MODELS[model], anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if model in OPENAI_MODELS:
        from openai import OpenAI
        return "openai", OPENAI_MODELS[model], OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    raise ValueError(f"Unknown model: {model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-file", default=str(DEFAULT_SELECTION_FILE))
    parser.add_argument(
        "--save-path",
        default=str(DEFAULT_RESULTS_DIR),
        help="output directory for per-model JSON files",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=COST_OBLIGATION_CONDITIONS, default=COST_OBLIGATION_CONDITIONS)
    parser.add_argument("--smoke-test", action="store_true", help="run 2 scenarios x all selected conditions x 1 rollout")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    args = parser.parse_args()

    scenarios = load_selected_scenarios(Path(args.selection_file))
    if args.limit_scenarios is not None:
        scenarios = scenarios[:args.limit_scenarios]
    if args.smoke_test:
        scenarios = scenarios[:2]
        args.runs = 1

    results_dir = Path(args.save_path)
    results_by_model = {
        model: _load_results(_result_path(results_dir, model))
        for model in args.models
    }
    done = _done_keys(results_by_model)
    total = len(args.models) * len(scenarios) * len(args.conditions) * args.runs
    remaining = total - sum(
        1
        for model in args.models
        for scenario in scenarios
        for condition in args.conditions
        for rollout in range(args.runs)
        if (model, scenario["scenario_id"], condition, rollout) in done
    )

    print("=" * 72)
    print("Cost/Obligation Decomposition Runner")
    print("=" * 72)
    print(f"Selection file: {args.selection_file}")
    print(f"Results dir:    {results_dir}")
    print(f"Models:         {', '.join(args.models)}")
    print(f"Scenarios:      {len(scenarios)}")
    print(f"Conditions:     {len(args.conditions)}")
    print(f"Runs:           {args.runs}")
    print(f"Total rows:     {total} (remaining: {remaining})")
    print("=" * 72)

    call_idx = total - remaining
    clients = {}
    for model in args.models:
        provider, model_id, client = _client_for_model(model)
        clients[model] = (provider, model_id, client)

    for model in args.models:
        provider, model_id, client = clients[model]
        for scenario in scenarios:
            for condition in args.conditions:
                for rollout in range(args.runs):
                    key = (model, scenario["scenario_id"], condition, rollout)
                    if key in done:
                        continue
                    call_idx += 1
                    print(f"[{call_idx}/{total}] {model} {scenario['scenario_id']} {condition} rollout={rollout + 1}", end=" ... ", flush=True)
                    row = run_one(client, provider, model, model_id, scenario, condition, rollout)
                    results_by_model[model].append(row)
                    _write_results(_result_path(results_dir, model), results_by_model[model])
                    print(row["decision"])
                    time.sleep(args.sleep)

    print(f"Saved per-model JSON results to {results_dir}")


if __name__ == "__main__":
    main()
