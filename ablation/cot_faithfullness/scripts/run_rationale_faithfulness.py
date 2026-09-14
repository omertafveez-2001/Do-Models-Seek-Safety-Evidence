from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODELS,
    COUNTERFACTUAL_PROMPTS,
    OPENAI_MAX_COMPLETION_TOKENS,
    OPENAI_MODELS,
    OPENAI_REASONING_MAX_COMPLETION_TOKENS,
    OPENAI_REASONING_MODELS,
    RAW_OUTPUTS,
)
from io_utils import append_jsonl, read_jsonl

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _call_openai(client, model_id: str, prompt: str) -> tuple[str, dict]:
    is_reasoning = model_id in OPENAI_REASONING_MODELS
    max_tokens = OPENAI_REASONING_MAX_COMPLETION_TOKENS if is_reasoning else OPENAI_MAX_COMPLETION_TOKENS
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    reasoning_effort = "low" if is_reasoning else None
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
        "completion_tokens": getattr(response.usage, "completion_tokens", None),
        "total_tokens": getattr(response.usage, "total_tokens", None),
    }
    config = {
        "model_id": model_id,
        "temperature_requested": None,
        "temperature_sent": None,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "seed": None,
    }
    return text.strip(), {"usage": usage, "generation_config": config}


def _call_anthropic(client, model_id: str, prompt: str) -> tuple[str, dict]:
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
    text = response.content[0].text if response.content else ""
    usage = {
        "prompt_tokens": getattr(response.usage, "input_tokens", None),
        "completion_tokens": getattr(response.usage, "output_tokens", None),
        "total_tokens": None,
    }
    config = {
        "model_id": model_id,
        "temperature_requested": None,
        "temperature_sent": None,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "reasoning_effort": None,
        "seed": None,
    }
    return text.strip(), {"usage": usage, "generation_config": config}


def _client_for_model(model: str):
    if model in OPENAI_MODELS:
        from openai import OpenAI

        return "openai", OPENAI_MODELS[model], OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model in ANTHROPIC_MODELS:
        import anthropic

        return "anthropic", ANTHROPIC_MODELS[model], anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unknown model: {model}")


def run_counterfactuals(
    prompts_path: Path = COUNTERFACTUAL_PROMPTS,
    output_path: Path = RAW_OUTPUTS,
    limit: int | None = None,
    sleep_seconds: float = 0.3,
):
    prompts = read_jsonl(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    existing = read_jsonl(output_path)
    done = {row["counterfactual_id"] for row in existing if row.get("counterfactual_id")}

    total = len(prompts)
    remaining = sum(1 for row in prompts if row["counterfactual_id"] not in done)
    print("=" * 72)
    print("Running rationale-faithfulness counterfactuals")
    print("=" * 72)
    print(f"Prompts:   {prompts_path}")
    print(f"Output:    {output_path}")
    print(f"Total:     {total}")
    print(f"Remaining: {remaining}")
    print("=" * 72)

    call_idx = total - remaining
    clients = {}
    for row in prompts:
        if row["counterfactual_id"] in done:
            continue
        call_idx += 1
        model = row["model"]
        if model not in clients:
            clients[model] = _client_for_model(model)
        provider, model_id, client = clients[model]
        print(f"[{call_idx}/{total}] {model} {row['edit_type']} {row['counterfactual_id']}", end=" ... ", flush=True)
        out = {
            "counterfactual_id": row["counterfactual_id"],
            "base_case_id": row["base_case_id"],
            "model": model,
            "model_id": model_id,
            "model_provider": provider,
            "bucket": row["bucket"],
            "edit_type": row["edit_type"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if provider == "anthropic":
                raw_output, metadata = _call_anthropic(client, model_id, row["counterfactual_prompt"])
            else:
                raw_output, metadata = _call_openai(client, model_id, row["counterfactual_prompt"])
            out.update({
                "raw_output": raw_output,
                "error": None,
                **metadata,
            })
            print("ok")
        except Exception as exc:
            out.update({
                "raw_output": "",
                "error": str(exc),
                "usage": {},
                "generation_config": {"model_id": model_id},
            })
            print("ERROR")
        append_jsonl(output_path, out)
        time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(COUNTERFACTUAL_PROMPTS))
    parser.add_argument("--output", default=str(RAW_OUTPUTS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()
    run_counterfactuals(
        prompts_path=Path(args.prompts),
        output_path=Path(args.output),
        limit=args.limit,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
