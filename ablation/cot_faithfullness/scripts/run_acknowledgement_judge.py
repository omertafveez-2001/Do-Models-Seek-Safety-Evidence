from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from acknowledgement import build_acknowledgement_judge_prompt, parse_acknowledgement_judge_output
from config import (
    ACKNOWLEDGEMENT_JUDGE_BATCH_SIZE,
    ACKNOWLEDGEMENT_JUDGE_MAX_COMPLETION_TOKENS,
    ACKNOWLEDGEMENT_JUDGE_MODEL,
    ACKNOWLEDGEMENT_JUDGE_MODEL_ID,
    COUNTERFACTUAL_PROMPTS,
    LLM_JUDGE_ACKNOWLEDGEMENT_OUTPUTS,
    PARSED_OUTPUTS,
)
from io_utils import append_jsonl, read_jsonl

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")


GEMINI_BATCH_POLL_SECONDS = 300
GEMINI_BATCH_BACKOFF_SECONDS = [120, 240, 480, 900]


def batched(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _compact_json(obj, limit: int = 300) -> str:
    try:
        text = json.dumps(obj, sort_keys=True)
    except Exception:
        text = str(obj)
    return text if len(text) <= limit else text[:limit] + "..."


def _is_resource_exhausted_error(message: object) -> bool:
    text = str(message)
    return "429" in text and "RESOURCE_EXHAUSTED" in text


def _jsonl_request_path(tag: str) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix=f"gemini_batch_{tag}_", suffix=".jsonl")
    os.close(fd)
    return Path(raw_path)


def _batch_state_name(state) -> str:
    if state is None:
        return "UNKNOWN"
    return str(state)


def _normalized_batch_state_name(state) -> str:
    state_name = _batch_state_name(state)
    if "." in state_name:
        state_name = state_name.split(".")[-1]
    return state_name


def _is_terminal_batch_state(state_name: str) -> bool:
    return state_name in {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PAUSED",
        "JOB_STATE_EXPIRED",
        "BATCH_STATE_SUCCEEDED",
        "BATCH_STATE_FAILED",
        "BATCH_STATE_CANCELLED",
        "BATCH_STATE_EXPIRED",
    }


def _to_plain_data(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_data(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain_data(v) for v in obj]
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain_data(model_dump())
        except Exception:
            pass
    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        try:
            return _to_plain_data(dict_method())
        except Exception:
            pass
    to_json_dict = getattr(obj, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            return _to_plain_data(to_json_dict())
        except Exception:
            pass
    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict) and obj_dict:
        return {k: _to_plain_data(v) for k, v in obj_dict.items() if not k.startswith("_")}
    return obj


def _find_first_responses_file(obj) -> str | None:
    if isinstance(obj, dict):
        for key in ("responsesFile", "responses_file", "file_name", "fileName"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
        for value in obj.values():
            found = _find_first_responses_file(value)
            if found:
                return found
        return None
    if isinstance(obj, list):
        for value in obj:
            found = _find_first_responses_file(value)
            if found:
                return found
        return None
    return None


def _extract_output_file_name(batch_job) -> str | None:
    return _find_first_responses_file(_to_plain_data(batch_job))


def _write_debug_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_plain_data(obj), f, indent=2)


def _write_debug_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _extract_response_text(response_obj: dict) -> str:
    if not isinstance(response_obj, dict):
        raise ValueError("Batch response payload is not an object.")

    direct_text = response_obj.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    prompt_feedback = response_obj.get("promptFeedback") or response_obj.get("prompt_feedback") or {}
    block_reason = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")
    block_message = prompt_feedback.get("blockReasonMessage") or prompt_feedback.get("block_reason_message")
    if block_reason or block_message:
        detail = block_message or block_reason
        raise ValueError(f"Batch response blocked before text generation: {detail}")

    candidates = response_obj.get("candidates") or []
    if not candidates:
        raise ValueError(f"Batch response is missing candidates. raw={_compact_json(response_obj)}")

    texts = []
    finish_reasons = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
        if finish_reason:
            finish_reasons.append(str(finish_reason))
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])

    text = "".join(texts).strip()
    if not text:
        raise ValueError(
            "Batch response candidate did not include text output. "
            f"finish_reasons={finish_reasons or ['UNKNOWN']} raw={_compact_json(response_obj)}"
        )
    return text


def _build_gemini_batch_request(prompt: str, row_id: str) -> dict:
    return {
        "key": row_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": ACKNOWLEDGEMENT_JUDGE_MAX_COMPLETION_TOKENS,
            },
        },
    }


def _run_single_gemini_batch_job(
    client,
    model: str,
    requests: list[dict],
    display_name: str,
    debug_batch_dir: Path | None = None,
) -> dict[str, str]:
    request_file = _jsonl_request_path(display_name.replace("/", "_"))
    try:
        with open(request_file, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        uploaded_file = client.files.upload(
            file=str(request_file),
            config={
                "display_name": display_name,
                "mime_type": "jsonl",
            },
        )
        uploaded_name = getattr(uploaded_file, "name", None) or uploaded_file["name"]

        batch_job = client.batches.create(
            model=model,
            src=uploaded_name,
            config={"display_name": display_name},
        )
        batch_name = getattr(batch_job, "name", None) or batch_job["name"]
        print(f"    batch job: {batch_name}", flush=True)
        if debug_batch_dir is not None:
            _write_debug_json(debug_batch_dir / f"{display_name}__created_batch_job.json", batch_job)

        while True:
            batch_job = client.batches.get(name=batch_name)
            raw_state_name = _batch_state_name(getattr(batch_job, "state", None))
            state_name = _normalized_batch_state_name(getattr(batch_job, "state", None))
            print(f"    state: {raw_state_name}", flush=True)
            if _is_terminal_batch_state(state_name):
                break
            time.sleep(GEMINI_BATCH_POLL_SECONDS)

        if debug_batch_dir is not None:
            _write_debug_json(debug_batch_dir / f"{display_name}__final_batch_job.json", batch_job)

        if state_name not in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
            raise RuntimeError(f"Gemini batch job did not succeed: {state_name}")

        output_file_name = _extract_output_file_name(batch_job)
        if not output_file_name:
            raise RuntimeError("Gemini batch job succeeded but no output file was returned.")

        raw_output = client.files.download(file=output_file_name).decode("utf-8")
        if debug_batch_dir is not None:
            _write_debug_text(debug_batch_dir / f"{display_name}__raw_output.jsonl", raw_output)
        results_by_key: dict[str, str] = {}
        for line in raw_output.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            key = payload.get("key")
            if key is None:
                raise ValueError("Gemini batch output line is missing key.")

            if "response" in payload:
                results_by_key[key] = _extract_response_text(payload["response"])
                continue

            if "error" in payload:
                results_by_key[key] = f"ERROR:{payload['error']}"
                continue

            if "status" in payload:
                results_by_key[key] = f"ERROR:{payload['status']}"
                continue

            results_by_key[key] = _extract_response_text(payload)

        if results_by_key and all(
            isinstance(value, str) and _is_resource_exhausted_error(value)
            for value in results_by_key.values()
        ):
            raise RuntimeError(next(iter(results_by_key.values())))

        return results_by_key
    finally:
        request_file.unlink(missing_ok=True)


def _run_gemini_batch_job(
    client,
    model: str,
    requests: list[dict],
    display_name: str,
    debug_batch_dir: Path | None = None,
) -> dict[str, str]:
    attempts = len(GEMINI_BATCH_BACKOFF_SECONDS) + 1
    for attempt in range(attempts):
        try:
            return _run_single_gemini_batch_job(client, model, requests, display_name, debug_batch_dir=debug_batch_dir)
        except Exception as exc:
            if not _is_resource_exhausted_error(exc):
                raise
            if attempt >= len(GEMINI_BATCH_BACKOFF_SECONDS):
                raise
            wait_seconds = GEMINI_BATCH_BACKOFF_SECONDS[attempt]
            wait_minutes = wait_seconds // 60
            print(f"    ERROR 429 RESOURCE_EXHAUSTED...resubmitting after {wait_minutes} minutes", flush=True)
            time.sleep(wait_seconds)


def run_acknowledgement_judge(
    prompts_path: Path = COUNTERFACTUAL_PROMPTS,
    parsed_path: Path = PARSED_OUTPUTS,
    output_path: Path = LLM_JUDGE_ACKNOWLEDGEMENT_OUTPUTS,
    limit: int | None = None,
    sleep_seconds: float = 0.3,
    debug_batch_dir: Path | None = None,
):
    from google import genai

    prompts = {row["counterfactual_id"]: row for row in read_jsonl(prompts_path)}
    parsed_rows = read_jsonl(parsed_path)
    if limit is not None:
        parsed_rows = parsed_rows[:limit]

    existing = read_jsonl(output_path)
    done = {row["counterfactual_id"] for row in existing if row.get("counterfactual_id")}
    rows_to_judge = [
        row for row in parsed_rows
        if row.get("counterfactual_id") not in done
        and row.get("parse_status") == "ok"
        and row.get("parsed_reasoning")
        and row.get("counterfactual_id") in prompts
    ]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    client = genai.Client(api_key=api_key)

    total = len(parsed_rows)
    remaining = len(rows_to_judge)
    chunk_total = (remaining + ACKNOWLEDGEMENT_JUDGE_BATCH_SIZE - 1) // ACKNOWLEDGEMENT_JUDGE_BATCH_SIZE if remaining else 0
    print("=" * 72)
    print("Running Gemini acknowledgement judge")
    print("=" * 72)
    print(f"Parsed outputs: {parsed_path}")
    print(f"Prompts:        {prompts_path}")
    print(f"Output:         {output_path}")
    print(f"Judge model:    {ACKNOWLEDGEMENT_JUDGE_MODEL} ({ACKNOWLEDGEMENT_JUDGE_MODEL_ID})")
    print(f"Total rows:     {total}")
    print(f"Remaining:      {remaining}")
    print(f"Chunk size:     {ACKNOWLEDGEMENT_JUDGE_BATCH_SIZE}")
    print("=" * 72)

    processed = 0
    for chunk_idx, chunk in enumerate(batched(rows_to_judge, ACKNOWLEDGEMENT_JUDGE_BATCH_SIZE), 1):
        requests = []
        chunk_rows = []
        for parsed in chunk:
            counterfactual_id = parsed["counterfactual_id"]
            prompt_row = prompts[counterfactual_id]
            judge_prompt = build_acknowledgement_judge_prompt(
                changed_factor=prompt_row.get("changed_factor", ""),
                change_description=prompt_row.get("change_description", ""),
                parsed_reasoning=parsed.get("parsed_reasoning", ""),
            )
            requests.append(_build_gemini_batch_request(judge_prompt, counterfactual_id))
            chunk_rows.append((parsed, prompt_row))

        print(
            f"chunk {chunk_idx}/{chunk_total} size={len(chunk_rows)} "
            f"counterfactuals={chunk_rows[0][0]['counterfactual_id']}..{chunk_rows[-1][0]['counterfactual_id']}",
            flush=True,
        )
        try:
            responses = _run_gemini_batch_job(
                client,
                ACKNOWLEDGEMENT_JUDGE_MODEL_ID,
                requests,
                f"acknowledgement-judge-chunk-{chunk_idx:03d}",
                debug_batch_dir=debug_batch_dir,
            )
        except Exception as exc:
            responses = {parsed["counterfactual_id"]: f"ERROR:{exc}" for parsed, _ in chunk_rows}

        for parsed, prompt_row in chunk_rows:
            processed += 1
            counterfactual_id = parsed["counterfactual_id"]
            raw_output = responses.get(counterfactual_id, "ERROR:Missing batch response for counterfactual_id")
            out = {
                "counterfactual_id": counterfactual_id,
                "base_case_id": parsed.get("base_case_id", prompt_row.get("base_case_id")),
                "model": parsed.get("model", prompt_row.get("model")),
                "bucket": parsed.get("bucket", prompt_row.get("bucket")),
                "edit_type": parsed.get("edit_type", prompt_row.get("edit_type")),
                "changed_factor": prompt_row.get("changed_factor", ""),
                "change_description": prompt_row.get("change_description", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if isinstance(raw_output, str) and raw_output.startswith("ERROR:"):
                out.update({
                    "raw_judge_output": "",
                    "error": raw_output.removeprefix("ERROR:"),
                    "acknowledged_by_llm": None,
                    "llm_acknowledgement_label": "ERROR",
                    "llm_acknowledgement_parse_status": "error",
                    "llm_acknowledgement_explanation": "",
                    "usage": {},
                    "judge_generation_config": {
                        "model": ACKNOWLEDGEMENT_JUDGE_MODEL,
                        "model_id": ACKNOWLEDGEMENT_JUDGE_MODEL_ID,
                        "max_completion_tokens": ACKNOWLEDGEMENT_JUDGE_MAX_COMPLETION_TOKENS,
                        "mode": "gemini_batch_api",
                    },
                })
                print(f"[{processed}/{remaining}] {parsed.get('model')} {parsed.get('edit_type')} {counterfactual_id} ... ERROR")
            else:
                parsed_judge = parse_acknowledgement_judge_output(raw_output)
                out.update({
                    "raw_judge_output": raw_output,
                    "error": None,
                    **parsed_judge,
                    "usage": {},
                    "judge_generation_config": {
                        "model": ACKNOWLEDGEMENT_JUDGE_MODEL,
                        "model_id": ACKNOWLEDGEMENT_JUDGE_MODEL_ID,
                        "max_completion_tokens": ACKNOWLEDGEMENT_JUDGE_MAX_COMPLETION_TOKENS,
                        "mode": "gemini_batch_api",
                    },
                })
                print(
                    f"[{processed}/{remaining}] {parsed.get('model')} {parsed.get('edit_type')} "
                    f"{counterfactual_id} ... {parsed_judge['llm_acknowledgement_label']}"
                )
            append_jsonl(output_path, out)
            time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(COUNTERFACTUAL_PROMPTS))
    parser.add_argument("--parsed", default=str(PARSED_OUTPUTS))
    parser.add_argument("--output", default=str(LLM_JUDGE_ACKNOWLEDGEMENT_OUTPUTS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--debug-batch-dir", default=None)
    args = parser.parse_args()

    run_acknowledgement_judge(
        prompts_path=Path(args.prompts),
        parsed_path=Path(args.parsed),
        output_path=Path(args.output),
        limit=args.limit,
        sleep_seconds=args.sleep,
        debug_batch_dir=Path(args.debug_batch_dir) if args.debug_batch_dir else None,
    )


if __name__ == "__main__":
    main()
