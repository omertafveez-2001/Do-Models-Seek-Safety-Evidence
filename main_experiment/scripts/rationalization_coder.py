"""
Rationalization Coder — Strategic Ignorance Step 3
==================================================
Runs two coding passes per run file and saves two separate JSONs:

  main_experiment/coded/{filename}                    — offered-report initial coding
  main_experiment/coded/post_{filename}               — offered-report post-inspection coding
  main_experiment/coded/{variant}/{filename}          — discovery-variant initial coding
  main_experiment/coded/{variant}/post_{filename}     — discovery-variant post-inspection coding

Usage:
    python main_experiment/scripts/rationalization_coder.py --results main_experiment/runs/offered_report/sonnet_offered_report.json
    python main_experiment/scripts/rationalization_coder.py --results main_experiment/coded/evidence_discovery/opus_evidence_discovery.json --force
    python main_experiment/scripts/rationalization_coder.py --results main_experiment/coded/evidence_discovery/post_opus_evidence_discovery.json --force
    python main_experiment/scripts/rationalization_coder.py --all --variant offered_report
    python main_experiment/scripts/rationalization_coder.py --all --variant report_discovery --model gpt-4o-mini
    python main_experiment/scripts/rationalization_coder.py --all --variant evidence_discovery --force
    python main_experiment/scripts/rationalization_coder.py --all --variant offered_report --save-path /path/to/output/dir
"""

import os
import re
import json
import time
import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_experiment.scripts.prompts import (
    BATCH_INITIAL_PROMPT,
    BATCH_POST_PROMPT,
    INITIAL_PROMPT,
    POST_PROMPT,
    VALID_LABELS,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.1-pro-preview"
SLEEP         = 0.3
BATCH_SIZE    = 20
GEMINI_BATCH_POLL_SECONDS = 300
GEMINI_BATCH_BACKOFF_SECONDS = [120, 240, 480, 900]
GEMINI_BATCH_REQUEST_CHUNK_SIZE = 500
VARIANT_NAMES = {"offered_report", "report_discovery", "evidence_discovery"}


def make_client(model: str):
    if model.startswith("claude"):
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if model.startswith("gemini"):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY for Gemini model.")

        try:
            from google import genai
            return "gemini_genai", genai.Client(api_key=api_key)
        except ModuleNotFoundError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                return "gemini_legacy", genai.GenerativeModel(model)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Gemini SDK not installed. Install `google-genai` or `google-generativeai`."
                ) from exc
    else:
        from openai import OpenAI
        return "openai", OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def call_model(provider: str, client, model: str, prompt: str) -> str:
    if provider == "anthropic":
        r = client.messages.create(
            model=model,
            max_tokens=32,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip().lower()
    if provider == "gemini_genai":
        r = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return (r.text or "").strip().lower()
    if provider == "gemini_legacy":
        r = client.generate_content(prompt)
        return (getattr(r, "text", "") or "").strip().lower()
    else:
        r = client.chat.completions.create(
            model=model,
            max_completion_tokens=1024,
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip().lower()


def batched(items: list, size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]


_CONTRADICTION_RE = re.compile(
    r'^contradiction\[([a-z_]+)->([a-z_]+)\]$'
)
_STEP_CHECK_RE = re.compile(r"^step_check:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LABEL_RE = re.compile(r"^label:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
POST_LABELS = VALID_LABELS["A"] | VALID_LABELS["B"] | VALID_LABELS["C"]


def parse_structured_output(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    step_match = _STEP_CHECK_RE.search(raw)
    label_match = _LABEL_RE.search(raw)
    step_check = step_match.group(1).strip() if step_match else ""
    label = label_match.group(1).strip().lower() if label_match else raw.lower()
    return step_check, label


def parse_batch_output(raw: str) -> dict[str, tuple[str, str]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("Batch response is not a JSON list.")

    out = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Batch response contains a non-object item.")
        item_id = str(item.get("item_id", "")).strip()
        if not item_id:
            raise ValueError("Batch response item missing item_id.")
        step_check = str(item.get("step_check", "")).strip()
        label = str(item.get("label", "")).strip().lower()
        out[item_id] = (step_check, label)
    return out


def _compact_json(obj, limit: int = 300) -> str:
    try:
        text = json.dumps(obj, sort_keys=True)
    except Exception:
        text = str(obj)
    return text if len(text) <= limit else text[:limit] + "..."


def _is_resource_exhausted_error(message: str) -> bool:
    text = str(message)
    return "429" in text and "RESOURCE_EXHAUSTED" in text


def extract_response_text(response_obj: dict) -> str:
    if not isinstance(response_obj, dict):
        raise ValueError("Batch response payload is not an object.")

    direct_text = response_obj.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip().lower()

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
    return text.lower()

def validate_initial_label(raw: str, decision: str) -> str:
    raw = raw.strip().lower()
    valid = VALID_LABELS.get(decision, set())
    return raw if raw in valid else f"INVALID:{raw}"


def validate_post_label(raw: str, decision: str) -> str:
    raw = raw.strip().lower()

    match = _CONTRADICTION_RE.match(raw)
    if match:
        left, right = match.groups()
        if left in POST_LABELS and right in POST_LABELS:
            return raw
        return f"INVALID:{raw}"

    valid = VALID_LABELS.get(decision, set())
    return raw if raw in valid else f"INVALID:{raw}"


def should_recode(existing_row: dict | None, force: bool, retry_errors: bool, label_key: str) -> bool:
    if force:
        return True
    if existing_row is None:
        return True
    if label_key not in existing_row:
        return True
    if retry_errors and str(existing_row.get(label_key, "")).startswith("ERROR:"):
        return True
    return False


def build_existing_index(existing: list[dict]) -> dict[tuple[str, int], dict]:
    return {(r["scenario_id"], r["run_idx"]): r for r in existing}


def upsert_row(rows: list[dict], row: dict, key: tuple[str, int]):
    for idx, existing in enumerate(rows):
        if (existing["scenario_id"], existing["run_idx"]) == key:
            rows[idx] = row
            return
    rows.append(row)


def _extract_instructions(prompt: str) -> str:
    marker = "===========================\nOUTPUT FORMAT\n===========================\n"
    if marker in prompt:
        return prompt.split(marker, 1)[0].strip()
    return prompt.strip()


INITIAL_BATCH_INSTRUCTIONS = _extract_instructions(INITIAL_PROMPT)
POST_BATCH_INSTRUCTIONS = _extract_instructions(POST_PROMPT)


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


def _extract_output_file_name(batch_job) -> str | None:
    dest = getattr(batch_job, "dest", None)
    if dest is not None:
        file_name = getattr(dest, "file_name", None) or getattr(dest, "fileName", None)
        if file_name:
            return file_name

    output = getattr(batch_job, "output", None)
    if output is not None:
        file_name = getattr(output, "responses_file", None) or getattr(output, "responsesFile", None)
        if file_name:
            return file_name

    if isinstance(batch_job, dict):
        dest = batch_job.get("dest") or {}
        file_name = dest.get("file_name") or dest.get("fileName")
        if file_name:
            return file_name
        output = batch_job.get("output") or {}
        return output.get("responses_file") or output.get("responsesFile")

    return None


def _run_single_gemini_batch_job(client, model: str, requests: list[dict], display_name: str) -> dict[str, str]:
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

        while True:
            batch_job = client.batches.get(name=batch_name)
            raw_state_name = _batch_state_name(getattr(batch_job, "state", None))
            state_name = _normalized_batch_state_name(getattr(batch_job, "state", None))
            print(f"    state: {raw_state_name}", flush=True)
            if _is_terminal_batch_state(state_name):
                break
            time.sleep(GEMINI_BATCH_POLL_SECONDS)

        if state_name not in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
            raise RuntimeError(f"Gemini batch job did not succeed: {state_name}")

        output_file_name = _extract_output_file_name(batch_job)
        if not output_file_name:
            raise RuntimeError("Gemini batch job succeeded but no output file was returned.")

        raw_output = client.files.download(file=output_file_name).decode("utf-8")
        results_by_key = {}
        for line in raw_output.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            key = payload.get("key")
            if key is None:
                raise ValueError("Gemini batch output line is missing key.")

            if "response" in payload:
                results_by_key[key] = extract_response_text(payload["response"])
                continue

            if "error" in payload:
                results_by_key[key] = f"ERROR:{payload['error']}"
                continue

            if "status" in payload:
                results_by_key[key] = f"ERROR:{payload['status']}"
                continue

            results_by_key[key] = extract_response_text(payload)

        if results_by_key and all(
            isinstance(value, str) and _is_resource_exhausted_error(value)
            for value in results_by_key.values()
        ):
            first_error = next(iter(results_by_key.values()))
            raise RuntimeError(first_error)

        return results_by_key
    finally:
        request_file.unlink(missing_ok=True)


def run_gemini_batch_job(client, model: str, requests: list[dict], display_name: str) -> dict[str, str]:
    attempts = len(GEMINI_BATCH_BACKOFF_SECONDS) + 1
    for attempt in range(attempts):
        try:
            return _run_single_gemini_batch_job(client, model, requests, display_name)
        except Exception as exc:
            if not _is_resource_exhausted_error(exc):
                raise
            if attempt >= len(GEMINI_BATCH_BACKOFF_SECONDS):
                raise
            wait_seconds = GEMINI_BATCH_BACKOFF_SECONDS[attempt]
            wait_minutes = wait_seconds // 60
            print(
                f"    ERROR 429 RESOURCE_EXHAUSTED...resubmitting after {wait_minutes} minutes",
                flush=True,
            )
            time.sleep(wait_seconds)


def build_gemini_batch_request(prompt: str, item_id: str) -> dict:
    return {
        "key": item_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        },
    }


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]

# ---------------------------------------------------------------------------
# Pass 1 — initial INSPECT/SKIP
# ---------------------------------------------------------------------------

def _load_scenario_prompts() -> dict[str, dict]:
    scenarios_path = Path(__file__).resolve().parents[2] / "data-gen" / "data" / "scenarios.json"
    with open(scenarios_path) as f:
        scenarios = json.load(f)
    return {
        s["scenario_id"]: {
            "prompt":             s["prompt"],
            "post_inspection_prompt": s["post_inspection_prompt"],
            "inspection_finding": s.get("issue_description", ""),
        }
        for s in scenarios
    }


def code_initial(
    results: list,
    results_file: Path,
    provider,
    client,
    model: str,
    coded_dir: Path,
    force: bool = False,
    retry_errors: bool = False,
):
    coded_file      = coded_dir / results_file.name
    scenario_prompts = _load_scenario_prompts()

    existing = []
    if coded_file.exists() and not force:
        with open(coded_file) as f:
            existing = json.load(f)
    coded = list(existing)
    existing_index = build_existing_index(existing)

    to_code = [r for r in results
               if r["decision"] in ("INSPECT", "SKIP")
               and should_recode(
                   existing_index.get((r["scenario_id"], r["run_idx"])),
                   force,
                   retry_errors,
                   "rationalization",
               )]
    uncoded = [r for r in results
               if r["decision"] not in ("INSPECT", "SKIP")
               and should_recode(
                   existing_index.get((r["scenario_id"], r["run_idx"])),
                   force,
                   retry_errors,
                   "rationalization",
               )]
    preserved = len(results) - len(to_code) - len(uncoded)

    print(f"\n  [initial] to code: {len(to_code)}  preserved: {preserved}  skipped: {len(uncoded)}")

    if provider == "gemini_genai" and to_code:
        total_chunks = (len(to_code) + GEMINI_BATCH_REQUEST_CHUNK_SIZE - 1) // GEMINI_BATCH_REQUEST_CHUNK_SIZE
        total_errors = 0
        for chunk_idx, result_chunk in enumerate(chunked(to_code, GEMINI_BATCH_REQUEST_CHUNK_SIZE), 1):
            requests = []
            for result in result_chunk:
                item_id = f"{result['scenario_id']}::{result['run_idx']}"
                prompt = INITIAL_PROMPT.format(
                    scenario=scenario_prompts.get(result["scenario_id"], {}).get("prompt", ""),
                    decision=result["decision"],
                    allowed_labels=", ".join(sorted(VALID_LABELS[result["decision"]])),
                    reasoning=result["response"],
                )
                requests.append(build_gemini_batch_request(prompt, item_id))
            try:
                print(
                    f"    Gemini batch chunk {chunk_idx}/{total_chunks} requests={len(requests)}",
                    flush=True,
                )
                parsed = run_gemini_batch_job(
                    client,
                    model,
                    requests=requests,
                    display_name=f"initial-{results_file.stem}-chunk-{chunk_idx:03d}",
                )
                chunk_errors = 0
                for result in result_chunk:
                    key = (result["scenario_id"], result["run_idx"])
                    item_id = f"{result['scenario_id']}::{result['run_idx']}"
                    raw = parsed[item_id]
                    if raw.startswith("ERROR:"):
                        step_check = ""
                        label = raw
                        chunk_errors += 1
                    else:
                        step_check, parsed_label = parse_structured_output(raw)
                        label = validate_initial_label(parsed_label, result["decision"])
                    if label.startswith("INVALID:"):
                        chunk_errors += 1
                    upsert_row(coded, {
                        **result,
                        "rationalization": label,
                        "judge_reasoning": step_check,
                        "coder_model": model,
                    }, key)
            except Exception as e:
                chunk_errors = len(result_chunk)
                for result in result_chunk:
                    key = (result["scenario_id"], result["run_idx"])
                    upsert_row(coded, {
                        **result,
                        "rationalization": f"ERROR:{e}",
                        "judge_reasoning": "",
                        "coder_model": model,
                    }, key)
            total_errors += chunk_errors
            print(f"    chunk errors={chunk_errors}")
            with open(coded_file, "w") as f:
                json.dump(coded, f, indent=2)
            time.sleep(SLEEP)
        print(f"    batch api errors={total_errors}")
    else:
        for batch_idx, batch in enumerate(batched(to_code), 1):
            batch_items = []
            for result in batch:
                item_id = f"{result['scenario_id']}::{result['run_idx']}"
                batch_items.append({
                    "item_id": item_id,
                    "scenario": scenario_prompts.get(result["scenario_id"], {}).get("prompt", ""),
                    "decision": result["decision"],
                    "allowed_labels": sorted(VALID_LABELS[result["decision"]]),
                    "reasoning": result["response"],
                })
            print(
                f"    batch {batch_idx}/{(len(to_code) + BATCH_SIZE - 1) // BATCH_SIZE} "
                f"items={len(batch)}",
                end=" ... ",
                flush=True,
            )
            try:
                raw = call_model(
                    provider,
                    client,
                    model,
                    BATCH_INITIAL_PROMPT.format(
                        instructions=INITIAL_BATCH_INSTRUCTIONS,
                        items=json.dumps(batch_items, indent=2),
                    ),
                )
                parsed = parse_batch_output(raw)
                batch_errors = 0
                for result in batch:
                    key = (result["scenario_id"], result["run_idx"])
                    item_id = f"{result['scenario_id']}::{result['run_idx']}"
                    step_check, parsed_label = parsed[item_id]
                    label = validate_initial_label(parsed_label, result["decision"])
                    if label.startswith("INVALID:"):
                        batch_errors += 1
                    upsert_row(coded, {
                        **result,
                        "rationalization": label,
                        "judge_reasoning": step_check,
                        "coder_model": model,
                    }, key)
            except Exception as e:
                batch_errors = len(batch)
                for result in batch:
                    key = (result["scenario_id"], result["run_idx"])
                    upsert_row(coded, {
                        **result,
                        "rationalization": f"ERROR:{e}",
                        "judge_reasoning": "",
                        "coder_model": model,
                    }, key)
            print(f"errors={batch_errors}")
            with open(coded_file, "w") as f:
                json.dump(coded, f, indent=2)
            time.sleep(SLEEP)

    for result in uncoded:
        key = (result["scenario_id"], result["run_idx"])
        upsert_row(coded, {
            **result,
            "rationalization": None,
            "judge_reasoning": None,
            "coder_model": model,
        }, key)
    with open(coded_file, "w") as f:
        json.dump(coded, f, indent=2)

    print(f"  Saved → {coded_file}")
    _print_distribution([r.get("rationalization") for r in coded])


# ---------------------------------------------------------------------------
# Pass 2 — post-inspection A/B/C
# ---------------------------------------------------------------------------

def code_post(
    results: list,
    results_file: Path,
    provider,
    client,
    model: str,
    coded_dir: Path,
    force: bool = False,
    retry_errors: bool = False,
):
    coded_file       = coded_dir / f"post_{results_file.name}"
    scenario_prompts = _load_scenario_prompts()

    existing = []
    if coded_file.exists() and not force:
        with open(coded_file) as f:
            existing = json.load(f)
    coded = list(existing)
    existing_index = build_existing_index(existing)

    # Only rows where the model inspected and got a valid A/B/C decision
    to_code = [r for r in results
               if r.get("post_inspection_decision") in ("A", "B", "C")
               and should_recode(
                   existing_index.get((r["scenario_id"], r["run_idx"])),
                   force,
                   retry_errors,
                   "post_rationalization",
               )]
    preserved = sum(
        1
        for r in results
        if r.get("post_inspection_decision") in ("A", "B", "C")
        and not should_recode(
            existing_index.get((r["scenario_id"], r["run_idx"])),
            force,
            retry_errors,
            "post_rationalization",
        )
    )
    skipped = len(results) - len(to_code) - preserved

    print(f"\n  [post]    to code: {len(to_code)}  preserved: {preserved}  skipped (no post data): {skipped}")

    if provider == "gemini_genai" and to_code:
        total_chunks = (len(to_code) + GEMINI_BATCH_REQUEST_CHUNK_SIZE - 1) // GEMINI_BATCH_REQUEST_CHUNK_SIZE
        total_errors = 0
        for chunk_idx, result_chunk in enumerate(chunked(to_code, GEMINI_BATCH_REQUEST_CHUNK_SIZE), 1):
            requests = []
            for result in result_chunk:
                sc = scenario_prompts.get(result["scenario_id"], {})
                item_id = f"{result['scenario_id']}::{result['run_idx']}"
                dec = result["post_inspection_decision"]
                prompt = POST_PROMPT.format(
                    scenario=sc.get("prompt", ""),
                    inspection_finding=sc.get("inspection_finding", ""),
                    post_decision=dec,
                    allowed_labels=", ".join(sorted(VALID_LABELS[dec])),
                    all_post_labels=", ".join(sorted(POST_LABELS)),
                    reasoning=result["post_inspection_response"],
                )
                requests.append(build_gemini_batch_request(prompt, item_id))
            try:
                print(
                    f"    Gemini batch chunk {chunk_idx}/{total_chunks} requests={len(requests)}",
                    flush=True,
                )
                parsed = run_gemini_batch_job(
                    client,
                    model,
                    requests=requests,
                    display_name=f"post-{results_file.stem}-chunk-{chunk_idx:03d}",
                )
                chunk_errors = 0
                for result in result_chunk:
                    key = (result["scenario_id"], result["run_idx"])
                    item_id = f"{result['scenario_id']}::{result['run_idx']}"
                    dec = result["post_inspection_decision"]
                    raw = parsed[item_id]
                    if raw.startswith("ERROR:"):
                        step_check = ""
                        label = raw
                        chunk_errors += 1
                    else:
                        step_check, parsed_label = parse_structured_output(raw)
                        label = validate_post_label(parsed_label, dec)
                    if label.startswith("INVALID:"):
                        chunk_errors += 1
                    upsert_row(coded, {
                        **result,
                        "post_rationalization":  label,
                        "judge_reasoning":       step_check,
                        "contradiction_exists":  bool(_CONTRADICTION_RE.match(label)),
                        "coder_model":           model,
                    }, key)
            except Exception as e:
                chunk_errors = len(result_chunk)
                for result in result_chunk:
                    key = (result["scenario_id"], result["run_idx"])
                    upsert_row(coded, {
                        **result,
                        "post_rationalization":  f"ERROR:{e}",
                        "judge_reasoning":       "",
                        "contradiction_exists":  False,
                        "coder_model":           model,
                    }, key)
            total_errors += chunk_errors
            print(f"    chunk errors={chunk_errors}")
            with open(coded_file, "w") as f:
                json.dump(coded, f, indent=2)
            time.sleep(SLEEP)
        print(f"    batch api errors={total_errors}")
    else:
        for batch_idx, batch in enumerate(batched(to_code), 1):
            batch_items = []
            for result in batch:
                sc = scenario_prompts.get(result["scenario_id"], {})
                item_id = f"{result['scenario_id']}::{result['run_idx']}"
                dec = result["post_inspection_decision"]
                batch_items.append({
                    "item_id": item_id,
                    "scenario": sc.get("prompt", ""),
                    "inspection_finding": sc.get("inspection_finding", ""),
                    "post_decision": dec,
                    "allowed_labels": sorted(VALID_LABELS[dec]),
                    "all_post_labels": sorted(POST_LABELS),
                    "reasoning": result["post_inspection_response"],
                })
            print(
                f"    batch {batch_idx}/{(len(to_code) + BATCH_SIZE - 1) // BATCH_SIZE} "
                f"items={len(batch)}",
                end=" ... ",
                flush=True,
            )
            try:
                raw = call_model(
                    provider,
                    client,
                    model,
                    BATCH_POST_PROMPT.format(
                        instructions=POST_BATCH_INSTRUCTIONS,
                        items=json.dumps(batch_items, indent=2),
                    ),
                )
                parsed = parse_batch_output(raw)
                batch_errors = 0
                for result in batch:
                    key = (result["scenario_id"], result["run_idx"])
                    item_id = f"{result['scenario_id']}::{result['run_idx']}"
                    dec = result["post_inspection_decision"]
                    step_check, parsed_label = parsed[item_id]
                    label = validate_post_label(parsed_label, dec)
                    if label.startswith("INVALID:"):
                        batch_errors += 1
                    upsert_row(coded, {
                        **result,
                        "post_rationalization":  label,
                        "judge_reasoning":       step_check,
                        "contradiction_exists":  bool(_CONTRADICTION_RE.match(label)),
                        "coder_model":           model,
                    }, key)
            except Exception as e:
                batch_errors = len(batch)
                for result in batch:
                    key = (result["scenario_id"], result["run_idx"])
                    upsert_row(coded, {
                        **result,
                        "post_rationalization":  f"ERROR:{e}",
                        "judge_reasoning":       "",
                        "contradiction_exists":  False,
                        "coder_model":           model,
                    }, key)
            print(f"errors={batch_errors}")
            with open(coded_file, "w") as f:
                json.dump(coded, f, indent=2)
            time.sleep(SLEEP)

    with open(coded_file, "w") as f:
        json.dump(coded, f, indent=2)

    print(f"  Saved → {coded_file}")
    _print_distribution([r.get("post_rationalization") for r in coded])


def _print_distribution(labels: list):
    labels = [label for label in labels if label]
    if labels:
        print("  Label distribution:")
        for label, count in Counter(labels).most_common():
            print(f"    {label:<30} {count}")


def _infer_single_file_mode(results_file: Path) -> str:
    return "post" if results_file.name.startswith("post_") else "initial"


def _single_file_output_name(results_file: Path, mode: str) -> str:
    if mode == "post" and results_file.name.startswith("post_"):
        return results_file.name.removeprefix("post_")
    return results_file.name


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", help="path to a single results JSON file")
    group.add_argument("--all",     action="store_true", help="code all files in main_experiment/runs/<variant>/")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--variant",     choices=sorted(VARIANT_NAMES), help="runs/coded variant directory name")
    parser.add_argument("--force",       action="store_true", help="recode and overwrite existing judge labels")
    parser.add_argument("--retry-errors", action="store_true", help="recode only rows whose existing label is an ERROR:* value")
    parser.add_argument("--save-path",   default=None, help="override output directory for coded files")
    args = parser.parse_args()

    provider, client = make_client(args.model)

    experiment_dir = Path(__file__).parent.parent
    results_root = experiment_dir / "runs"
    coded_root = experiment_dir / "coded"

    if args.all and not args.variant:
        parser.error("--all requires --variant")

    if args.results:
        results_file = Path(args.results)
        results_dir = results_file.parent
        variant_name = args.variant or results_dir.name
    else:
        variant_name = args.variant
        results_dir = results_root / variant_name

    if args.save_path:
        coded_dir = Path(args.save_path)
    else:
        coded_dir = coded_root / variant_name if variant_name in VARIANT_NAMES else coded_root
    coded_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob("*.json")) if args.all else [results_file]
    if not files:
        print(f"No JSON files found in {results_dir}")
        return

    print("=" * 60)
    print("Rationalization Coder")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Variant:{variant_name}")
    print(f"Files:  {len(files)}")
    print(f"Force:  {args.force}")
    print("=" * 60)

    for results_file in files:
        print(f"\n{results_file.name}")
        with open(results_file) as f:
            results = json.load(f)
        if args.results:
            mode = _infer_single_file_mode(results_file)
            output_name = _single_file_output_name(results_file, mode)
            print(f"Mode:   {mode}")
            dispatch_file = Path(output_name)
            if mode == "post":
                code_post(
                    results,
                    dispatch_file,
                    provider,
                    client,
                    args.model,
                    coded_dir,
                    force=args.force,
                    retry_errors=args.retry_errors,
                )
            else:
                code_initial(
                    results,
                    dispatch_file,
                    provider,
                    client,
                    args.model,
                    coded_dir,
                    force=args.force,
                    retry_errors=args.retry_errors,
                )
            continue

        code_initial(
            results,
            results_file,
            provider,
            client,
            args.model,
            coded_dir,
            force=args.force,
            retry_errors=args.retry_errors,
        )
        code_post(
            results,
            results_file,
            provider,
            client,
            args.model,
            coded_dir,
            force=args.force,
            retry_errors=args.retry_errors,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
