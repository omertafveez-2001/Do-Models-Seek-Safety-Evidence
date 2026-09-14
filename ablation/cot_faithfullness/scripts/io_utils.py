from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from config import CODED_DIR, RESULTS_DIR, SCENARIOS_FILE


def read_json(path: Path):
    with open(path) as fp:
        return json.load(fp)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_scenarios() -> dict[str, dict]:
    return {row["scenario_id"]: row for row in read_json(SCENARIOS_FILE)}


def reasoning_from_text(text: str | None) -> str:
    if not text:
        return ""
    marker = "reasoning:"
    lower = text.lower()
    idx = lower.find(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker):].strip()


def scenario_match_key(row: dict) -> tuple:
    scenario_id = row.get("scenario_id", "")
    parts = scenario_id.split("__")
    if len(parts) >= 5:
        return tuple(parts[1:5])
    return (
        row.get("domain"),
        row.get("cost_label"),
        row.get("severity_label"),
        row.get("probability_label"),
    )


def model_from_path(path: Path, variant: str, post: bool = False) -> str:
    stem = path.stem
    if post:
        stem = stem.removeprefix("post_")
    return stem.removesuffix(f"_{variant}")


def result_file(model: str, variant: str, coded: bool = False, post: bool = False) -> Path:
    base = CODED_DIR if coded else RESULTS_DIR
    directory = base / variant
    prefix = "post_" if post else ""
    return directory / f"{prefix}{model}_{variant}.json"


def load_rows_for(model: str, variant: str, coded: bool = False, post: bool = False) -> list[dict]:
    path = result_file(model, variant, coded=coded, post=post)
    return read_json(path) if path.exists() else []


def inspect_fraction_by_key(rows: list[dict]) -> dict[tuple, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[scenario_match_key(row)].append(row)
    return {
        key: sum(1 for r in vals if r.get("decision") == "INSPECT") / len(vals)
        for key, vals in grouped.items()
        if vals
    }


def first_prompt_for_row(row: dict, scenarios: dict[str, dict]) -> str:
    prompt = row.get("prompt")
    if prompt:
        return prompt
    scenario = scenarios.get(row.get("scenario_id", ""))
    return scenario.get("prompt", "") if scenario else ""


def post_prompt_for_row(row: dict, scenarios: dict[str, dict]) -> str:
    prompt = row.get("post_inspection_prompt")
    if prompt:
        return prompt
    scenario = scenarios.get(row.get("scenario_id", ""))
    return scenario.get("post_inspection_prompt", "") if scenario else ""
