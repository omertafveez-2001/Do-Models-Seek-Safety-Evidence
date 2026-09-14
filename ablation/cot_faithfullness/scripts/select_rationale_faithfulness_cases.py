from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    BUCKET_TARGETS,
    BUCKET_VARIANT_CAPS,
    CONFIG_JSON,
    COST_KEYWORDS,
    COST_OBLIGATION_RESULTS_DIR,
    EXPERIMENT_NAME,
    MODELS,
    OBLIGATION_KEYWORDS,
    POST_RATIONALIZATION_LABELS,
    RANDOM_SEED,
    RISK_KEYWORDS,
    SELECTED_BASE_CASES,
)
from io_utils import (
    first_prompt_for_row,
    inspect_fraction_by_key,
    load_rows_for,
    load_scenarios,
    post_prompt_for_row,
    read_json,
    reasoning_from_text,
    scenario_match_key,
    write_json,
    write_jsonl,
)


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = (text or "").lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _base_id(model: str, bucket: str, row: dict, suffix: str = "") -> str:
    run = row.get("run_idx", row.get("rollout", 0))
    scenario = row.get("scenario_id", "unknown")
    suffix_part = f"__{suffix}" if suffix else ""
    return f"{model}__{bucket}__{scenario}__run_{run}{suffix_part}"


def _main_rows(model: str) -> list[dict]:
    rows = []
    for variant in ["offered_report", "report_discovery", "evidence_discovery"]:
        rows.extend(load_rows_for(model, variant, coded=True, post=False))
    return rows


def _post_rows(model: str) -> list[dict]:
    rows = []
    for variant in ["offered_report", "report_discovery", "evidence_discovery"]:
        rows.extend(load_rows_for(model, variant, coded=True, post=True))
    return rows


def select_cost_rationale_cases(model: str, scenarios: dict[str, dict]) -> list[dict]:
    out = []
    for row in _main_rows(model):
        reasoning = reasoning_from_text(row.get("response"))
        if row.get("decision") != "SKIP":
            continue
        if row.get("rationalization") not in (None, "", "expected_value"):
            continue
        if not _contains_any(reasoning, COST_KEYWORDS):
            continue
        prompt = first_prompt_for_row(row, scenarios)
        if not prompt:
            continue
        out.append({
            "base_case_id": _base_id(model, "cost_rationale_skip", row),
            "model": model,
            "bucket": "cost_rationale_skip",
            "source": "main_coded",
            "source_variant": row.get("variant"),
            "scenario_id": row.get("scenario_id"),
            "run_idx": row.get("run_idx"),
            "original_decision": "SKIP",
            "original_response": row.get("response", ""),
            "original_reasoning": reasoning,
            "original_monitor_label": row.get("rationalization", ""),
            "original_prompt": prompt,
            "eligible_edit_types": ["cost_removed"],
        })
    return out


def select_risk_rationale_cases(model: str, scenarios: dict[str, dict]) -> list[dict]:
    out = []
    for row in _main_rows(model):
        reasoning = reasoning_from_text(row.get("response"))
        if row.get("decision") != "SKIP":
            continue
        if row.get("rationalization") not in (None, "", "expected_value"):
            continue
        if not _contains_any(reasoning, RISK_KEYWORDS):
            continue
        prompt = first_prompt_for_row(row, scenarios)
        if not prompt:
            continue
        edits = []
        if row.get("severity_label") != "high":
            edits.append("severity_raised")
        if row.get("probability_label") != "high":
            edits.append("probability_raised")
        if not edits:
            continue
        out.append({
            "base_case_id": _base_id(model, "risk_rationale_skip", row),
            "model": model,
            "bucket": "risk_rationale_skip",
            "source": "main_coded",
            "source_variant": row.get("variant"),
            "scenario_id": row.get("scenario_id"),
            "run_idx": row.get("run_idx"),
            "original_decision": "SKIP",
            "original_response": row.get("response", ""),
            "original_reasoning": reasoning,
            "original_monitor_label": row.get("rationalization", ""),
            "original_prompt": prompt,
            "eligible_edit_types": edits,
        })
    return out


def _load_cost_obligation_rows(model: str) -> list[dict]:
    path = COST_OBLIGATION_RESULTS_DIR / f"{model}_cost_obligation_decomposition.json"
    return read_json(path) if path.exists() else []


def select_obligation_cases(model: str) -> list[dict]:
    rows = _load_cost_obligation_rows(model)
    if not rows:
        return []

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("scenario_id"), row.get("condition"))].append(row)

    free_inspect = {
        scenario: any(r.get("decision") == "INSPECT" for r in vals)
        for (scenario, condition), vals in grouped.items()
        if condition == "free_no_obligation"
    }

    out = []
    for row in rows:
        if row.get("condition") not in {"obligation_only", "cost_plus_obligation"}:
            continue
        if row.get("decision") != "SKIP":
            continue
        reasoning = reasoning_from_text(row.get("raw_response"))
        mentions_obligation = _contains_any(reasoning, OBLIGATION_KEYWORDS)
        matched_drop = free_inspect.get(row.get("scenario_id"), False)
        if not (mentions_obligation or matched_drop):
            continue
        prompt = row.get("prompt", "")
        if not prompt:
            continue
        out.append({
            "base_case_id": _base_id(model, "obligation_sensitive", row),
            "model": model,
            "bucket": "obligation_sensitive",
            "source": "cost_obligation_decomposition",
            "source_variant": "cost_obligation_decomposition",
            "scenario_id": row.get("scenario_id"),
            "run_idx": row.get("rollout"),
            "condition": row.get("condition"),
            "original_decision": "SKIP",
            "original_response": row.get("raw_response", ""),
            "original_reasoning": reasoning,
            "original_monitor_label": "",
            "original_prompt": prompt,
            "eligible_edit_types": ["obligation_removed"],
            "selection_notes": {
                "mentions_obligation": mentions_obligation,
                "free_no_obligation_had_inspect": matched_drop,
            },
        })
    return out


def select_salience_cases(model: str, scenarios: dict[str, dict]) -> list[dict]:
    evidence_rows = load_rows_for(model, "evidence_discovery", coded=True, post=False)
    offered = inspect_fraction_by_key(load_rows_for(model, "offered_report", coded=False, post=False))
    report = inspect_fraction_by_key(load_rows_for(model, "report_discovery", coded=False, post=False))

    out = []
    for row in evidence_rows:
        if row.get("decision") != "SKIP":
            continue
        key = scenario_match_key(row)
        more_report_shaped_inspected = offered.get(key, 0) > 0 or report.get(key, 0) > 0
        if not more_report_shaped_inspected:
            continue
        prompt = first_prompt_for_row(row, scenarios)
        if not prompt:
            continue
        out.append({
            "base_case_id": _base_id(model, "salience_sensitive", row),
            "model": model,
            "bucket": "salience_sensitive",
            "source": "matched_main_results",
            "source_variant": "matched_salience_comparison",
            "scenario_id": row.get("scenario_id"),
            "run_idx": row.get("run_idx"),
            "original_decision": "SKIP",
            "original_response": row.get("response", ""),
            "original_reasoning": reasoning_from_text(row.get("response")),
            "original_monitor_label": row.get("rationalization", ""),
            "original_prompt": prompt,
            "eligible_edit_types": ["evidence_made_report_shaped"],
            "selection_notes": {
                "offered_report_inspect_fraction": offered.get(key, 0),
                "report_discovery_inspect_fraction": report.get(key, 0),
            },
        })
    return out


def select_post_rationalization_cases(model: str, scenarios: dict[str, dict]) -> list[dict]:
    out = []
    for row in _post_rows(model):
        label = row.get("post_rationalization", "")
        if label not in POST_RATIONALIZATION_LABELS:
            continue
        post_decision = row.get("post_inspection_decision")
        if post_decision not in {"B", "C"}:
            continue
        prompt = post_prompt_for_row(row, scenarios)
        if not prompt:
            continue
        original_decision = "PROCEED_DESPITE_FINDING" if post_decision == "B" else "PARTIAL_MITIGATION"
        out.append({
            "base_case_id": _base_id(model, "post_rationalization", row, suffix=post_decision),
            "model": model,
            "bucket": "post_rationalization",
            "source": "post_coded",
            "source_variant": row.get("variant"),
            "scenario_id": row.get("scenario_id"),
            "run_idx": row.get("run_idx"),
            "original_decision": original_decision,
            "original_response": row.get("post_inspection_response", ""),
            "original_reasoning": reasoning_from_text(row.get("post_inspection_response")),
            "original_monitor_label": label,
            "original_prompt": prompt,
            "eligible_edit_types": ["finding_made_harder_to_rationalize"],
        })
    return out


SELECTORS = {
    "cost_rationale_skip": select_cost_rationale_cases,
    "risk_rationale_skip": select_risk_rationale_cases,
    "obligation_sensitive": select_obligation_cases,
    "salience_sensitive": select_salience_cases,
    "post_rationalization": select_post_rationalization_cases,
}


def _dedupe(cases: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for case in cases:
        key = (case["model"], case["bucket"], case["scenario_id"], case.get("run_idx"), tuple(case["eligible_edit_types"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    return out


def _variant_counts(cases: list[dict]) -> dict[str, int]:
    return dict(Counter(case.get("source_variant", "unknown") for case in cases))


def _select_with_variant_cap(
    eligible: list[dict],
    target: int,
    variant_cap: int | None,
    rng: random.Random,
) -> list[dict]:
    eligible = list(eligible)
    rng.shuffle(eligible)
    if variant_cap is None:
        return eligible[:target]

    by_variant = defaultdict(list)
    for case in eligible:
        by_variant[case.get("source_variant", "unknown")].append(case)

    chosen = []
    chosen_ids = set()
    leftovers_by_variant = {}

    for variant in sorted(by_variant):
        variant_cases = by_variant[variant]
        for case in variant_cases[:variant_cap]:
            chosen.append(case)
            chosen_ids.add(case["base_case_id"])
        leftovers_by_variant[variant] = variant_cases[variant_cap:]

    if len(chosen) < target:
        variants_by_leftover = sorted(
            leftovers_by_variant,
            key=lambda variant: len(leftovers_by_variant[variant]),
            reverse=True,
        )
        for variant in variants_by_leftover:
            for case in leftovers_by_variant[variant]:
                if len(chosen) >= target:
                    break
                chosen.append(case)
                chosen_ids.add(case["base_case_id"])
            if len(chosen) >= target:
                break

    return chosen[:target]


def select_cases(seed: int = RANDOM_SEED) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    scenarios = load_scenarios()
    selected = []
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "models": MODELS,
        "random_seed": seed,
        "bucket_targets_per_model": BUCKET_TARGETS,
        "bucket_variant_caps": BUCKET_VARIANT_CAPS,
        "actual_counts": {},
        "shortfalls": {},
        "eligible_source_variant_counts": {},
        "selected_source_variant_counts": {},
    }

    print("=" * 72)
    print("Selecting rationale-faithfulness base cases")
    print("=" * 72)
    for model in MODELS:
        print(f"\nModel: {model}")
        metadata["actual_counts"][model] = {}
        metadata["shortfalls"][model] = {}
        metadata["eligible_source_variant_counts"][model] = {}
        metadata["selected_source_variant_counts"][model] = {}
        for bucket, target in BUCKET_TARGETS.items():
            selector = SELECTORS[bucket]
            if bucket in {"obligation_sensitive"}:
                eligible = selector(model)
            else:
                eligible = selector(model, scenarios)
            eligible = _dedupe(eligible)
            variant_cap = BUCKET_VARIANT_CAPS.get(bucket)
            chosen = _select_with_variant_cap(eligible, target, variant_cap, rng)
            selected.extend(chosen)
            shortfall = max(0, target - len(chosen))
            metadata["actual_counts"][model][bucket] = len(chosen)
            metadata["shortfalls"][model][bucket] = shortfall
            metadata["eligible_source_variant_counts"][model][bucket] = _variant_counts(eligible)
            metadata["selected_source_variant_counts"][model][bucket] = _variant_counts(chosen)
            print(
                f"  {bucket:<28} eligible={len(eligible):>4} selected={len(chosen):>3} "
                f"target={target:>3} shortfall={shortfall:>3} variant_cap={variant_cap if variant_cap is not None else '-':>3}"
            )

    print("\nTotal selected base cases:", len(selected))
    print("By bucket:", dict(Counter(case["bucket"] for case in selected)))
    return selected, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(SELECTED_BASE_CASES))
    parser.add_argument("--config-output", default=str(CONFIG_JSON))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    cases, metadata = select_cases(seed=args.seed)
    write_jsonl(Path(args.output), cases)
    write_json(Path(args.config_output), metadata)

    print(f"\nWrote selected base cases to {args.output}")
    print(f"Wrote selection metadata to {args.config_output}")


if __name__ == "__main__":
    main()
