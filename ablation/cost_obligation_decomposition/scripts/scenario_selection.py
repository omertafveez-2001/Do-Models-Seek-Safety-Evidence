from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .config import (
        DEFAULT_SCENARIO_COUNT,
        DEFAULT_SELECTION_FILE,
        PROJECT_ROOT,
        SCENARIOS_FILE,
        SOURCE_RESULTS_DIR,
        VARIANT,
    )
except ImportError:
    from config import (
        DEFAULT_SCENARIO_COUNT,
        DEFAULT_SELECTION_FILE,
        PROJECT_ROOT,
        SCENARIOS_FILE,
        SOURCE_RESULTS_DIR,
        VARIANT,
    )

ROOT = PROJECT_ROOT
DATA_GEN_DIR = ROOT / "data-gen"
if str(DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_DIR))

from scenario_config import COST_LEVELS, DOMAINS  # noqa: E402


HIGH_NUMERIC_COST = next(c for c in COST_LEVELS if c["label"] == "high" and c["cost_type"] == "numeric")


def _load_json(path: Path):
    with open(path) as fp:
        return json.load(fp)


def _scenario_key(row: dict) -> tuple[str, str, str]:
    severity = row.get("severity_label", row.get("severity_level", row.get("severity")))
    probability = row.get("probability_label", row.get("probability_level"))
    return (row["domain"], severity, probability)


def _result_files(variant: str) -> list[Path]:
    directory = SOURCE_RESULTS_DIR / variant
    suffix = f"_{variant}.json"
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(f"*{suffix}") if p.is_file())


def _model_key(path: Path, variant: str) -> str:
    return path.stem.removesuffix(f"_{variant}")


def load_inspect_fractions() -> dict[str, dict[str, dict[tuple[str, str, str], float]]]:
    fractions: dict[str, dict[str, dict[tuple[str, str, str], float]]] = {}
    for variant in ["offered_report", "report_discovery", "evidence_discovery"]:
        fractions[variant] = {}
        for path in _result_files(variant):
            model = _model_key(path, variant)
            grouped = defaultdict(list)
            for row in _load_json(path):
                if row.get("variant") != variant:
                    continue
                key = _scenario_key(row)
                grouped[key].append(row)
            fractions[variant][model] = {
                key: sum(1 for r in rows if r.get("decision") == "INSPECT") / len(rows)
                for key, rows in grouped.items()
                if rows
            }
    return fractions


def _baseline_features(key: tuple[str, str, str], fractions: dict) -> dict:
    offered = fractions.get("offered_report", {})
    report = fractions.get("report_discovery", {})
    evidence = fractions.get("evidence_discovery", {})

    model_values = []
    offered_values = []
    report_values = []
    evidence_values = []
    evidence_drops = []
    sonnet_drop = 0.0
    gpt_distance = 1.0

    for model, model_fractions in offered.items():
        offered_fraction = model_fractions.get(key)
        evidence_fraction = evidence.get(model, {}).get(key)
        report_fraction = report.get(model, {}).get(key)
        if offered_fraction is not None:
            model_values.append(offered_fraction)
            offered_values.append(offered_fraction)
        if evidence_fraction is not None:
            model_values.append(evidence_fraction)
            evidence_values.append(evidence_fraction)
        if report_fraction is not None:
            report_values.append(report_fraction)
        if offered_fraction is not None and evidence_fraction is not None:
            evidence_drops.append(max(0.0, offered_fraction - evidence_fraction))
        if report_fraction is not None and evidence_fraction is not None:
            evidence_drops.append(max(0.0, report_fraction - evidence_fraction))
        if model == "sonnet" and offered_fraction is not None and evidence_fraction is not None:
            sonnet_drop = max(0.0, offered_fraction - evidence_fraction)
        if model == "gpt-5.5" and evidence_fraction is not None:
            gpt_distance = abs(evidence_fraction - 0.5)

    if model_values:
        ceiling_floor_penalty = sum(1 for v in model_values if v in (0.0, 1.0)) / len(model_values)
        boundary_score = sum(1.0 - abs(v - 0.5) * 2 for v in model_values) / len(model_values)
    else:
        ceiling_floor_penalty = 1.0
        boundary_score = 0.0

    return {
        "boundary_score": boundary_score,
        "ceiling_floor_penalty": ceiling_floor_penalty,
        "max_evidence_drop": max(evidence_drops) if evidence_drops else 0.0,
        "sonnet_drop": sonnet_drop,
        "gpt_boundary_score": 1.0 - min(1.0, gpt_distance * 2),
        "offered_report_baseline_inspect_fraction": (
            sum(offered_values) / len(offered_values) if offered_values else ""
        ),
        "report_discovery_inspect_fraction": (
            sum(report_values) / len(report_values) if report_values else ""
        ),
        "evidence_discovery_inspect_fraction": (
            sum(evidence_values) / len(evidence_values) if evidence_values else ""
        ),
    }


def _selection_buckets(scenarios: list[dict], fractions: dict) -> dict[str, list[dict]]:
    buckets = {
        "medium_severity_medium_high_cost": [],
        "low_severity_medium_high_cost": [],
        "high_severity_medium_high_cost": [],
        "evidence_discovery_drop": [],
    }

    for scenario in scenarios:
        key = _scenario_key(scenario)
        _, severity, _ = key
        features = _baseline_features(key, fractions)
        row = {**scenario, **features}

        if severity == "medium" and scenario["source_cost_label"] in {"medium", "high"}:
            buckets["medium_severity_medium_high_cost"].append(row)
        if severity == "low" and scenario["source_cost_label"] in {"medium", "high"}:
            buckets["low_severity_medium_high_cost"].append(row)
        if severity == "high" and scenario["source_cost_label"] in {"medium", "high"}:
            buckets["high_severity_medium_high_cost"].append(row)
        if features["max_evidence_drop"] > 0:
            buckets["evidence_discovery_drop"].append(row)

    for bucket_name, rows in buckets.items():
        rows.sort(
            key=lambda r: (
                -r["boundary_score"],
                -r["max_evidence_drop"],
                -r["sonnet_drop"],
                -r["gpt_boundary_score"],
                r["ceiling_floor_penalty"],
                r["domain"],
                r["severity_level"],
                r["probability_level"],
            )
        )
    return buckets


def _build_base_scenarios(scenarios_file: Path) -> list[dict]:
    source = [
        s for s in _load_json(scenarios_file)
        if s.get("variant") == VARIANT and s.get("cost_label") in {"medium", "high"}
    ]
    by_key = {}
    for scenario in source:
        key = _scenario_key(scenario)
        if key in by_key and by_key[key]["source_cost_label"] == "high":
            continue
        by_key[key] = {
            "scenario_source": scenario["scenario_id"],
            "source_cost_label": scenario["cost_label"],
            "scenario_id": f"cost_obligation__{scenario['domain']}__{scenario['severity_label']}__{scenario['probability_label']}",
            "variant": VARIANT,
            "domain": scenario["domain"],
            "role": DOMAINS[scenario["domain"]]["role"],
            "system": DOMAINS[scenario["domain"]]["system"],
            "context": _extract_context(scenario["prompt"]),
            "project_value": scenario["project_value"],
            "failure_cost": scenario["failure_cost"],
            "probability": scenario["probability_pct"],
            "probability_level": scenario["probability_label"],
            "severity": scenario["severity_label"],
            "severity_level": scenario["severity_label"],
            "severity_description": scenario["severity_description"],
            "decomp_inspection_cost_points": HIGH_NUMERIC_COST["cost_units"],
        }
    return list(by_key.values())


def _extract_context(prompt: str) -> str:
    start = prompt.index("Context:\n") + len("Context:\n")
    end = prompt.index("\n\nObjective:", start)
    return prompt[start:end].strip()


def select_diagnostic_scenarios(
    scenarios_file: Path = SCENARIOS_FILE,
    count: int = DEFAULT_SCENARIO_COUNT,
) -> list[dict]:
    base = _build_base_scenarios(scenarios_file)
    fractions = load_inspect_fractions()
    buckets = _selection_buckets(base, fractions)
    targets = {
        "medium_severity_medium_high_cost": 15,
        "low_severity_medium_high_cost": 10,
        "high_severity_medium_high_cost": 10,
        "evidence_discovery_drop": 15,
    }

    selected: list[dict] = []
    selected_ids = set()
    for reason, target in targets.items():
        for scenario in buckets[reason]:
            if len([s for s in selected if s["selection_reason"] == reason]) >= target:
                break
            if scenario["scenario_id"] in selected_ids:
                continue
            selected.append({**scenario, "selection_reason": reason})
            selected_ids.add(scenario["scenario_id"])

    fallback = sorted(
        base,
        key=lambda s: (
            -_baseline_features(_scenario_key(s), fractions)["boundary_score"],
            s["domain"],
            s["severity_level"],
            s["probability_level"],
        ),
    )
    for scenario in fallback:
        if len(selected) >= count:
            break
        if scenario["scenario_id"] in selected_ids:
            continue
        features = _baseline_features(_scenario_key(scenario), fractions)
        selected.append({**scenario, **features, "selection_reason": "balanced_boundary_fallback"})
        selected_ids.add(scenario["scenario_id"])

    return selected[:count]


def write_selection_file(output: Path = DEFAULT_SELECTION_FILE, count: int = DEFAULT_SCENARIO_COUNT) -> list[dict]:
    selected = select_diagnostic_scenarios(count=count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fp:
        json.dump(
            {
                "experiment": "cost_obligation_decomposition",
                "selection_method": (
                    "Balanced decision-boundary subset selected from evidence_discovery scenarios "
                    "using existing offered_report/report_discovery/evidence_discovery inspect fractions. "
                    "Targets: 15 medium severity with medium/high source cost, 10 low, 10 high, "
                    "and 15 evidence-discovery drop cases; fallback fills remaining slots by "
                    "boundary score."
                ),
                "count": len(selected),
                "selection_reason_counts": dict(Counter(s["selection_reason"] for s in selected)),
                "scenarios": selected,
            },
            fp,
            indent=2,
        )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_SELECTION_FILE))
    parser.add_argument("--count", type=int, default=DEFAULT_SCENARIO_COUNT)
    args = parser.parse_args()

    selected = write_selection_file(Path(args.output), args.count)
    print(f"Wrote {len(selected)} selected scenarios to {args.output}")
    print(Counter(s["selection_reason"] for s in selected))


if __name__ == "__main__":
    main()
