from __future__ import annotations

from scenario_config import (
    COST_LEVELS,
    DOMAINS,
    PROBABILITY_LEVELS,
    SEVERITY_LEVELS,
    generate_domain_contexts,
)
from post_inspection_builder import build_issue_description, build_post_inspection_prompt
from scenario_builder import assemble_scenario


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_scenarios(client, contexts: dict | None = None) -> list[dict]:
    if contexts is None and client is None:
        raise ValueError("client is required when contexts are not provided")
    if contexts is None:
        contexts = generate_domain_contexts(client)

    scenarios = []

    for variant in ["offered_report", "report_discovery", "evidence_discovery"]:
        for domain_key in DOMAINS:
            for cost in COST_LEVELS:
                for (severity_label, severity_description) in SEVERITY_LEVELS:
                    for (probability_label, probability) in PROBABILITY_LEVELS:

                        scenario = assemble_scenario(
                            domain_key=domain_key,
                            context=contexts[domain_key],
                            cost_label=cost["label"],
                            cost_units=cost["cost_units"],
                            cost_type=cost["cost_type"],
                            mandatory_review=cost["mandatory_review"],
                            cost_order=cost["cost_order"],
                            project_value=cost["project_value"],
                            failure_cost=cost["failure_cost"],
                            severity_label=severity_label,
                            severity_description=severity_description,
                            probability_label=probability_label,
                            probability=probability,
                            variant=variant,
                        )

                        # Attach post-inspection prompt for every scenario
                        # (used only when subject model chooses INSPECT)
                        issue_description = build_issue_description(
                            domain_key=domain_key,
                            severity_label=severity_label,
                        )
                        scenario["issue_description"] = issue_description
                        scenario["post_inspection_prompt"] = build_post_inspection_prompt(
                            domain_key=domain_key,
                            severity_label=severity_label,
                            project_value=cost["project_value"],
                            failure_cost=cost["failure_cost"],
                        )
                        scenarios.append(scenario)

    return scenarios
