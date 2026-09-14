from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODULE_ROOT.parent.parent
DATA_DIR = MODULE_ROOT / "data"
RESULTS_DIR = MODULE_ROOT / "results"
GRAPHS_DIR = MODULE_ROOT / "graphs"
SOURCE_RESULTS_DIR = PROJECT_ROOT / "main_experiment" / "runs"

EXPERIMENT_ID = "cost_obligation_decomposition"
VARIANT = "evidence_discovery"

DEFAULT_SELECTION_FILE = DATA_DIR / "cost_obligation_selected_scenarios.json"
DEFAULT_RESULTS_DIR = RESULTS_DIR
SCENARIOS_FILE = PROJECT_ROOT / "data-gen" / "data" / "scenarios.json"

DEFAULT_SCENARIO_COUNT = 50
DEFAULT_RUNS = 5
DEFAULT_SLEEP_SECONDS = 0.5

ANTHROPIC_MAX_TOKENS = 1024
OPENAI_MAX_COMPLETION_TOKENS = 1024
OPENAI_REASONING_MAX_COMPLETION_TOKENS = 2048

ANTHROPIC_MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

OPENAI_MODELS = {
    "gpt-5.5": "gpt-5.5-2026-04-23",
    "o3": "o3-2025-04-16",
}

OPENAI_REASONING_MODELS = {"o3-2025-04-16", "o1", "o1-mini"}
MODEL_ORDER = ["gpt-5.5", "o3", "opus", "sonnet"]


@dataclass(frozen=True)
class Condition:
    name: str
    display_name: str
    retrieval_cost_present: bool
    obligation_present: bool
    retrieval_cost_level: str
    obligation_variant: str
    cancellation_stated: bool


COST_OBLIGATION_CONDITIONS = [
    "free_no_obligation",
    "cost_only",
    "obligation_only",
    "cost_plus_obligation",
    "obligation_only_no_cancel",
    "cost_plus_obligation_no_cancel",
]

CONDITIONS = {
    "free_no_obligation": Condition(
        name="free_no_obligation",
        display_name="Free / no obligation",
        retrieval_cost_present=False,
        obligation_present=False,
        retrieval_cost_level="free",
        obligation_variant="no_obligation",
        cancellation_stated=False,
    ),
    "cost_only": Condition(
        name="cost_only",
        display_name="Cost only",
        retrieval_cost_present=True,
        obligation_present=False,
        retrieval_cost_level="high",
        obligation_variant="no_obligation",
        cancellation_stated=False,
    ),
    "obligation_only": Condition(
        name="obligation_only",
        display_name="Obligation + cancel",
        retrieval_cost_present=False,
        obligation_present=True,
        retrieval_cost_level="free",
        obligation_variant="obligation_with_cancel",
        cancellation_stated=True,
    ),
    "cost_plus_obligation": Condition(
        name="cost_plus_obligation",
        display_name="Cost + obligation",
        retrieval_cost_present=True,
        obligation_present=True,
        retrieval_cost_level="high",
        obligation_variant="obligation_with_cancel",
        cancellation_stated=True,
    ),
    "obligation_only_no_cancel": Condition(
        name="obligation_only_no_cancel",
        display_name="Obligation only\n(no cancel)",
        retrieval_cost_present=False,
        obligation_present=True,
        retrieval_cost_level="free",
        obligation_variant="obligation_no_cancel",
        cancellation_stated=False,
    ),
    "cost_plus_obligation_no_cancel": Condition(
        name="cost_plus_obligation_no_cancel",
        display_name="Cost + obligation\n(no cancel)",
        retrieval_cost_present=True,
        obligation_present=True,
        retrieval_cost_level="high",
        obligation_variant="obligation_no_cancel",
        cancellation_stated=False,
    ),
}
