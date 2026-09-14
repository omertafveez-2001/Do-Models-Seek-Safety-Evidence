from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .config import CONDITIONS, COST_OBLIGATION_CONDITIONS, DEFAULT_RESULTS_DIR, EXPERIMENT_ID, GRAPHS_DIR
except ImportError:
    from config import CONDITIONS, COST_OBLIGATION_CONDITIONS, DEFAULT_RESULTS_DIR, EXPERIMENT_ID, GRAPHS_DIR

GRAPH_DIR = GRAPHS_DIR

PAPER_BLUE = "#4E79A7"
PAPER_RED = "#D65F4A"
PAPER_RED_DARK = "#B54735"
PAPER_GOLD = "#D9A441"
PAPER_SAGE = "#5E8B7E"
PAPER_SLATE = "#6B7280"
PAPER_TAUPE = "#9A8F84"
PAPER_GUIDE = "#C9C3BA"
LEGEND_FONT_SIZE = 14
TITLE_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 13
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.8
BAR_LABEL_FONT_SIZE = 10

MODEL_DISPLAY = {
    "gpt-5.5": "GPT-5.5",
    "o3": "o3",
    "opus": "Opus-4.8",
    "sonnet": "Sonnet-4.6",
}

MODEL_COLORS = {
    "gpt-5.5": "#5E8B7E",
    "o3": "#4E79A7",
    "opus": "#D65F4A",
    "sonnet": "#7FA7D8",
}
EFFECT_ORDER = [
    "cost_effect_no_obligation",
    "cost_effect_obligation",
    "cost_effect_obligation_no_cancel",
    "obligation_effect_free",
    "obligation_effect_cost",
    "obligation_no_cancel_effect_free",
    "obligation_no_cancel_effect_cost",
    "cancellation_consequence_effect_free",
    "cancellation_consequence_effect_cost",
]
PLOTTED_EFFECT_ORDER = EFFECT_ORDER
MAIN_TEXT_EFFECT_ORDER = [
    "cost_effect_no_obligation",
    "cost_effect_obligation",
    "obligation_effect_cost",
    "cancellation_consequence_effect_free",
    "cancellation_consequence_effect_cost",
]
EFFECT_DISPLAY = {
    "cost_effect_no_obligation": "Cost\n(no obligation)",
    "cost_effect_obligation": "Cost\n(cancel)",
    "cost_effect_obligation_no_cancel": "Cost\n(no cancel)",
    "obligation_effect_free": "Obligation\n(free)",
    "obligation_effect_cost": "Obligation\n(costly)",
    "obligation_no_cancel_effect_free": "Obligation\n(no cancel,\nfree)",
    "obligation_no_cancel_effect_cost": "Obligation\n(no cancel,\ncostly)",
    "cancellation_consequence_effect_free": "Cancellation\n(free)",
    "cancellation_consequence_effect_cost": "Cancellation\n(costly)",
}
MAIN_TEXT_EFFECT_DISPLAY = {
    "cost_effect_no_obligation": "Cost",
    "cost_effect_obligation": "Cost + cancel",
    "obligation_effect_cost": "Obligation (costly)",
    "cancellation_consequence_effect_free": "Cancellation (free)",
    "cancellation_consequence_effect_cost": "Cancellation (costly)",
}

DIRECTION_ORDER = ["Lower inspection", "No change", "Higher inspection"]
DIRECTION_COLORS = {
    "Lower inspection": PAPER_RED_DARK,
    "No change": PAPER_SLATE,
    "Higher inspection": PAPER_SAGE,
}


def _default_save_path(filename: str) -> Path:
    return GRAPH_DIR / filename


def _as_path(path: str | Path | None, default: Path) -> Path:
    return Path(path) if path is not None else default


def _save_or_show(save_path: str | Path | None):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()


def _appendix_contrast_save_path(save_path: str | Path | None) -> Path:
    if save_path is None:
        return _default_save_path("scenario_level_contrasts_full.pdf")
    path = Path(save_path)
    return path.with_name("scenario_level_contrasts_full.pdf")


def _wilson_errors(successes, trials, scale: float = 1.0) -> np.ndarray:
    """Return asymmetric 95% Wilson-score errors for binomial proportions."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    proportions = np.divide(successes, trials, out=np.zeros_like(successes), where=trials > 0)
    safe_trials = np.where(trials > 0, trials, 1)
    z_squared = 1.959963984540054**2
    denominator = 1 + z_squared / safe_trials
    center = (proportions + z_squared / (2 * safe_trials)) / denominator
    half_width = 1.959963984540054 / denominator * np.sqrt(
        proportions * (1 - proportions) / safe_trials + z_squared / (4 * safe_trials**2)
    )
    lower = np.where(trials > 0, np.clip(center - half_width, 0, 1), np.nan)
    upper = np.where(trials > 0, np.clip(center + half_width, 0, 1), np.nan)
    return np.vstack((
        np.maximum(proportions - lower, 0) * scale,
        np.maximum(upper - proportions, 0) * scale,
    ))


def load_results(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Load all per-model cost/obligation JSON result files into one DataFrame."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f"*_{EXPERIMENT_ID}.json"))
    if not files:
        raise FileNotFoundError(f"No *_{EXPERIMENT_ID}.json files found in {results_dir}")

    rows = []
    for path in files:
        with open(path) as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        rows.extend(data)

    if not rows:
        raise ValueError(f"No result rows found in {results_dir}")
    return pd.DataFrame(rows)


def inspection_rate_by_condition(results: pd.DataFrame | None = None, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Compute P(INSPECT) and UNCLEAR rate by model and condition."""
    df = load_results(results_dir) if results is None else results.copy()
    valid = df[df["decision"].isin(["INSPECT", "SKIP"])].copy()

    rates = (
        valid.groupby(["model", "condition"])["decision"]
        .agg(
            inspect_rate=lambda s: (s == "INSPECT").mean(),
            inspect_count=lambda s: (s == "INSPECT").sum(),
            n_valid="size",
        )
        .reset_index()
    )
    unclear = (
        df.groupby(["model", "condition"])["decision"]
        .apply(lambda s: (s == "UNCLEAR").mean())
        .reset_index(name="unclear_rate")
    )
    counts = (
        df.groupby(["model", "condition"])
        .size()
        .reset_index(name="n_rows")
    )
    out = rates.merge(unclear, on=["model", "condition"], how="outer").merge(
        counts, on=["model", "condition"], how="outer"
    )
    out["condition_display_name"] = out["condition"].map(lambda c: CONDITIONS[c].display_name)
    out["model_display"] = out["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    return out.sort_values(["model", "condition"])


def inspection_rate_table(results: pd.DataFrame | None = None, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Return the paper-style condition table with one row per model."""
    summary = inspection_rate_by_condition(results, results_dir)
    table = summary.pivot(index="model_display", columns="condition", values="inspect_rate")
    table = table.reindex(columns=COST_OBLIGATION_CONDITIONS)
    table = table.rename(columns={c: CONDITIONS[c].display_name for c in COST_OBLIGATION_CONDITIONS})
    return (table * 100).round(1).reset_index().rename(columns={"model_display": "Model"})


def matched_effects(results: pd.DataFrame | None = None, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Compute matched scenario-level deltas for the 2x3 cost/obligation design."""
    df = load_results(results_dir) if results is None else results.copy()
    valid = df[df["decision"].isin(["INSPECT", "SKIP"])].copy()
    rates = (
        valid.groupby(["model", "scenario_id", "condition"])["decision"]
        .apply(lambda s: (s == "INSPECT").mean())
        .reset_index(name="inspect_fraction")
    )
    wide = rates.pivot_table(
        index=["model", "scenario_id"],
        columns="condition",
        values="inspect_fraction",
    ).reset_index()
    wide = wide.dropna(subset=COST_OBLIGATION_CONDITIONS)
    wide["cost_effect_no_obligation"] = wide["cost_only"] - wide["free_no_obligation"]
    wide["cost_effect_obligation"] = wide["cost_plus_obligation"] - wide["obligation_only"]
    wide["cost_effect_obligation_no_cancel"] = (
        wide["cost_plus_obligation_no_cancel"] - wide["obligation_only_no_cancel"]
    )
    wide["obligation_effect_free"] = wide["obligation_only"] - wide["free_no_obligation"]
    wide["obligation_effect_cost"] = wide["cost_plus_obligation"] - wide["cost_only"]
    wide["obligation_no_cancel_effect_free"] = (
        wide["obligation_only_no_cancel"] - wide["free_no_obligation"]
    )
    wide["obligation_no_cancel_effect_cost"] = (
        wide["cost_plus_obligation_no_cancel"] - wide["cost_only"]
    )
    wide["cancellation_consequence_effect_free"] = (
        wide["obligation_only"] - wide["obligation_only_no_cancel"]
    )
    wide["cancellation_consequence_effect_cost"] = (
        wide["cost_plus_obligation"] - wide["cost_plus_obligation_no_cancel"]
    )
    wide["model_display"] = wide["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    return wide


def mean_effects_by_model(results: pd.DataFrame | None = None, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Return mean matched effects by model."""
    effects = matched_effects(results, results_dir)
    means = effects.groupby(["model", "model_display"])[EFFECT_ORDER].mean().reset_index()
    return means


def direction_of_change_counts(results: pd.DataFrame | None = None, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Count scenarios with lower/no-change/higher inspection for each effect type."""
    effects = matched_effects(results, results_dir)
    rows = []
    for _, row in effects.iterrows():
        for effect in PLOTTED_EFFECT_ORDER:
            value = row[effect]
            direction = "Lower inspection" if value < 0 else "Higher inspection" if value > 0 else "No change"
            rows.append({
                "model": row["model"],
                "model_display": row["model_display"],
                "scenario_id": row["scenario_id"],
                "effect_type": effect,
                "effect_display": EFFECT_DISPLAY[effect],
                "direction": direction,
                "effect_value": value,
            })
    return pd.DataFrame(rows)


def plot_inspection_rate_by_condition(
    results: pd.DataFrame | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Combined figure: inspection rates and direction-of-change counts, split by provider family."""
    summary = inspection_rate_by_condition(results, results_dir)
    direction = direction_of_change_counts(results, results_dir)
    model_order = list(dict.fromkeys(summary["model"]))
    provider_groups = [
        [model for model in model_order if model in {"gpt-5.5", "o3"}],
        [model for model in model_order if model in {"sonnet", "opus"}],
    ]
    condition_labels = [
        CONDITIONS[c].display_name
        .replace(" / ", " /\n")
        .replace(" + ", " +\n")
        .replace(" only", "\nonly")
        .replace(" (", "\n(")
        for c in COST_OBLIGATION_CONDITIONS
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20.5, 9.2),
        dpi=300,
        sharey="col",
        gridspec_kw={"width_ratios": [1, 1]},
    )
    (ax1, ax2), (ax3, ax4) = axes

    def _draw_inspection_panel(ax, panel_models: list[str], row_label: str, show_xlabels: bool):
        x = np.arange(len(COST_OBLIGATION_CONDITIONS)) * 0.88
        width = 0.145
        spacing = 0.29
        for idx, model in enumerate(panel_models):
            model_rows = summary[summary["model"] == model].set_index("condition").reindex(COST_OBLIGATION_CONDITIONS)
            vals = model_rows["inspect_rate"].to_numpy() * 100
            successes = model_rows["inspect_count"].to_numpy()
            trials = model_rows["n_valid"].to_numpy()
            offset = (idx - (len(panel_models) - 1) / 2) * spacing
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=MODEL_DISPLAY.get(model, model),
                color=MODEL_COLORS.get(model, PAPER_TAUPE),
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
            ax.errorbar(
                x + offset,
                vals,
                yerr=_wilson_errors(successes, trials, scale=100),
                fmt="none",
                ecolor="#2F2F2F",
                elinewidth=1,
                capsize=2.5,
                zorder=4,
            )
            for bar_idx, (bar, value) in enumerate(zip(bars, vals)):
                if value <= 0:
                    continue
                first_blue_full = model in {"o3", "sonnet"} and bar_idx == 0 and value >= 99.5
                ax.text(
                    bar.get_x() + bar.get_width() / 2 + (width * 0.18 if first_blue_full else 0.0),
                    value + 3,
                    f"{value:.0f}%",
                    ha="left" if first_blue_full else "center",
                    va="bottom",
                    fontsize=10,
                    color="#2F2F2F",
                )

        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{tick:.0f}%" for tick in np.arange(0, 101, 20)], fontsize=11)
        ax.set_xticks(x)
        if show_xlabels:
            ax.set_xticklabels(condition_labels, fontsize=11)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Inspection rate (%)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_axisbelow(True)
        ax.legend(
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.40 if row_label == "Anthropic" else -0.10),
            ncol=len(panel_models),
            handlelength=1.6,
            columnspacing=1.4,
        )
        ax.text(-0.18, 0.5, row_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=12)

    def _draw_direction_panel(
        ax,
        panel_models: list[str],
        effects: list[str],
        effect_labels: dict[str, str],
        show_xlabels: bool,
        show_direction_legend: bool = False,
        model_label_font_size: float = 9,
    ):
        subset = direction[direction["model"].isin(panel_models)].copy()
        x = np.arange(len(effects)) * 1.1
        width = 0.28
        spacing = 0.44
        for idx, model in enumerate(panel_models):
            model_subset = subset[subset["model"] == model]
            base_x = x + (idx - (len(panel_models) - 1) / 2) * spacing
            bottom = np.zeros(len(effects))
            for label in DIRECTION_ORDER:
                vals = []
                successes = []
                trials = []
                for effect in effects:
                    effect_subset = model_subset[model_subset["effect_type"] == effect]
                    denom = len(effect_subset)
                    num = (effect_subset["direction"] == label).sum()
                    vals.append(num / denom * 100 if denom else 0.0)
                    successes.append(num)
                    trials.append(denom)
                bars = ax.bar(
                    base_x,
                    vals,
                    width,
                    bottom=bottom,
                    label=label if idx == 0 and show_direction_legend else None,
                    color=DIRECTION_COLORS[label],
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                )
                for bar, val, base in zip(bars, vals, bottom):
                    if val >= 8:
                        if label in {"Lower inspection", "No change"}:
                            text_color = "white"
                        else:
                            text_color = "black"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            base + val / 2,
                            f"{val:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=text_color,
                        )
                bottom += vals

            # Name each bar directly so model identity does not depend on position or fill.
            for bar_x in base_x:
                ax.text(
                    bar_x,
                    102,
                    MODEL_DISPLAY.get(model, model),
                    ha="center",
                    va="bottom",
                    fontsize=model_label_font_size,
                    color="#2F2F2F",
                )

        ax.set_ylim(0, 108)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{tick:.0f}%" for tick in np.arange(0, 101, 20)], fontsize=11)
        ax.set_xticks(x)
        if show_xlabels:
            ax.set_xticklabels([effect_labels[e] for e in effects], fontsize=11)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Scenarios (%)", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=10)
        if show_direction_legend:
            direction_handles = [
                Patch(facecolor=DIRECTION_COLORS[label], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, label=label)
                for label in DIRECTION_ORDER
            ]
            ax.legend(
                handles=direction_handles,
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=3,
                handlelength=1.6,
                columnspacing=1.4,
            )
    _draw_inspection_panel(ax1, provider_groups[0], "OpenAI", False)
    _draw_inspection_panel(ax3, provider_groups[1], "Anthropic", True)
    _draw_direction_panel(ax2, provider_groups[0], MAIN_TEXT_EFFECT_ORDER, MAIN_TEXT_EFFECT_DISPLAY, False)
    _draw_direction_panel(
        ax4,
        provider_groups[1],
        MAIN_TEXT_EFFECT_ORDER,
        MAIN_TEXT_EFFECT_DISPLAY,
        True,
        show_direction_legend=True,
        model_label_font_size=8,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30, top=0.90, hspace=0.38, wspace=0.14)
    _save_or_show(save_path or _default_save_path("inspection_rate_by_condition.png"))

    appendix_path = _appendix_contrast_save_path(save_path)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharey=True)
    ax_openai, ax_anthropic = axes
    _draw_direction_panel(ax_openai, provider_groups[0], PLOTTED_EFFECT_ORDER, EFFECT_DISPLAY, False)
    _draw_direction_panel(
        ax_anthropic,
        provider_groups[1],
        PLOTTED_EFFECT_ORDER,
        EFFECT_DISPLAY,
        True,
        show_direction_legend=True,
        model_label_font_size=8,
    )
    ax_openai.text(-0.06, 0.5, "OpenAI", transform=ax_openai.transAxes, rotation=90, va="center", ha="center", fontsize=12)
    ax_anthropic.text(-0.06, 0.5, "Anthropic", transform=ax_anthropic.transAxes, rotation=90, va="center", ha="center", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, top=0.95, hspace=0.32)
    _save_or_show(appendix_path)
    return summary.round({"inspect_rate": 3, "unclear_rate": 3})


def plot_matched_effect_distributions(
    results: pd.DataFrame | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Graph 2: matched scenario-level delta distributions for the split 2x3 effects, faceted by model."""
    effects = matched_effects(results, results_dir)
    model_order = list(dict.fromkeys(effects["model"]))
    ncols = min(2, len(model_order))
    nrows = int(np.ceil(len(model_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), dpi=300, sharey=True)
    axes = np.array(axes).reshape(-1)
    x = np.arange(len(PLOTTED_EFFECT_ORDER))

    for ax, model in zip(axes, model_order):
        model_df = effects[effects["model"] == model]
        data = [model_df[effect].dropna().to_numpy() for effect in PLOTTED_EFFECT_ORDER]
        ax.boxplot(data, positions=x, widths=0.5, patch_artist=True)
        rng = np.random.default_rng(11)
        for idx, vals in enumerate(data):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), x[idx]) + jitter,
                vals,
                s=18,
                alpha=0.65,
                color=MODEL_COLORS.get(model, PAPER_TAUPE),
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=TITLE_FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([EFFECT_DISPLAY[e] for e in PLOTTED_EFFECT_ORDER], rotation=0, ha="center", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(model_order):]:
        ax.axis("off")
    for ax in axes[::ncols]:
        ax.set_ylabel("Scenario-level delta")
    fig.suptitle("Matched Effect Distributions", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    _save_or_show(save_path or _default_save_path("matched_effect_distributions.png"))
    return effects.round({effect: 3 for effect in EFFECT_ORDER})


def plot_mean_effects_by_model(
    results: pd.DataFrame | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Graph 3: mean matched deltas by model for the split 2x3 design."""
    means = mean_effects_by_model(results, results_dir)
    model_order = list(means["model"])
    x = np.arange(len(model_order))
    width = min(0.18, 0.8 / len(EFFECT_ORDER))
    effect_colors = {
        "cost_effect_no_obligation": PAPER_RED,
        "cost_effect_obligation": "#E28743",
        "cost_effect_obligation_no_cancel": "#F0B35B",
        "obligation_effect_free": PAPER_GOLD,
        "obligation_effect_cost": "#E7C96A",
        "obligation_no_cancel_effect_free": PAPER_BLUE,
        "obligation_no_cancel_effect_cost": "#7AA6D1",
        "cancellation_consequence_effect_free": PAPER_SAGE,
        "cancellation_consequence_effect_cost": "#8FB9AE",
    }

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    for idx, effect in enumerate(EFFECT_ORDER):
        offset = (idx - (len(EFFECT_ORDER) - 1) / 2) * width
        ax.bar(x + offset, means[effect].to_numpy(), width, label=EFFECT_DISPLAY[effect], color=effect_colors[effect], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_order])
    ax.set_ylabel("Mean matched delta", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title("Mean Causal Effects by Model", fontsize=TITLE_FONT_SIZE)
    ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(EFFECT_ORDER))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    _save_or_show(save_path or _default_save_path("mean_effects_by_model.png"))
    return means.round({effect: 3 for effect in EFFECT_ORDER})


def make_all_plots(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    graph_dir: str | Path = GRAPH_DIR,
) -> dict[str, pd.DataFrame]:
    """Generate all required visualizations and return their backing DataFrames."""
    graph_dir = Path(graph_dir)
    results = load_results(results_dir)
    return {
        "inspection_rate_by_condition": plot_inspection_rate_by_condition(
            results=results,
            save_path=graph_dir / "inspection_rate_by_condition.png",
        ),
        "matched_effect_distributions": plot_matched_effect_distributions(
            results=results,
            save_path=graph_dir / "matched_effect_distributions.png",
        ),
        "mean_effects_by_model": plot_mean_effects_by_model(
            results=results,
            save_path=graph_dir / "mean_effects_by_model.png",
        ),
    }
