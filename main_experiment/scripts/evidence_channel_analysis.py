from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_experiment.scripts.utils import (
    LEGEND_FONT_SIZE,
    PAPER_BLUE,
    PAPER_GOLD,
    PAPER_RED,
    PAPER_SAGE,
    PAPER_SLATE,
    majority_vote,
    wilson_errors,
)


MATCH_KEYS = ("domain", "cost_label", "severity_label", "probability_label")

POST_LABEL_COLORS = {
    "precautionary": PAPER_BLUE,
    "obligation_aware": PAPER_SLATE,
    "rationalization": "#B54735",
    "responsibility_diffusion": PAPER_RED,
    "objective_override": PAPER_GOLD,
    "genuine_mitigation": PAPER_SAGE,
    "performative_mitigation": PAPER_GOLD,
}

POST_MODE_CONFIG = {
    "halt": {
        "decision": "A",
        "title": "Halt Deployment and Address Issue",
        "labels": ["precautionary", "obligation_aware"],
    },
    "proceed despite finding": {
        "decision": "B",
        "title": "Proceed Despite Finding",
        "labels": ["rationalization", "responsibility_diffusion", "objective_override"],
    },
    "proceed with partial mitigation": {
        "decision": "C",
        "title": "Proceed With Partial Mitigation",
        "labels": ["genuine_mitigation", "performative_mitigation"],
    },
}

VARIANT_ORDER = ["offered_report", "report_discovery", "evidence_discovery"]
VARIANT_DISPLAY = {
    "offered_report": "Offered Report",
    "report_discovery": "Report Discovery",
    "evidence_discovery": "Evidence Discovery",
}
VARIANT_COLORS = {
    "offered_report": PAPER_BLUE,
    "report_discovery": PAPER_GOLD,
    "evidence_discovery": PAPER_SAGE,
}
VARIANT_HATCHES = {
    "offered_report": "",
    "report_discovery": "//",
    "evidence_discovery": "..",
}
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.8
BAR_LABEL_FONT_SIZE = 10
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE_LOCAL = 13


def _experiment_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _model_key(stem: str) -> str:
    for suffix in ("_evidence_discovery", "_report_discovery", "_offered_report"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def _matches_variant(path: Path, variant: str) -> bool:
    stem = path.stem.removeprefix("post_")
    if variant == "offered_report":
        return stem.endswith("_offered_report")
    if variant == "report_discovery":
        return stem.endswith("_report_discovery") and not stem.endswith("_evidence_discovery")
    if variant == "evidence_discovery":
        return stem.endswith("_evidence_discovery")
    raise ValueError(f"Unknown variant: {variant}")


def _variant_files(directory: Path, variant: str, post: bool = False) -> list[Path]:
    files = sorted(directory.rglob("*.json"))
    if post:
        files = [f for f in files if f.name.startswith("post_")]
    else:
        files = [f for f in files if not f.name.startswith("post_")]
    return [f for f in files if _matches_variant(f, variant)]


def _default_variant_dir(kind: str, variant: str) -> Path:
    if kind == "results":
        kind = "runs"
    base = _experiment_root() / kind
    return base / variant


def _variant_dirs(kind: str, overrides: dict[str, str | Path | None]) -> dict[str, Path]:
    return {
        variant: Path(overrides.get(variant) or _default_variant_dir(kind, variant))
        for variant in VARIANT_ORDER
    }


def _files_by_model_for_variants(
    kind: str,
    overrides: dict[str, str | Path | None],
    post: bool = False,
    require_offered_report: bool = True,
) -> dict[str, dict[str, Path]]:
    variant_dirs = _variant_dirs(kind, overrides)
    by_variant: dict[str, dict[str, Path]] = {}

    for variant in VARIANT_ORDER:
        directory = variant_dirs[variant]
        files = _variant_files(directory, variant, post=post)
        if variant == "offered_report" and require_offered_report and not files:
            prefix = "post_*_" if post else "*_"
            raise FileNotFoundError(f"No {prefix}{variant}.json files found in {directory}")
        if files:
            key_fn = _post_model_key if post else _model_key
            by_variant[variant] = {key_fn(f.stem): f for f in files}

    if "offered_report" not in by_variant:
        raise FileNotFoundError("No offered_report files found.")

    return by_variant


def _common_model_keys(by_variant: dict[str, dict[str, Path]]) -> list[str]:
    if "offered_report" not in by_variant:
        raise ValueError("offered_report files are required for comparisons.")

    comparison_variants = [v for v in VARIANT_ORDER[1:] if v in by_variant]
    if not comparison_variants:
        raise ValueError("No discovery-variant files found to compare against offered_report.")

    offered_report_keys = set(by_variant["offered_report"])
    model_keys = set()
    for variant in comparison_variants:
        model_keys |= offered_report_keys & set(by_variant[variant])
    if not model_keys:
        variants = "/".join(["offered_report", *comparison_variants])
        raise ValueError(f"No matching {variants} model files found.")
    return sorted(model_keys)


def _available_variants_for_model(
    by_variant: dict[str, dict[str, Path]],
    model_key: str,
) -> list[str]:
    return [
        variant for variant in VARIANT_ORDER
        if variant in by_variant and model_key in by_variant[variant]
    ]


def _active_variants(by_variant: dict[str, dict[str, Path]]) -> list[str]:
    return [variant for variant in VARIANT_ORDER if variant in by_variant]


def _post_model_key(stem: str) -> str:
    return _model_key(stem.removeprefix("post_"))


def _short_label(key: str) -> str:
    labels = {
        "gpt-5.5": "GPT-5.5",
        "o3": "o3",
        "opus": "Opus-4.8",
        "sonnet": "Sonnet-4.6",
    }
    return labels.get(key, key)


def _load_json(path: Path) -> list[dict]:
    with open(path) as fp:
        return json.load(fp)


def plot_offered_report_vs_report_discovery_inspection_rate(
    mode: str = "all",
    offered_report_results_dir: str | Path = None,
    report_discovery_results_dir: str | Path = None,
    evidence_discovery_results_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Grouped bar chart comparing offered_report against discovery variants.

    mode:
    - "all": use all rollouts
    - "majority": use one majority-vote decision per scenario

    Returns a DataFrame with inspection rates and discovery-minus-offered_report deltas
    in percentage points.
    """
    if mode not in ("all", "majority"):
        raise ValueError("mode must be 'all' or 'majority'")
    by_variant = _files_by_model_for_variants(
        "runs",
        {
            "offered_report": offered_report_results_dir,
            "report_discovery": report_discovery_results_dir,
            "evidence_discovery": evidence_discovery_results_dir,
        },
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    def _inspect_rate(path: Path) -> float:
        data = _load_json(path)
        if mode == "majority":
            data = majority_vote(data)
        total = len(data) or 1
        return sum(1 for r in data if r.get("decision") == "INSPECT") / total * 100

    rows = []
    for key in model_keys:
        model_variants = _available_variants_for_model(by_variant, key)
        offered_report_rate = _inspect_rate(by_variant["offered_report"][key])
        row = {
            "Model": _short_label(key),
            "model_key": key,
            "Offered Report inspect rate": round(offered_report_rate, 1),
        }
        for variant in model_variants:
            if variant == "offered_report":
                continue
            rate = _inspect_rate(by_variant[variant][key])
            display = VARIANT_DISPLAY[variant]
            row[f"{display} inspect rate"] = round(rate, 1)
            row[f"Delta {variant}-offered_report pp"] = round(rate - offered_report_rate, 1)
        rows.append(row)

    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    width = min(0.24, 0.8 / len(variants))
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    for offset, variant in zip(offsets, variants):
        display = VARIANT_DISPLAY[variant]
        bars = ax.bar(
            x + offset,
            df[f"{display} inspect rate"],
            width,
            label=display,
            color=VARIANT_COLORS[variant],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=BAR_LABEL_FONT_SIZE)

    for i, row in df.iterrows():
        labels = []
        row_variants = []
        for variant in variants:
            if variant == "offered_report":
                continue
            delta_col = f"Delta {variant}-offered_report pp"
            rate_col = f"{VARIANT_DISPLAY[variant]} inspect rate"
            if delta_col not in row or pd.isna(row[delta_col]):
                continue
            labels.append(f"{variant.replace('_discovery', '')}: {row[delta_col]:+.1f} pp")
            row_variants.append(variant)
        if labels:
            y = max(
                row[f"{VARIANT_DISPLAY[v]} inspect rate"]
                for v in ["offered_report", *row_variants]
                if f"{VARIANT_DISPLAY[v]} inspect rate" in row and not pd.isna(row[f"{VARIANT_DISPLAY[v]} inspect rate"])
            ) + 8
            ax.text(i, y, "\n".join(labels), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], fontsize=10)
    ax.set_ylabel("P(INSPECT) (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(
        0,
        min(
            122,
            max(
                df[f"{VARIANT_DISPLAY[v]} inspect rate"].max()
                for v in variants
                if f"{VARIANT_DISPLAY[v]} inspect rate" in df
            ) + 22,
        ),
    )
    ax.legend(fontsize=LEGEND_FONT_SIZE_LOCAL, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=len(variants), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df


def matched_report_discovery_deltas(
    offered_report_results_dir: str | Path = None,
    report_discovery_results_dir: str | Path = None,
    evidence_discovery_results_dir: str | Path = None,
) -> pd.DataFrame:
    """
    Compute matched scenario-level discovery-variant effects.

    Each row is one matched model/scenario parameter combination. The delta is:
    discovery-variant inspect fraction - offered_report inspect fraction, where each
    fraction is computed from all rollout rows for that scenario.
    """
    by_variant = _files_by_model_for_variants(
        "runs",
        {
            "offered_report": offered_report_results_dir,
            "report_discovery": report_discovery_results_dir,
            "evidence_discovery": evidence_discovery_results_dir,
        },
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    def _fractions_by_scenario(rows: list[dict]) -> dict[tuple, float]:
        grouped = {}
        for row in rows:
            key = tuple(row.get(k) for k in MATCH_KEYS)
            grouped.setdefault(key, []).append(row)
        return {
            key: sum(1 for r in vals if r.get("decision") == "INSPECT") / len(vals)
            for key, vals in grouped.items()
        }

    rows = []
    for model_key in model_keys:
        discovery_variants = [
            v for v in _available_variants_for_model(by_variant, model_key)
            if v != "offered_report"
        ]
        offered_report = _fractions_by_scenario(_load_json(by_variant["offered_report"][model_key]))
        discovery_by_variant = {
            variant: _fractions_by_scenario(_load_json(by_variant[variant][model_key]))
            for variant in discovery_variants
        }

        for variant in discovery_variants:
            scenario_keys = set(offered_report) & set(discovery_by_variant[variant])
            for scenario_key in sorted(scenario_keys):
                offered_report_fraction = offered_report[scenario_key]
                discovery_fraction = discovery_by_variant[variant][scenario_key]
                delta = discovery_fraction - offered_report_fraction
                domain, cost, severity, probability = scenario_key
                rows.append({
                    "Model": _short_label(model_key),
                    "model_key": model_key,
                    "variant": variant,
                    "Variant": VARIANT_DISPLAY[variant],
                    "domain": domain,
                    "cost_label": cost,
                    "severity_label": severity,
                    "probability_label": probability,
                    "scenario_key": " · ".join(scenario_key),
                    "offered_report_inspect_fraction": offered_report_fraction,
                    "discovery_inspect_fraction": discovery_fraction,
                    f"{variant}_inspect_fraction": discovery_fraction,
                    "delta": delta,
                    "delta_rollouts": round(delta * 5),
                    "direction": "higher inspection" if delta > 0 else "lower inspection" if delta < 0 else "no change",
                })

    if not rows:
        raise ValueError("No matched scenarios found.")
    return pd.DataFrame(rows)


def plot_matched_report_discovery_deltas(
    offered_report_results_dir: str | Path = None,
    report_discovery_results_dir: str | Path = None,
    evidence_discovery_results_dir: str | Path = None,
    save_path: str | Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Plot matched scenario-level direction-of-change shares.

    Returns:
    - matched_df: one row per matched model/scenario with delta in [-1, 1]
    - direction_df: percentage of scenarios with lower/no-change/higher inspection
      per model
    """
    matched_df = matched_report_discovery_deltas(
        offered_report_results_dir=offered_report_results_dir,
        report_discovery_results_dir=report_discovery_results_dir,
        evidence_discovery_results_dir=evidence_discovery_results_dir,
    )

    model_order = list(dict.fromkeys(matched_df["Model"]))
    variant_order = [VARIANT_DISPLAY[v] for v in VARIANT_ORDER[1:] if v in set(matched_df["variant"])]
    direction_order = ["lower inspection", "no change", "higher inspection"]
    direction_colors = {
        "lower inspection": PAPER_RED,
        "no change": "#8F8F8F",
        "higher inspection": PAPER_SAGE,
    }

    direction_counts = (
        matched_df.groupby(["Variant", "Model", "direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=direction_order, fill_value=0)
    )
    direction_pct = direction_counts.div(direction_counts.sum(axis=1), axis=0) * 100
    direction_df = (
        direction_pct.reset_index()
        .rename(columns={
            "lower inspection": "Lower inspection (%)",
            "no change": "No change (%)",
            "higher inspection": "Higher inspection (%)",
        })
    )

    fig, ax_dir = plt.subplots(figsize=(8.5, 5), dpi=300)

    y_labels = [
        f"{model} · {variant}"
        for model in model_order
        for variant in variant_order
        if not matched_df[(matched_df["Model"] == model) & (matched_df["Variant"] == variant)].empty
    ]

    direction_plot = (
        direction_df.set_index(["Model", "Variant"])
        .reindex(
            pd.MultiIndex.from_tuples(
                [(label.split(" · ", 1)[0], label.split(" · ", 1)[1]) for label in y_labels],
                names=["Model", "Variant"],
            )
        )
        .fillna(0)
    )
    direction_count_plot = (
        direction_counts.reindex(
            pd.MultiIndex.from_tuples(
                [(label.split(" · ", 1)[0], label.split(" · ", 1)[1]) for label in y_labels],
                names=["Model", "Variant"],
            ),
            fill_value=0,
        )
    )
    y_positions = np.arange(len(y_labels))
    left = np.zeros(len(y_labels))
    for direction in direction_order:
        col = {
            "lower inspection": "Lower inspection (%)",
            "no change": "No change (%)",
            "higher inspection": "Higher inspection (%)",
        }[direction]
        vals = direction_plot[col].to_numpy()
        counts = direction_count_plot[direction].to_numpy()
        trials = direction_count_plot.sum(axis=1).to_numpy()
        bars = ax_dir.barh(
            y_positions,
            vals,
            left=left,
            label=direction.title(),
            color=direction_colors[direction],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax_dir.errorbar(
            left + vals,
            y_positions,
            xerr=wilson_errors(counts, trials, scale=100),
            fmt="none",
            ecolor="#2F2F2F",
            elinewidth=1,
            capsize=2.5,
            zorder=4,
        )
        for bar, val in zip(bars, vals):
            if val >= 7:
                ax_dir.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if direction != "no change" else "black",
                )
        left += vals

    ax_dir.set_xlim(0, 100)
    ax_dir.set_yticks(y_positions)
    ax_dir.set_yticklabels(y_labels)
    ax_dir.set_xlabel("Matched scenarios (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_dir.tick_params(axis="x", labelsize=10)
    ax_dir.tick_params(axis="y", labelsize=9)
    ax_dir.legend(fontsize=LEGEND_FONT_SIZE_LOCAL, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    ax_dir.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.96)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return matched_df, direction_df


def plot_offered_report_vs_report_discovery_by_cost(
    offered_report_results_dir: str | Path = None,
    report_discovery_results_dir: str | Path = None,
    evidence_discovery_results_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Plot offered_report vs discovery-variant inspection rates by cost level for each model.

    Returns a DataFrame with one row per model/cost/variant containing rates and
    discovery-minus-offered_report gaps in percentage points.
    """
    matched_df = matched_report_discovery_deltas(
        offered_report_results_dir=offered_report_results_dir,
        report_discovery_results_dir=report_discovery_results_dir,
        evidence_discovery_results_dir=evidence_discovery_results_dir,
    )

    cost_order = ["free", "low", "medium", "high", "extreme"]
    cost_labels = {
        "free": "Free",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "extreme": "Extreme /\nmandatory review",
    }
    model_order = list(dict.fromkeys(matched_df["Model"]))

    rows = []
    for model in model_order:
        model_df = matched_df[matched_df["Model"] == model]
        for cost in cost_order:
            cost_df = model_df[model_df["cost_label"] == cost]
            if cost_df.empty:
                continue
            offered_report_rate = cost_df["offered_report_inspect_fraction"].mean() * 100
            for variant in [v for v in VARIANT_ORDER[1:] if v in set(cost_df["variant"])]:
                variant_df = cost_df[cost_df["variant"] == variant]
                discovery_rate = variant_df["discovery_inspect_fraction"].mean() * 100
                rows.append({
                    "Model": model,
                    "variant": variant,
                    "Variant": VARIANT_DISPLAY[variant],
                    "cost_label": cost,
                    "Cost": cost_labels[cost].replace("\n", " "),
                    "Offered Report inspect rate": offered_report_rate,
                    "Discovery inspect rate": discovery_rate,
                    "Gap discovery-offered_report pp": discovery_rate - offered_report_rate,
                    "n_scenarios": len(variant_df),
                })

    cost_df = pd.DataFrame(rows)
    if cost_df.empty:
        raise ValueError("No matched cost-level data found.")

    n_models = len(model_order)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.2 * nrows),
        dpi=300,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x = np.arange(len(cost_order))

    for ax, model in zip(axes, model_order):
        model_cost = (
            cost_df[cost_df["Model"] == model]
        )
        offered_report = (
            model_cost.drop_duplicates("cost_label")
            .set_index("cost_label")
            .reindex(cost_order)["Offered Report inspect rate"]
            .to_numpy()
        )

        ax.plot(
            x,
            offered_report,
            marker="o",
            linewidth=2,
            color=PAPER_BLUE,
            label="Offered Report",
        )

        for variant in [v for v in VARIANT_ORDER[1:] if v in set(model_cost["variant"])]:
            variant_cost = (
                model_cost[model_cost["variant"] == variant]
                .set_index("cost_label")
                .reindex(cost_order)
            )
            discovery = variant_cost["Discovery inspect rate"].to_numpy()
            gap = variant_cost["Gap discovery-offered_report pp"].to_numpy()
            ax.plot(
                x,
                discovery,
                marker="o",
                linewidth=2,
                color=VARIANT_COLORS[variant],
                label=VARIANT_DISPLAY[variant],
            )
            for i, (base, value, diff) in enumerate(zip(offered_report, discovery, gap)):
                if np.isnan(base) or np.isnan(value):
                    continue
                ax.vlines(i, base, value, color=VARIANT_COLORS[variant], alpha=0.25, linewidth=1.2)
                y = max(base, value) + (4 if variant == "report_discovery" else 10)
                ax.text(
                    i,
                    y,
                    f"{diff:+.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=VARIANT_COLORS[variant],
                )

        ax.set_xticks(x)
        ax.set_xticklabels([cost_labels[c] for c in cost_order], fontsize=8)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n_models:]:
        ax.axis("off")

    axes[0].legend(fontsize=LEGEND_FONT_SIZE_LOCAL, loc="upper center",
                   bbox_to_anchor=(1.08, 1.22), ncol=3, frameon=False)
    for ax in axes[::ncols]:
        ax.set_ylabel("P(INSPECT) (%)", fontsize=AXIS_LABEL_FONT_SIZE)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return cost_df.round({
        "Offered Report inspect rate": 1,
        "Discovery inspect rate": 1,
        "Gap discovery-offered_report pp": 1,
    })


def plot_end_to_end_path_comparison(
    offered_report_results_dir: str | Path = None,
    report_discovery_results_dir: str | Path = None,
    evidence_discovery_results_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Compare end-to-end deployment paths across offered_report and discovery variants.

    Path categories:
    - SKIP
    - INSPECT -> HALT
    - INSPECT -> PROCEED
    - INSPECT -> PARTIAL MITIGATION

    Returns a DataFrame with one row per model/variant/path containing counts
    and percentages.
    """
    by_variant = _files_by_model_for_variants(
        "runs",
        {
            "offered_report": offered_report_results_dir,
            "report_discovery": report_discovery_results_dir,
            "evidence_discovery": evidence_discovery_results_dir,
        },
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    path_order = [
        "SKIP",
        "INSPECT -> HALT",
        "INSPECT -> PROCEED",
        "INSPECT -> PARTIAL MITIGATION",
    ]
    path_colors = {
        "SKIP": PAPER_RED,
        "INSPECT -> HALT": PAPER_BLUE,
        "INSPECT -> PROCEED": "#B54735",
        "INSPECT -> PARTIAL MITIGATION": PAPER_SAGE,
    }

    def _path(row: dict) -> str:
        if row.get("decision") == "SKIP":
            return "SKIP"
        if row.get("decision") != "INSPECT":
            return "UNCLEAR"
        post_decision = row.get("post_inspection_decision")
        if post_decision == "A":
            return "INSPECT -> HALT"
        if post_decision == "B":
            return "INSPECT -> PROCEED"
        if post_decision == "C":
            return "INSPECT -> PARTIAL MITIGATION"
        return "UNCLEAR"

    rows = []
    for model_key in model_keys:
        for variant in _available_variants_for_model(by_variant, model_key):
            path = by_variant[variant][model_key]
            data = _load_json(path)
            total = len(data)
            counts = {p: 0 for p in path_order}
            unclear_count = 0
            for row in data:
                path_name = _path(row)
                if path_name in counts:
                    counts[path_name] += 1
                else:
                    unclear_count += 1

            denominator = total or 1
            for path_name in path_order:
                rows.append({
                    "Model": _short_label(model_key),
                    "model_key": model_key,
                    "variant": variant,
                    "Variant": VARIANT_DISPLAY[variant],
                    "Path": path_name,
                    "count": counts[path_name],
                    "percentage": counts[path_name] / denominator * 100,
                    "total": total,
                    "unclear_count": unclear_count,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No end-to-end path data found.")

    fig, axes = plt.subplots(
        1,
        len(model_keys),
        figsize=(max(12, len(model_keys) * 3.2), 5),
        dpi=300,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax, model_key in zip(axes, model_keys):
        model_name = _short_label(model_key)
        model_df = df[df["model_key"] == model_key]
        variant_labels = [VARIANT_DISPLAY[v] for v in variants]
        bottom = np.zeros(len(variant_labels))
        x = np.arange(len(variant_labels))

        for path_name in path_order:
            vals = [
                model_df[
                    (model_df["Variant"] == variant)
                    & (model_df["Path"] == path_name)
                ]["percentage"].sum()
                for variant in variant_labels
            ]
            bars = ax.bar(
                x,
                vals,
                bottom=bottom,
                color=path_colors[path_name],
                label=path_name,
                width=0.62,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
            for bar, val, base in zip(bars, vals, bottom):
                if val >= 6:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + val / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                    )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels, fontsize=8, rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Share of all rollouts (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=LEGEND_FONT_SIZE_LOCAL,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df.round({"percentage": 1})


def plot_skip_label_comparison(
    mode: str = "percentage",
    offered_report_coded_dir: str | Path = None,
    report_discovery_coded_dir: str | Path = None,
    evidence_discovery_coded_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Compare SKIP rationale labels across offered_report and discovery variants.

    Labels:
    - expected_value
    - strategic_ignorance
    - responsibility_diffusion

    mode:
    - "count": raw number of SKIP decisions with each label
    - "percentage": percentage among SKIP decisions for that model/variant

    Returns a DataFrame with one row per model/variant/label.
    """
    if mode not in ("count", "percentage"):
        raise ValueError("mode must be 'count' or 'percentage'")
    by_variant = _files_by_model_for_variants(
        "coded",
        {
            "offered_report": offered_report_coded_dir,
            "report_discovery": report_discovery_coded_dir,
            "evidence_discovery": evidence_discovery_coded_dir,
        },
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    label_order = [
        "expected_value",
        "strategic_ignorance",
        "responsibility_diffusion",
    ]
    label_display = {
        "expected_value": "Expected value",
        "strategic_ignorance": "Strategic ignorance",
        "responsibility_diffusion": "Responsibility diffusion",
    }

    rows = []
    for model_key in model_keys:
        for variant in _available_variants_for_model(by_variant, model_key):
            path = by_variant[variant][model_key]
            data = _load_json(path)
            skips = [r for r in data if r.get("decision") == "SKIP"]
            total_skips = len(skips)
            for label in label_order:
                count = sum(1 for r in skips if r.get("rationalization") == label)
                rows.append({
                    "Model": _short_label(model_key),
                    "model_key": model_key,
                    "variant": variant,
                    "Variant": VARIANT_DISPLAY[variant],
                    "Label": label,
                    "Label text": label_display[label],
                    "count": count,
                    "percentage": count / (total_skips or 1) * 100,
                    "total_skip_decisions": total_skips,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No SKIP label data found.")

    model_order = [_short_label(k) for k in model_keys]
    variant_labels = [VARIANT_DISPLAY[v] for v in variants]
    metric = "count" if mode == "count" else "percentage"
    ylabel = "SKIP decisions" if mode == "count" else "Share among SKIP decisions (%)"

    n_models = len(model_order)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.2 * nrows),
        dpi=300,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x = np.arange(len(label_order))
    width = min(0.24, 0.8 / len(variants))
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width

    for ax, model in zip(axes, model_order):
        model_df = df[df["Model"] == model]
        for offset, variant, variant_label in zip(offsets, variants, variant_labels):
            vals = [
                model_df[
                    (model_df["Variant"] == variant_label)
                    & (model_df["Label"] == label)
                ][metric].sum()
                for label in label_order
            ]
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=variant_label,
                color=VARIANT_COLORS[variant],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
            fmt = "%.0f" if mode == "count" else "%.1f"
            ax.bar_label(bars, fmt=fmt, padding=2, fontsize=BAR_LABEL_FONT_SIZE)

        ax.set_xticks(x)
        ax.set_xticklabels([label_display[label].replace(" ", "\n") for label in label_order],
                           fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if mode == "percentage":
            ax.set_ylim(0, 105)

    for ax in axes[n_models:]:
        ax.axis("off")

    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    axes[0].legend(fontsize=LEGEND_FONT_SIZE_LOCAL, loc="upper center",
                   bbox_to_anchor=(1.08, 1.22), ncol=len(variants), frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df.round({"percentage": 1})


def plot_inspect_label_comparison(
    mode: str = "percentage",
    offered_report_coded_dir: str | Path = None,
    report_discovery_coded_dir: str | Path = None,
    evidence_discovery_coded_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Compare INSPECT rationale labels across offered_report and discovery variants.

    Labels:
    - risk_averse
    - obligation_aware
    - expected_value

    mode:
    - "count": raw number of INSPECT decisions with each label
    - "percentage": percentage among INSPECT decisions for that model/variant

    Returns a DataFrame with one row per model/variant/label.
    """
    if mode not in ("count", "percentage"):
        raise ValueError("mode must be 'count' or 'percentage'")
    by_variant = _files_by_model_for_variants(
        "coded",
        {
            "offered_report": offered_report_coded_dir,
            "report_discovery": report_discovery_coded_dir,
            "evidence_discovery": evidence_discovery_coded_dir,
        },
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    label_order = [
        "risk_averse",
        "obligation_aware",
        "expected_value",
    ]
    label_display = {
        "risk_averse": "Risk averse",
        "obligation_aware": "Obligation aware",
        "expected_value": "Expected value",
    }

    rows = []
    for model_key in model_keys:
        for variant in _available_variants_for_model(by_variant, model_key):
            path = by_variant[variant][model_key]
            data = _load_json(path)
            inspects = [r for r in data if r.get("decision") == "INSPECT"]
            total_inspects = len(inspects)
            for label in label_order:
                count = sum(1 for r in inspects if r.get("rationalization") == label)
                rows.append({
                    "Model": _short_label(model_key),
                    "model_key": model_key,
                    "variant": variant,
                    "Variant": VARIANT_DISPLAY[variant],
                    "Label": label,
                    "Label text": label_display[label],
                    "count": count,
                    "percentage": count / (total_inspects or 1) * 100,
                    "total_inspect_decisions": total_inspects,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No INSPECT label data found.")

    model_order = [_short_label(k) for k in model_keys]
    variant_labels = [VARIANT_DISPLAY[v] for v in variants]
    metric = "count" if mode == "count" else "percentage"
    ylabel = "INSPECT decisions" if mode == "count" else "Share among INSPECT decisions (%)"

    n_models = len(model_order)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.2 * nrows),
        dpi=300,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x = np.arange(len(label_order))
    width = min(0.24, 0.8 / len(variants))
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width

    for ax, model in zip(axes, model_order):
        model_df = df[df["Model"] == model]
        for offset, variant, variant_label in zip(offsets, variants, variant_labels):
            vals = [
                model_df[
                    (model_df["Variant"] == variant_label)
                    & (model_df["Label"] == label)
                ][metric].sum()
                for label in label_order
            ]
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=variant_label,
                color=VARIANT_COLORS[variant],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
            fmt = "%.0f" if mode == "count" else "%.1f"
            ax.bar_label(bars, fmt=fmt, padding=2, fontsize=BAR_LABEL_FONT_SIZE)

        ax.set_xticks(x)
        ax.set_xticklabels([label_display[label].replace(" ", "\n") for label in label_order],
                           fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if mode == "percentage":
            ax.set_ylim(0, 105)

    for ax in axes[n_models:]:
        ax.axis("off")

    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    axes[0].legend(fontsize=LEGEND_FONT_SIZE_LOCAL, loc="upper center",
                   bbox_to_anchor=(1.08, 1.22), ncol=len(variants), frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df.round({"percentage": 1})


def plot_post_label_comparison(
    mode: str,
    offered_report_coded_dir: str | Path = None,
    report_discovery_coded_dir: str | Path = None,
    evidence_discovery_coded_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Compare post-inspection rationale-label percentages across variants.

    mode accepts:
    - "halt"
    - "proceed despite finding"
    - "proceed with partial mitigation"

    Returns a DataFrame with one row per model/variant/post label.
    """
    mode_key = mode.strip().lower()
    if mode_key not in POST_MODE_CONFIG:
        raise ValueError(
            "mode must be one of: 'halt', 'proceed despite finding', "
            "'proceed with partial mitigation'"
        )
    by_variant = _files_by_model_for_variants(
        "coded",
        {
            "offered_report": offered_report_coded_dir,
            "report_discovery": report_discovery_coded_dir,
            "evidence_discovery": evidence_discovery_coded_dir,
        },
        post=True,
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    config = POST_MODE_CONFIG[mode_key]
    decision = config["decision"]
    labels = config["labels"]

    rows = []
    for model_key in model_keys:
        for variant in _available_variants_for_model(by_variant, model_key):
            path = by_variant[variant][model_key]
            data = _load_json(path)
            branch_rows = [
                r for r in data
                if r.get("post_inspection_decision") == decision
                and r.get("post_rationalization") in labels
            ]
            total_branch_rows = len(branch_rows)
            for label in labels:
                count = sum(1 for r in branch_rows if r.get("post_rationalization") == label)
                rows.append({
                    "Model": _short_label(model_key),
                    "model_key": model_key,
                    "variant": variant,
                    "Variant": VARIANT_DISPLAY[variant],
                    "Decision": decision,
                    "Label": label,
                    "count": count,
                    "percentage": count / (total_branch_rows or 1) * 100,
                    "total_post_decisions": total_branch_rows,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No post-label comparison data found.")

    model_order = [_short_label(k) for k in model_keys]
    variant_labels = [VARIANT_DISPLAY[v] for v in variants]

    n_models = len(model_order)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.2 * nrows),
        dpi=300,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(labels))
    width = min(0.24, 0.8 / len(variants))
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width
    for ax, model in zip(axes, model_order):
        model_df = df[df["Model"] == model]
        for offset, variant, variant_label in zip(offsets, variants, variant_labels):
            vals = [
                model_df[
                    (model_df["Variant"] == variant_label)
                    & (model_df["Label"] == label)
                ]["percentage"].sum()
                for label in labels
            ]
            colors = [POST_LABEL_COLORS[label] for label in labels]
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=variant_label,
                color=colors,
                hatch=VARIANT_HATCHES[variant],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
            ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=BAR_LABEL_FONT_SIZE)

        ax.set_xticks(x)
        ax.set_xticklabels([label.replace("_", "\n") for label in labels],
                           fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n_models:]:
        ax.axis("off")

    for ax in axes[::ncols]:
        ax.set_ylabel("% of coded post decisions", fontsize=AXIS_LABEL_FONT_SIZE)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        fontsize=LEGEND_FONT_SIZE_LOCAL,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(variants),
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.88)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df.round({"percentage": 1})


def plot_post_decision_by_severity_comparison(
    mode: str = "halt",
    offered_report_results_dir: str | Path = None,
    report_discovery_results_dir: str | Path = None,
    evidence_discovery_results_dir: str | Path = None,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Compare post-inspection decision rates by severity across variants.

    Rates are conditional on inspected cases only:
    P(post_decision | inspected, severity)

    mode accepts:
    - "halt"
    - "proceed despite finding" / "proceed"
    - "partial mitigation" / "proceed with partial mitigation"
    - "all"

    Returns a DataFrame with one row per model/variant/severity/post decision.
    """
    by_variant = _files_by_model_for_variants(
        "runs",
        {
            "offered_report": offered_report_results_dir,
            "report_discovery": report_discovery_results_dir,
            "evidence_discovery": evidence_discovery_results_dir,
        },
    )
    variants = _active_variants(by_variant)
    model_keys = _common_model_keys(by_variant)

    severity_order = ["negligible", "low", "medium", "high"]
    decision_options = {
        "halt": ("A", "Halt rate"),
        "proceed despite finding": ("B", "Proceed rate"),
        "proceed": ("B", "Proceed rate"),
        "partial mitigation": ("C", "Partial mitigation rate"),
        "proceed with partial mitigation": ("C", "Partial mitigation rate"),
    }
    all_decisions = [("A", "Halt rate"), ("B", "Proceed rate"), ("C", "Partial mitigation rate")]
    mode_key = mode.strip().lower()
    if mode_key == "all":
        selected_decisions = all_decisions
    elif mode_key in decision_options:
        selected_decisions = [decision_options[mode_key]]
    else:
        raise ValueError(
            "mode must be one of: 'halt', 'proceed despite finding', 'proceed', "
            "'partial mitigation', 'proceed with partial mitigation', 'all'"
        )

    rows = []
    for model_key in model_keys:
        for variant in _available_variants_for_model(by_variant, model_key):
            path = by_variant[variant][model_key]
            data = _load_json(path)
            for severity in severity_order:
                inspected_rows = [
                    r for r in data
                    if r.get("severity_label") == severity
                    and r.get("decision") == "INSPECT"
                    and r.get("post_inspection_decision") in ("A", "B", "C")
                ]
                denominator = len(inspected_rows)
                for decision_code, title in all_decisions:
                    count = sum(
                        1 for r in inspected_rows
                        if r.get("post_inspection_decision") == decision_code
                    )
                    rows.append({
                        "Model": _short_label(model_key),
                        "model_key": model_key,
                        "variant": variant,
                        "Variant": VARIANT_DISPLAY[variant],
                        "severity_label": severity,
                        "Decision": decision_code,
                        "Decision label": title,
                        "count": count,
                        "denominator": denominator,
                        "rate": count / denominator if denominator else 0.0,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No post-decision severity comparison data found.")

    variant_labels = [VARIANT_DISPLAY[v] for v in variants]

    if mode_key == "all":
        fig, axes = plt.subplots(
            len(model_keys),
            len(selected_decisions),
            figsize=(14, max(8, len(model_keys) * 2.6)),
            dpi=300,
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes)
        for row_idx, model_key in enumerate(model_keys):
            model_name = _short_label(model_key)
            for col_idx, (decision_code, title) in enumerate(selected_decisions):
                ax = axes[row_idx, col_idx]
                for variant, variant_label in zip(variants, variant_labels):
                    plot_rows = df[
                        (df["model_key"] == model_key)
                        & (df["Variant"] == variant_label)
                        & (df["Decision"] == decision_code)
                    ].set_index("severity_label").reindex(severity_order)
                    ax.plot(
                        severity_order,
                        plot_rows["rate"],
                        marker="o",
                        linewidth=2,
                        color=VARIANT_COLORS[variant],
                        label=variant_label,
                    )
                if col_idx == 0:
                    ax.set_ylabel(f"{model_name}\nRate", fontsize=11)
                ax.set_ylim(0, 1.05)
                ax.grid(axis="y", alpha=0.2)
                ax.spines[["top", "right"]].set_visible(False)
    else:
        ncols = 2
        nrows = int(np.ceil(len(model_keys) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(12, 4 * nrows),
            dpi=300,
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).reshape(-1)
        decision_code, title = selected_decisions[0]
        for ax, model_key in zip(axes, model_keys):
            model_name = _short_label(model_key)
            for variant, variant_label in zip(variants, variant_labels):
                plot_rows = df[
                    (df["model_key"] == model_key)
                    & (df["Variant"] == variant_label)
                    & (df["Decision"] == decision_code)
                ].set_index("severity_label").reindex(severity_order)
                ax.plot(
                    severity_order,
                    plot_rows["rate"],
                    marker="o",
                    linewidth=2,
                    color=VARIANT_COLORS[variant],
                    label=variant_label,
                )
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[len(model_keys):]:
            ax.axis("off")
        for ax in axes[::ncols]:
            ax.set_ylabel("Rate among inspected cases", fontsize=AXIS_LABEL_FONT_SIZE)

    for ax in np.array(axes).reshape(-1):
        if ax.has_data():
            ax.set_xlabel("Severity", fontsize=AXIS_LABEL_FONT_SIZE)
            ax.tick_params(axis="x", labelsize=8)

    handles, labels = np.array(axes).reshape(-1)[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=LEGEND_FONT_SIZE_LOCAL,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(variants),
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.9)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df.round({"rate": 3})
