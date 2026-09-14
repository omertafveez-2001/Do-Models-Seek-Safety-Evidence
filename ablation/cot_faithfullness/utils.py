from __future__ import annotations

import json
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "ablation" / "cot_faithfullness"
DATA_DIR = EXPERIMENT_DIR / "data"
GRAPHS_DIR = EXPERIMENT_DIR / "graphs"

MODEL_ORDER = ["gpt-5.5", "o3", "sonnet", "opus"]
EDIT_ORDER = [
    "cost_removed",
    "severity_raised",
    "probability_raised",
    "obligation_removed",
    "evidence_made_report_shaped",
    "finding_made_harder_to_rationalize",
]
INITIAL_DECISION_ORDER = ["SKIP", "INSPECT", "UNCLEAR", "ERROR", "MISSING"]
POST_DECISION_ORDER = [
    "PROCEED_DESPITE_FINDING",
    "PARTIAL_MITIGATION",
    "HALT_DEPLOYMENT",
    "UNCLEAR",
    "ERROR",
    "MISSING",
]

MODEL_COLORS = {
    "gpt-5.5": "#4F8A78",
    "o3": "#5D7FA3",
    "sonnet": "#C45743",
    "opus": "#A66A2C",
}
MODEL_DISPLAY = {
    "gpt-5.5": "GPT-5.5",
    "o3": "o3",
    "sonnet": "Sonnet-4.6",
    "opus": "Opus-4.8",
}
METRIC_COLORS = {
    "flip_rate": "#C45743",
    "acknowledgement_rate": "#6E9B76",
    "unacknowledged_flip_rate": "#D97A63",
    "changed_from_original_rate": "#6C8FB8",
}

EDIT_LABELS = {
    "cost_removed": "Cost removed",
    "severity_raised": "Severity raised",
    "probability_raised": "Probability raised",
    "obligation_removed": "Obligation removed",
    "evidence_made_report_shaped": "Report-framed",
    "finding_made_harder_to_rationalize": "Reduced mitigability",
}
METRIC_LABELS = {
    "flip_rate": "Changed to predicted decision (%)",
    "acknowledgement_rate": "Mentioned edited factor (%)",
    "unacknowledged_flip_rate": "Changed without mentioning edited factor (%)",
    "changed_from_original_rate": "Changed from original decision (%)",
    "unclear_rate": "Unclear parse rate (%)",
}
METRIC_CAPTIONS = {
    "flip_rate": "Numerator: parsed outputs whose new decision equals the pre-registered predicted counterfactual decision.",
    "acknowledgement_rate": "Numerator: parsed rationales that explicitly acknowledge the edited prompt factor under the selected scorer.",
    "unacknowledged_flip_rate": "Numerator: predicted-direction decision changes where the rationale did not mention the edited factor.",
    "changed_from_original_rate": "Numerator: parsed outputs whose decision differs from the original base-case decision.",
}
TITLE_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.8
BAR_LABEL_FONT_SIZE = 10


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _json_rows(path: Path) -> list[dict] | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = EXPERIMENT_DIR / resolved
    return resolved


def _save(fig, save_path: str | Path | None):
    path = _as_path(save_path)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=450, bbox_inches="tight")


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting functions. Install it in the notebook "
            "environment, or use the table/loading helpers without plotting."
        ) from exc
    return plt


def _ordered(values: Iterable[str], preferred: list[str]) -> list[str]:
    values = list(dict.fromkeys(v for v in values if pd.notna(v)))
    preferred_present = [v for v in preferred if v in values]
    remaining = sorted(v for v in values if v not in preferred)
    return preferred_present + remaining


def _pct_label(value: float) -> str:
    return f"{100 * value:.0f}%"


def _set_clean_axes(ax, ylabel: str = "", xlabel: str = ""):
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def _add_caption(fig, caption: str, y: float = 0.01):
    return None


def _legend_below(fig, handles, labels, ncol: int):
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=max(1, ncol),
    )


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z2 / (4 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _write_figure9_counts_csv(rows: list[dict], save_path: str | Path | None):
    path = _as_path(save_path)
    if path is None:
        path = GRAPHS_DIR / "flip_rate_by_edit_type_counts.csv"
    else:
        path = path.with_name(f"{path.stem}_counts.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["panel", "model", "category", "k", "n", "rate", "ci_low", "ci_high"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def load_selected_base_cases(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load selected base cases as a DataFrame."""
    return pd.DataFrame(_jsonl_rows(Path(data_dir) / "selected_base_cases.jsonl"))


def load_counterfactual_prompts(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load generated counterfactual prompt rows as a DataFrame."""
    return pd.DataFrame(_jsonl_rows(Path(data_dir) / "counterfactual_prompts.jsonl"))


def load_raw_outputs(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load raw counterfactual model outputs as a DataFrame."""
    return pd.DataFrame(_jsonl_rows(Path(data_dir) / "raw_outputs.jsonl"))


def load_parsed_outputs(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load parsed counterfactual outputs as a DataFrame."""
    return pd.DataFrame(_jsonl_rows(Path(data_dir) / "parsed_outputs.jsonl"))


def load_metrics(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load scored faithfulness metrics as a DataFrame."""
    df = pd.DataFrame(_jsonl_rows(Path(data_dir) / "faithfulness_metrics.jsonl"))
    bool_cols = [
        "changed_from_original",
        "flipped_predicted_direction",
        "acknowledged",
        "acknowledged_by_keyword",
        "acknowledged_by_llm",
        "unacknowledged_flip",
        "unacknowledged_flip_keyword",
        "unacknowledged_flip_llm",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    return df


def load_summary_by_model_and_edit(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load the JSON summary produced by score_rationale_faithfulness.py."""
    return pd.DataFrame(_json_rows(Path(data_dir) / "summary_by_model_and_edit.json"))


def load_summary_by_label(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load the model/label/edit JSON summary produced by the scoring script."""
    return pd.DataFrame(_json_rows(Path(data_dir) / "summary_by_label.json"))


def load_merged_results(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Merge prompt, parsed-output, and metric rows by counterfactual_id."""
    data_dir = Path(data_dir)
    prompts = load_counterfactual_prompts(data_dir)
    parsed = load_parsed_outputs(data_dir)
    metrics = load_metrics(data_dir)
    if prompts.empty:
        return metrics

    keep_parsed = [
        c for c in ["counterfactual_id", "raw_output", "parsed_reasoning", "error"]
        if c in parsed.columns
    ]
    merged = prompts.merge(parsed[keep_parsed], on="counterfactual_id", how="left")
    if not metrics.empty:
        metric_cols = [
            c for c in metrics.columns
            if c not in merged.columns or c == "counterfactual_id"
        ]
        merged = merged.merge(metrics[metric_cols], on="counterfactual_id", how="left")
    return merged


def _ack_columns(acknowledgement_source: str) -> tuple[str, str]:
    if acknowledgement_source == "keyword":
        return "acknowledged_by_keyword", "unacknowledged_flip_keyword"
    if acknowledgement_source == "llm":
        return "acknowledged_by_llm", "unacknowledged_flip_llm"
    if acknowledgement_source == "selected":
        return "acknowledged", "unacknowledged_flip"
    raise ValueError("acknowledgement_source must be 'keyword', 'llm', or 'selected'")


def _ack_source_label(acknowledgement_source: str) -> str:
    return {
        "keyword": "Rule-based",
        "llm": "gemini-3.6-flash",
        "selected": "Selected scorer",
    }[acknowledgement_source]


def summarize_metrics(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    acknowledgement_source: str = "keyword",
) -> pd.DataFrame:
    """Return one row per model/edit with core rates and counts."""
    if df is None:
        df = load_metrics(data_dir)
    ack_col, unack_col = _ack_columns(acknowledgement_source)
    if df.empty:
        return _empty_df([
            "model",
            "edit_type",
            "n_total",
            "n_parse_ok",
            "unclear_rate",
            "flip_rate",
            "acknowledgement_rate",
            "unacknowledged_flip_rate",
            "changed_from_original_rate",
        ])

    rows = []
    for (model, edit_type), group in df.groupby(["model", "edit_type"], dropna=False):
        ok = group[group["parse_status"].eq("ok")]
        if ack_col not in ok.columns:
            ack_rate = 0
        else:
            ack_rate = ok[ack_col].fillna(False).mean() if len(ok) else 0
        if unack_col not in ok.columns:
            unack_rate = 0
        else:
            unack_rate = ok[unack_col].fillna(False).mean() if len(ok) else 0
        rows.append({
            "model": model,
            "edit_type": edit_type,
            "n_total": len(group),
            "n_parse_ok": len(ok),
            "unclear_rate": 1 - len(ok) / len(group) if len(group) else 0,
            "flip_rate": ok["flipped_predicted_direction"].mean() if len(ok) else 0,
            "acknowledgement_rate": ack_rate,
            "unacknowledged_flip_rate": unack_rate,
            "changed_from_original_rate": ok["changed_from_original"].mean() if len(ok) else 0,
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=_ordered(out["model"], MODEL_ORDER), ordered=True)
    out["edit_type"] = pd.Categorical(out["edit_type"], categories=_ordered(out["edit_type"], EDIT_ORDER), ordered=True)
    return out.sort_values(["edit_type", "model"]).reset_index(drop=True)


def summarize_by_label(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    label_col: str = "original_monitor_label",
    acknowledgement_source: str = "keyword",
) -> pd.DataFrame:
    """Return one row per model/rationale-label/edit with core rates and counts."""
    if df is None:
        df = load_metrics(data_dir)
    ack_col, unack_col = _ack_columns(acknowledgement_source)
    if df.empty or label_col not in df.columns:
        return _empty_df([
            "model",
            label_col,
            "edit_type",
            "n_total",
            "n_parse_ok",
            "flip_rate",
            "acknowledgement_rate",
            "unacknowledged_flip_rate",
        ])

    rows = []
    for key, group in df.groupby(["model", label_col, "edit_type"], dropna=False):
        model, label, edit_type = key
        ok = group[group["parse_status"].eq("ok")]
        ack_rate = ok[ack_col].fillna(False).mean() if len(ok) and ack_col in ok.columns else 0
        unack_rate = ok[unack_col].fillna(False).mean() if len(ok) and unack_col in ok.columns else 0
        rows.append({
            "model": model,
            label_col: label if pd.notna(label) and label != "" else "unlabeled",
            "edit_type": edit_type,
            "n_total": len(group),
            "n_parse_ok": len(ok),
            "flip_rate": ok["flipped_predicted_direction"].mean() if len(ok) else 0,
            "acknowledgement_rate": ack_rate,
            "unacknowledged_flip_rate": unack_rate,
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=_ordered(out["model"], MODEL_ORDER), ordered=True)
    out["edit_type"] = pd.Categorical(out["edit_type"], categories=_ordered(out["edit_type"], EDIT_ORDER), ordered=True)
    return out.sort_values(["model", label_col, "edit_type"]).reset_index(drop=True)


def counterfactual_count_table(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Count counterfactual prompts by model, bucket, and edit type."""
    df = load_counterfactual_prompts(data_dir)
    if df.empty:
        return _empty_df(["model", "bucket", "edit_type", "n"])
    return (
        df.groupby(["model", "bucket", "edit_type"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["model", "bucket", "edit_type"])
        .reset_index(drop=True)
    )


def decision_shift_table(df: pd.DataFrame | None = None, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Count original-to-counterfactual decision transitions."""
    if df is None:
        df = load_metrics(data_dir)
    if df.empty:
        return _empty_df(["model", "edit_type", "original_decision", "counterfactual_decision", "n"])
    return (
        df.groupby(["model", "edit_type", "original_decision", "counterfactual_decision"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["model", "edit_type", "original_decision", "counterfactual_decision"])
        .reset_index(drop=True)
    )


def model_summary_table(df: pd.DataFrame | None = None, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Compact model-level faithfulness summary for notebook display."""
    if df is None:
        df = load_metrics(data_dir)
    if df.empty:
        return _empty_df([
            "model",
            "n_total",
            "n_parse_ok",
            "flip_rate",
            "acknowledgement_rate",
            "unacknowledged_flip_rate",
            "changed_from_original_rate",
        ])
    rows = []
    for model, group in df.groupby("model", dropna=False):
        ok = group[group["parse_status"].eq("ok")]
        rows.append({
            "model": model,
            "n_total": len(group),
            "n_parse_ok": len(ok),
            "flip_rate": ok["flipped_predicted_direction"].mean() if len(ok) else 0,
            "acknowledgement_rate_keyword": ok["acknowledged_by_keyword"].fillna(False).mean() if len(ok) and "acknowledged_by_keyword" in ok.columns else 0,
            "acknowledgement_rate_llm": ok["acknowledged_by_llm"].fillna(False).mean() if len(ok) and "acknowledged_by_llm" in ok.columns else 0,
            "unacknowledged_flip_rate_keyword": ok["unacknowledged_flip_keyword"].fillna(False).mean() if len(ok) and "unacknowledged_flip_keyword" in ok.columns else 0,
            "unacknowledged_flip_rate_llm": ok["unacknowledged_flip_llm"].fillna(False).mean() if len(ok) and "unacknowledged_flip_llm" in ok.columns else 0,
            "changed_from_original_rate": ok["changed_from_original"].mean() if len(ok) else 0,
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=_ordered(out["model"], MODEL_ORDER), ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def _plot_metric_by_edit(
    metric: str,
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    title: str | None = None,
    show_n: bool = True,
    figsize: tuple[float, float] = (12, 5.6),
    acknowledgement_source: str = "keyword",
):
    plt = _get_plt()
    summary = summarize_metrics(df, data_dir, acknowledgement_source=acknowledgement_source)
    if summary.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        _save(fig, save_path)
        return summary, fig, ax

    edits = _ordered(summary["edit_type"].astype(str), EDIT_ORDER)
    models = _ordered(summary["model"].astype(str), MODEL_ORDER)
    x = np.arange(len(edits)) * 1.14
    width = min(0.13, 0.62 / max(len(models), 1))
    group_spacing = width * 1.48

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    grouped_label_entries: list[list[dict]] = [[] for _ in edits]
    for i, model in enumerate(models):
        values = []
        ns = []
        for edit in edits:
            row = summary[(summary["model"].astype(str) == model) & (summary["edit_type"].astype(str) == edit)]
            values.append(float(row[metric].iloc[0]) if not row.empty else 0)
            ns.append(int(row["n_parse_ok"].iloc[0]) if not row.empty else 0)
        xpos = x + (i - (len(models) - 1) / 2) * group_spacing
        bars = ax.bar(xpos, values, width=width, label=_display_model(model), color=MODEL_COLORS.get(model), edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        if show_n:
            for edit_idx, (bar, value, n) in enumerate(zip(bars, values, ns)):
                grouped_label_entries[edit_idx].append({
                    "bar": bar,
                    "value": value,
                    "n": n,
                })

    if show_n:
        for entries in grouped_label_entries:
            crowded_full = sum(1 for entry in entries if entry["value"] >= 0.995) >= 2
            full_entries = [entry for entry in entries if entry["value"] >= 0.995]
            for idx, entry in enumerate(entries):
                bar = entry["bar"]
                value = entry["value"]
                if value <= 0:
                    continue
                x_pos = bar.get_x() + bar.get_width() / 2
                if crowded_full and entry in full_entries:
                    center_idx = (len(full_entries) - 1) / 2
                    full_idx = full_entries.index(entry)
                    x_pos += (full_idx - center_idx) * bar.get_width() * 0.45
                ax.text(
                    x_pos,
                    value + 0.025,
                    f"{_pct_label(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_FONT_SIZE,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([EDIT_LABELS.get(e, e) for e in edits], rotation=0, ha="center")
    ax.set_ylim(0, 1.08 if show_n else 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _set_clean_axes(ax, ylabel=METRIC_LABELS.get(metric, metric))
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(fig, handles, labels, len(models))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _add_caption(fig, METRIC_CAPTIONS.get(metric, ""))
    _save(fig, save_path)
    return summary, fig, ax


def _plot_metric_by_edit_source_comparison(
    metric: str,
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    title: str | None = None,
    show_n: bool = True,
    figsize: tuple[float, float] = (15.2, 6.8),
):
    plt = _get_plt()
    keyword = summarize_metrics(df, data_dir, acknowledgement_source="keyword")
    llm = summarize_metrics(df, data_dir, acknowledgement_source="llm")
    keyword["acknowledgement_source"] = _ack_source_label("keyword")
    llm["acknowledgement_source"] = _ack_source_label("llm")
    summary = pd.concat([keyword, llm], ignore_index=True)

    if summary.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        _save(fig, save_path)
        return summary, fig, ax

    edits = _ordered(summary["edit_type"].astype(str), EDIT_ORDER)
    models = _ordered(summary["model"].astype(str), MODEL_ORDER)
    rows = []
    for model in models:
        for edit in edits:
            sub = summary[
                (summary["edit_type"].astype(str) == edit)
                & (summary["model"].astype(str) == model)
            ]
            keyword_row = sub[sub["acknowledgement_source"].eq(_ack_source_label("keyword"))]
            llm_row = sub[sub["acknowledgement_source"].eq(_ack_source_label("llm"))]
            rows.append({
                "model": model,
                "edit": edit,
                "keyword_value": float(keyword_row[metric].iloc[0]) if not keyword_row.empty and pd.notna(keyword_row[metric].iloc[0]) else 0.0,
                "llm_value": float(llm_row[metric].iloc[0]) if not llm_row.empty and pd.notna(llm_row[metric].iloc[0]) else 0.0,
                "n_parse_ok": int(keyword_row["n_parse_ok"].iloc[0]) if not keyword_row.empty and pd.notna(keyword_row["n_parse_ok"].iloc[0]) else 0,
            })
    plot_df = pd.DataFrame(rows)

    x = np.arange(len(edits)) * 0.8
    width = min(0.9 / max(len(models), 1), 0.21)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    keyword_color = "#D97A63"
    llm_color = "#6C8FB8"
    label_gap_threshold = 0.05

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model)
        model_rows = plot_df[plot_df["model"].astype(str) == model].copy()
        model_rows["edit"] = pd.Categorical(model_rows["edit"], categories=edits, ordered=True)
        model_rows = model_rows.sort_values("edit")
        xpos = x + offsets[i]
        keyword_values = model_rows["keyword_value"].tolist()
        llm_values = model_rows["llm_value"].tolist()

        for xp, keyword_value, llm_value in zip(xpos, keyword_values, llm_values):
            ax.plot(
                [xp, xp],
                [keyword_value, llm_value],
                color="black",
                linewidth=3.2,
                alpha=1.0,
                zorder=1,
            )

        ax.scatter(
            xpos,
            keyword_values,
            s=116,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            xpos,
            llm_values,
            s=126,
            marker="D",
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

        if show_n:
            for xp, keyword_value, llm_value in zip(xpos, keyword_values, llm_values):
                low = min(keyword_value, llm_value)
                high = max(keyword_value, llm_value)
                gap = abs(high - low)
                if high <= 0:
                    continue
                if high >= 0.995 and low >= 0.995:
                    continue
                if gap < label_gap_threshold:
                    if high >= 0.995:
                        continue
                    ax.text(
                        xp,
                        min(high + 0.03, 1.03),
                        _pct_label(high),
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )
                    continue
                if gap < 0.035:
                    ax.text(
                        xp,
                        min(high + 0.03, 1.03),
                        _pct_label(high),
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )
                    if low > 0:
                        ax.text(
                            xp,
                            max(low - 0.036, 0.015),
                            _pct_label(low),
                            ha="center",
                            va="top",
                            fontsize=12,
                        )
                else:
                    upper_pad = 0.038 if gap < 0.08 else 0.028
                    lower_pad = 0.042 if gap < 0.08 else 0.034
                    ax.text(
                        xp,
                        min(high + upper_pad, 1.03),
                        _pct_label(high),
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )
                    if low > 0:
                        ax.text(
                            xp,
                            max(low - lower_pad, 0.015),
                            _pct_label(low),
                            ha="center",
                            va="top",
                            fontsize=12,
                        )

    if len(x) > 1:
        separators = (x[:-1] + x[1:]) / 2
        for xv in separators:
            ax.axvline(xv, color="#D1D5DB", linestyle=":", linewidth=1.0, alpha=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([EDIT_LABELS.get(e, e) for e in edits], rotation=0, ha="center", fontsize=12.5)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _set_clean_axes(ax, ylabel=METRIC_LABELS.get(metric, metric), xlabel="")
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.label.set_size(12.5)
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS.get(model), markeredgecolor="black", markeredgewidth=1.2, markersize=8.5, label=_display_model(model))
        for model in models
    ]
    scorer_handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4, markersize=8.5, linewidth=0, label=_ack_source_label("keyword")),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4, markersize=8.5, linewidth=0, label=_ack_source_label("llm")),
    ]
    handles = model_handles + scorer_handles
    labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=13,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=min(4, len(handles)),
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.19)
    _add_caption(fig, METRIC_CAPTIONS.get(metric, ""))
    _save(fig, save_path)
    return plot_df, fig, ax


def plot_flip_rate_by_edit_type(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    show_n: bool = True,
):
    """Combined figure: flip rate by edit type and by rationale label."""
    plt = _get_plt()
    summary_input = load_metrics(data_dir) if df is None else df.copy()
    summary = summarize_metrics(summary_input, data_dir, acknowledgement_source="keyword")
    label_summary = summarize_by_label(
        summary_input,
        data_dir,
        label_col="original_monitor_label",
        acknowledgement_source="keyword",
    )
    if summary.empty:
        fig, ax = plt.subplots(figsize=(12, 5.6), dpi=300)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        _save(fig, save_path)
        return summary, fig, ax

    grouped = pd.DataFrame()
    if not label_summary.empty:
        grouped = (
            label_summary.groupby(["model", "original_monitor_label"], dropna=False)
            .apply(lambda g: pd.Series({
                "n_parse_ok": g["n_parse_ok"].sum(),
                "flip_rate": np.average(g["flip_rate"], weights=g["n_parse_ok"].clip(lower=1)),
            }))
            .reset_index()
        )
        grouped = grouped[
            grouped["original_monitor_label"].notna()
            & (grouped["original_monitor_label"].astype(str).str.strip() != "")
            & (grouped["original_monitor_label"].astype(str).str.lower() != "unlabeled")
        ].copy()

    ok_df = summary_input[summary_input["parse_status"].eq("ok")].copy()
    edit_counts: dict[tuple[str, str], dict[str, float]] = {}
    for (model, edit_type), group in ok_df.groupby(["model", "edit_type"], dropna=False):
        k = int(group["flipped_predicted_direction"].fillna(False).sum())
        n = int(len(group))
        lo, hi = wilson_ci(k, n)
        edit_counts[(str(model), str(edit_type))] = {
            "k": k,
            "n": n,
            "rate": k / n if n else 0.0,
            "ci_low": lo,
            "ci_high": hi,
        }

    label_counts: dict[tuple[str, str], dict[str, float]] = {}
    if not ok_df.empty and "original_monitor_label" in ok_df.columns:
        label_ok = ok_df.copy()
        label_ok["original_monitor_label"] = label_ok["original_monitor_label"].fillna("").astype(str)
        label_ok = label_ok[
            label_ok["original_monitor_label"].str.strip().ne("")
            & label_ok["original_monitor_label"].str.lower().ne("unlabeled")
        ].copy()
        for (model, label), group in label_ok.groupby(["model", "original_monitor_label"], dropna=False):
            k = int(group["flipped_predicted_direction"].fillna(False).sum())
            n = int(len(group))
            lo, hi = wilson_ci(k, n)
            label_counts[(str(model), str(label))] = {
                "k": k,
                "n": n,
                "rate": k / n if n else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }

    edits = _ordered(summary["edit_type"].astype(str), EDIT_ORDER)
    models = _ordered(summary["model"].astype(str), MODEL_ORDER)
    print("model, numerator, denominator, rate, wilson_lower, wilson_upper")
    for model in models:
        cell = edit_counts.get((str(model), "severity_raised"))
        if cell is not None:
            print(model, cell["k"], cell["n"], cell["rate"], cell["ci_low"], cell["ci_high"], sep=", ")
    model_groups = [
        [model for model in models if model in {"gpt-5.5", "o3"}],
        [model for model in models if model in {"sonnet", "opus"}],
    ]
    x = np.arange(len(edits)) * 1.05
    width = 0.26
    group_spacing = 0.32
    edit_tick_labels = [
        EDIT_LABELS.get(e, e)
        .replace("Cost removed", "Cost\nremoved")
        .replace("Severity raised", "Severity\nraised")
        .replace("Probability raised", "Probability\nraised")
        .replace("Obligation removed", "Obligation\nremoved")
        .replace("Report-framed", "Report-\nframed")
        .replace("Reduced mitigability", "Reduced\nmitigability")
        for e in edits
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18.5, 9.5),
        dpi=300,
        sharex="col",
        sharey="col",
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )
    (ax1, ax2), (ax3, ax4) = axes

    x_tick_font_size = 12
    y_tick_font_size = 12
    y_label_font_size = 13
    title_font_size = TITLE_FONT_SIZE
    legend_font_size = LEGEND_FONT_SIZE
    value_font_size = 11

    labels = sorted(grouped["original_monitor_label"].astype(str).unique()) if not grouped.empty else []
    figure9_rows: list[dict] = []
    for model in models:
        for edit in edits:
            cell = edit_counts.get((str(model), str(edit)))
            if cell is None:
                figure9_rows.append({
                    "panel": "edit_type",
                    "model": str(model),
                    "category": str(edit),
                    "k": 0,
                    "n": 0,
                    "rate": 0.0,
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                })
            else:
                figure9_rows.append({
                    "panel": "edit_type",
                    "model": str(model),
                    "category": str(edit),
                    "k": int(cell["k"]),
                    "n": int(cell["n"]),
                    "rate": float(cell["rate"]),
                    "ci_low": float(cell["ci_low"]),
                    "ci_high": float(cell["ci_high"]),
                })
    for model in models:
        for label in labels:
            cell = label_counts.get((str(model), str(label)))
            if cell is None:
                continue
            figure9_rows.append({
                "panel": "rationale_label",
                "model": str(model),
                "category": str(label),
                "k": int(cell["k"]),
                "n": int(cell["n"]),
                "rate": float(cell["rate"]),
                "ci_low": float(cell["ci_low"]),
                "ci_high": float(cell["ci_high"]),
            })

    def _annotate_grouped_percentage_bars(ax, grouped_entries_by_bucket: list[list[dict]], y_pad: float, font_size: int):
        for entries in grouped_entries_by_bucket:
            label_entries = [entry for entry in entries if entry["value"] > 0]
            if not label_entries:
                continue
            label_entries.sort(key=lambda item: item["value"])
            placed_y: list[float] = []
            for entry in label_entries:
                bar = entry["bar"]
                value = entry["value"]
                x_pos = bar.get_x() + bar.get_width() / 2
                label = f"{_pct_label(value)}"
                y_pos = entry.get("label_y", value) + y_pad
                while any(abs(y_pos - prev_y) < 0.018 for prev_y in placed_y):
                    y_pos += 0.015
                placed_y.append(y_pos)
                ax.text(
                    x_pos,
                    y_pos,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=font_size,
                    color="black",
                )

    def _yerr_arrays(values: list[float], cis: list[tuple[float, float]]) -> list[list[float]]:
        lower = []
        upper = []
        for value, (lo, hi) in zip(values, cis):
            if np.isnan(lo) or np.isnan(hi):
                lower.append(np.nan)
                upper.append(np.nan)
            else:
                lower.append(max(0.0, value - lo))
                upper.append(max(0.0, min(1.0, hi) - value))
        return [lower, upper]

    def _draw_edit_panel(ax, panel_models: list[str], panel_title: str | None, show_xlabels: bool):
        grouped_label_entries: list[list[dict]] = [[] for _ in edits]
        for i, model in enumerate(panel_models):
            values = []
            cis = []
            for edit in edits:
                cell = edit_counts.get((str(model), str(edit)))
                if cell is None:
                    values.append(0.0)
                    cis.append((float("nan"), float("nan")))
                else:
                    values.append(float(cell["rate"]))
                    cis.append((float(cell["ci_low"]), float(cell["ci_high"])))
            xpos = x + (i - (len(panel_models) - 1) / 2) * group_spacing
            bars = ax.bar(
                xpos,
                values,
                width=width,
                yerr=_yerr_arrays(values, cis),
                capsize=3,
                error_kw={"linewidth": 0.8, "ecolor": "0.25"},
                label=_display_model(model),
                color=MODEL_COLORS.get(model),
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
            if show_n:
                for edit_idx, (bar, value, ci) in enumerate(zip(bars, values, cis)):
                    grouped_label_entries[edit_idx].append({
                        "bar": bar,
                        "value": value,
                        "label_y": value if any(np.isnan(v) for v in ci) else min(1.0, ci[1]),
                    })
        if show_n:
            _annotate_grouped_percentage_bars(ax, grouped_label_entries, y_pad=0.02, font_size=value_font_size)
        ax.set_xticks(x)
        if show_xlabels:
            ax.set_xticklabels(edit_tick_labels, rotation=0, ha="center", fontsize=x_tick_font_size)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0, 1.08 if show_n else 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        _set_clean_axes(ax, ylabel=METRIC_LABELS.get("flip_rate", "flip_rate"))
        ax.tick_params(axis="y", labelsize=y_tick_font_size)
        ax.yaxis.label.set_size(y_label_font_size)
        if panel_title:
            ax.set_title(panel_title, fontsize=title_font_size, pad=14)
        ax.legend(frameon=False, fontsize=legend_font_size, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(panel_models))

    if grouped.empty:
        for ax in (ax2, ax4):
            ax.text(0.5, 0.5, "No labeled summary available", ha="center", va="center")
            _set_clean_axes(ax, ylabel="")
    else:
        x2 = np.arange(len(labels)) * 1.0
        width2 = 0.26
        group_spacing2 = 0.32

        def _draw_label_panel(ax, panel_models: list[str], panel_title: str | None, show_xlabels: bool):
            grouped_entries: list[list[dict]] = [[] for _ in labels]
            for i, model in enumerate(panel_models):
                values = []
                cis = []
                for label in labels:
                    cell = label_counts.get((str(model), str(label)))
                    if cell is None:
                        values.append(0.0)
                        cis.append((float("nan"), float("nan")))
                    else:
                        values.append(float(cell["rate"]))
                        cis.append((float(cell["ci_low"]), float(cell["ci_high"])))
                xpos = x2 + (i - (len(panel_models) - 1) / 2) * group_spacing2
                bars = ax.bar(
                    xpos,
                    values,
                    width=width2,
                    yerr=_yerr_arrays(values, cis),
                    capsize=3,
                    error_kw={"linewidth": 0.8, "ecolor": "0.25"},
                    label=_display_model(model),
                    color=MODEL_COLORS.get(model),
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                )
                for label_idx, (bar, value, ci) in enumerate(zip(bars, values, cis)):
                    grouped_entries[label_idx].append({
                        "bar": bar,
                        "value": value,
                        "label_y": value if any(np.isnan(v) for v in ci) else min(1.0, ci[1]),
                    })
            _annotate_grouped_percentage_bars(ax, grouped_entries, y_pad=0.02, font_size=value_font_size)
            ax.set_xticks(x2)
            if show_xlabels:
                ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=x_tick_font_size)
            else:
                ax.set_xticklabels([])
            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            _set_clean_axes(ax, ylabel="")
            ax.tick_params(axis="y", labelsize=y_tick_font_size)
            ax.set_xlabel("")
            if panel_title:
                ax.set_title(panel_title, fontsize=title_font_size, pad=14)
            ax.legend(frameon=False, fontsize=legend_font_size, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(panel_models))

        _draw_label_panel(ax2, model_groups[0], None, False)
        _draw_label_panel(ax4, model_groups[1], None, True)

    _draw_edit_panel(ax1, model_groups[0], None, False)
    _draw_edit_panel(ax3, model_groups[1], None, True)

    ax1.text(-0.16, 0.5, "OpenAI", transform=ax1.transAxes, rotation=90, va="center", ha="center", fontsize=12)
    ax3.text(-0.16, 0.5, "Anthropic", transform=ax3.transAxes, rotation=90, va="center", ha="center", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13, hspace=0.42, wspace=0.14)
    _add_caption(
        fig,
        "Left column: flip rate by counterfactual edit. Right column: flip rate aggregated by original rationale label. Top row is OpenAI; bottom row is Anthropic.",
    )
    counts_path = _write_figure9_counts_csv(figure9_rows, save_path)
    print(f"Wrote Figure 9 counts to {counts_path}")
    _save(fig, save_path)
    return {"by_edit": summary, "by_label": grouped}, fig, axes


def plot_acknowledgement_rate_by_edit_type(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    show_n: bool = True,
    acknowledgement_source: str = "both",
):
    """Graph 3: fraction of parsed rationales that mention the edited factor."""
    if acknowledgement_source == "both":
        return _plot_metric_by_edit_source_comparison(
            "acknowledgement_rate",
            df=df,
            data_dir=data_dir,
            save_path=save_path,
            title="Rationales Mentioning the Edited Factor",
            show_n=show_n,
        )
    return _plot_metric_by_edit(
        "acknowledgement_rate",
        df=df,
        data_dir=data_dir,
        save_path=save_path,
        title="Rationales Mentioning the Edited Factor",
        show_n=show_n,
        acknowledgement_source=acknowledgement_source,
    )


def plot_unacknowledged_flip_rate(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    show_n: bool = True,
    acknowledgement_source: str = "both",
):
    """Graph 4: predicted-direction changes whose rationale did not mention the edit."""
    if acknowledgement_source == "both":
        return _plot_metric_by_edit_source_comparison(
            "unacknowledged_flip_rate",
            df=df,
            data_dir=data_dir,
            save_path=save_path,
            title="Changed to Predicted Decision Without Mentioning Edited Factor",
            show_n=show_n,
        )
    return _plot_metric_by_edit(
        "unacknowledged_flip_rate",
        df=df,
        data_dir=data_dir,
        save_path=save_path,
        title="Changed to Predicted Decision Without Mentioning Edited Factor",
        show_n=show_n,
        acknowledgement_source=acknowledgement_source,
    )


def plot_changed_from_original_rate_by_edit_type(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    show_n: bool = True,
):
    """Plot any decision change, regardless of whether it matches the predicted direction."""
    return _plot_metric_by_edit(
        "changed_from_original_rate",
        df=df,
        data_dir=data_dir,
        save_path=save_path,
        title="Any Decision Change by Counterfactual Edit",
        show_n=show_n,
    )


def plot_o3_obligation_faithfulness(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    include_all_models: bool = False,
    figsize: tuple[float, float] = (8.5, 5.2),
    acknowledgement_source: str = "both",
):
    """Graph 5: isolate obligation-sensitive cases, defaulting to o3 only."""
    plt = _get_plt()
    if df is None:
        df = load_metrics(data_dir)

    if acknowledgement_source == "both":
        summaries = []
        sub = df[df["edit_type"].eq("obligation_removed")]
        if not include_all_models:
            sub = sub[sub["model"].eq("o3")]
        for source_key, source_label in [("keyword", _ack_source_label("keyword")), ("llm", _ack_source_label("llm"))]:
            ack_col, unack_col = _ack_columns(source_key)
            for model, group in sub.groupby("model", dropna=False):
                ok = group[group["parse_status"].eq("ok")]
                for metric in ["flipped_predicted_direction", ack_col, unack_col]:
                    summaries.append({
                        "model": model,
                        "metric": metric,
                        "rate": ok[metric].fillna(False).mean() if len(ok) and metric in ok.columns else 0,
                        "n_parse_ok": len(ok),
                        "acknowledgement_source": source_label,
                    })
        summary = pd.DataFrame(summaries)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=300, sharey=True)
        metric_sources = [
            (_ack_source_label("keyword"), "acknowledged_by_keyword", "unacknowledged_flip_keyword"),
            (_ack_source_label("llm"), "acknowledged_by_llm", "unacknowledged_flip_llm"),
        ]
        for ax, (source, ack_col, unack_col) in zip(axes, metric_sources):
            source_summary = summary[summary["acknowledgement_source"].eq(source)]
            metric_order = ["flipped_predicted_direction", ack_col, unack_col]
            metric_names = {
                "flipped_predicted_direction": "Changed to\npredicted decision",
                ack_col: "Mentioned\nobligation edit",
                unack_col: "Changed without\nmentioning edit",
            }
            models = _ordered(source_summary["model"].astype(str), MODEL_ORDER)
            x = np.arange(len(metric_order))
            width = min(0.78 / max(len(models), 1), 0.34)
            for i, model in enumerate(models):
                values = []
                for metric in metric_order:
                    row = source_summary[
                        (source_summary["model"].astype(str) == model)
                        & (source_summary["metric"] == metric)
                    ]
                    values.append(float(row["rate"].iloc[0]) if not row.empty else 0)
                ax.bar(
                    x + (i - (len(models) - 1) / 2) * width,
                    values,
                    width=width,
                    label=_display_model(model),
                    color=MODEL_COLORS.get(model),
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([metric_names[m] for m in metric_order], rotation=0, ha="center")
            ax.set_ylim(0, 1.1)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            _set_clean_axes(ax, ylabel="Rate among parsed outputs" if ax is axes[0] else "")
        handles, labels = axes[0].get_legend_handles_labels()
        if include_all_models:
            _legend_below(fig, handles, labels, len(labels))
        fig.tight_layout()
        fig.subplots_adjust(top=0.86, bottom=0.16 if include_all_models else 0.13)
        _add_caption(fig, "Obligation edit: remove mandatory remediation/delay/cancellation consequences after reviewing evidence.")
        _save(fig, save_path)
        return summary, fig, axes

    if df.empty:
        summary = _empty_df(["model", "metric", "rate", "n_parse_ok"])
    else:
        sub = df[df["edit_type"].eq("obligation_removed")]
        if not include_all_models:
            sub = sub[sub["model"].eq("o3")]
        rows = []
        for model, group in sub.groupby("model", dropna=False):
            ok = group[group["parse_status"].eq("ok")]
            ack_col, unack_col = _ack_columns(acknowledgement_source)
            for metric in ["flipped_predicted_direction", ack_col, unack_col]:
                rows.append({
                    "model": model,
                    "metric": metric,
                    "rate": ok[metric].fillna(False).mean() if len(ok) and metric in ok.columns else 0,
                    "n_parse_ok": len(ok),
                })
        summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if summary.empty:
        ax.text(0.5, 0.5, "No obligation_removed cases available", ha="center", va="center")
        _save(fig, save_path)
        return summary, fig, ax

    ack_col, unack_col = _ack_columns(acknowledgement_source)
    metric_order = ["flipped_predicted_direction", ack_col, unack_col]
    metric_names = {
        "flipped_predicted_direction": "Changed to\npredicted decision",
        ack_col: f"Mentioned\nobligation edit\n({_ack_source_label(acknowledgement_source)})",
        unack_col: f"Changed without\nmentioning edit\n({_ack_source_label(acknowledgement_source)})",
    }
    models = _ordered(summary["model"].astype(str), MODEL_ORDER)
    x = np.arange(len(metric_order))
    width = min(0.78 / max(len(models), 1), 0.34)
    for i, model in enumerate(models):
        values = []
        ns = []
        for metric in metric_order:
            row = summary[(summary["model"].astype(str) == model) & (summary["metric"] == metric)]
            values.append(float(row["rate"].iloc[0]) if not row.empty else 0)
            ns.append(int(row["n_parse_ok"].iloc[0]) if not row.empty else 0)
        xpos = x + (i - (len(models) - 1) / 2) * width
        bars = ax.bar(xpos, values, width=width, label=_display_model(model), color=MODEL_COLORS.get(model), edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        for bar, value, n in zip(bars, values, ns):
            if value <= 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{_pct_label(value)}\nn={n}", ha="center", fontsize=BAR_LABEL_FONT_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_names[m] for m in metric_order], rotation=0, ha="center")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _set_clean_axes(ax, ylabel="Rate among parsed outputs")
    if include_all_models:
        handles, labels = ax.get_legend_handles_labels()
        _legend_below(fig, handles, labels, len(models))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18 if include_all_models else 0.18)
    _add_caption(fig, "Obligation edit: remove mandatory remediation/delay/cancellation consequences after reviewing evidence.")
    _save(fig, save_path)
    return summary, fig, ax


def plot_parse_status_by_model(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.5, 4.8),
):
    """Stacked count bars showing parse status by model."""
    plt = _get_plt()
    if df is None:
        df = load_metrics(data_dir)
    if df.empty:
        table = _empty_df(["model", "parse_status", "n"])
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        _save(fig, save_path)
        return table, fig, ax

    table = df.groupby(["model", "parse_status"], dropna=False).size().reset_index(name="n")
    pivot = table.pivot(index="model", columns="parse_status", values="n").fillna(0)
    pivot = pivot.reindex(_ordered(pivot.index.astype(str), MODEL_ORDER))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    bottom = np.zeros(len(pivot))
    colors = {"ok": "#2f7f66", "unclear": "#d8902f", "error": "#b7352d"}
    for status in sorted(pivot.columns):
        values = pivot[status].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=status, color=colors.get(status), edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        bottom += values
    _set_clean_axes(ax, ylabel="Counterfactual outputs")
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(fig, handles, labels, len(labels))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    _save(fig, save_path)
    return table, fig, ax


def plot_counterfactual_counts_by_edit(
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 5.2),
):
    """Count generated counterfactuals by edit type and model."""
    plt = _get_plt()
    df = load_counterfactual_prompts(data_dir)
    if df.empty:
        table = _empty_df(["model", "edit_type", "n"])
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.text(0.5, 0.5, "No counterfactual prompts available", ha="center", va="center")
        _save(fig, save_path)
        return table, fig, ax

    table = df.groupby(["model", "edit_type"], dropna=False).size().reset_index(name="n")
    edits = _ordered(table["edit_type"].astype(str), EDIT_ORDER)
    models = _ordered(table["model"].astype(str), MODEL_ORDER)
    x = np.arange(len(edits))
    width = min(0.78 / max(len(models), 1), 0.34)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for i, model in enumerate(models):
        values = []
        for edit in edits:
            row = table[(table["model"].astype(str) == model) & (table["edit_type"].astype(str) == edit)]
            values.append(int(row["n"].iloc[0]) if not row.empty else 0)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width=width, label=_display_model(model), color=MODEL_COLORS.get(model), edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", fontsize=BAR_LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([EDIT_LABELS.get(e, e) for e in edits], rotation=0, ha="center")
    _set_clean_axes(ax, ylabel="Counterfactual prompts")
    handles, labels = ax.get_legend_handles_labels()
    _legend_below(fig, handles, labels, len(models))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save(fig, save_path)
    return table, fig, ax


def plot_decision_shift_heatmap(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    model: str | None = None,
    edit_type: str | None = None,
    figsize: tuple[float, float] = (7, 5),
):
    """Heatmap of original decision to counterfactual decision transitions."""
    plt = _get_plt()
    if df is None:
        df = load_metrics(data_dir)
    if model is not None:
        df = df[df["model"].eq(model)]
    if edit_type is not None:
        df = df[df["edit_type"].eq(edit_type)]

    if df.empty:
        table = _empty_df(["original_decision", "counterfactual_decision", "n"])
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        _save(fig, save_path)
        return table, fig, ax

    table = (
        df.groupby(["original_decision", "counterfactual_decision"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    decisions = _ordered(
        set(table["original_decision"].astype(str)) | set(table["counterfactual_decision"].astype(str)),
        INITIAL_DECISION_ORDER + POST_DECISION_ORDER,
    )
    pivot = table.pivot(index="original_decision", columns="counterfactual_decision", values="n").fillna(0)
    pivot = pivot.reindex(index=[d for d in decisions if d in pivot.index], columns=[d for d in decisions if d in pivot.columns], fill_value=0)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha="center")
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = int(pivot.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > pivot.to_numpy().max() / 2 else "black")
    ax.set_xlabel("Counterfactual decision")
    ax.set_ylabel("Original decision")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    _save(fig, save_path)
    return table, fig, ax


def plot_metric_dashboard(
    df: pd.DataFrame | None = None,
    data_dir: str | Path = DATA_DIR,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (13, 8.5),
    acknowledgement_source: str = "keyword",
):
    """Compact 2x2 dashboard for the core metrics by edit type."""
    plt = _get_plt()
    summary = summarize_metrics(df, data_dir, acknowledgement_source=acknowledgement_source)
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=300, sharex=True, sharey=True)
    metrics = [
        "flip_rate",
        "acknowledgement_rate",
        "unacknowledged_flip_rate",
        "changed_from_original_rate",
    ]
    if summary.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        _save(fig, save_path)
        return summary, fig, axes

    edits = _ordered(summary["edit_type"].astype(str), EDIT_ORDER)
    models = _ordered(summary["model"].astype(str), MODEL_ORDER)
    x = np.arange(len(edits))
    width = min(0.78 / max(len(models), 1), 0.34)

    for ax, metric in zip(axes.flat, metrics):
        for i, model in enumerate(models):
            values = []
            for edit in edits:
                row = summary[(summary["model"].astype(str) == model) & (summary["edit_type"].astype(str) == edit)]
                values.append(float(row[metric].iloc[0]) if not row.empty else 0)
            ax.bar(
                x + (i - (len(models) - 1) / 2) * width,
                values,
                width=width,
                label=_display_model(model),
                color=MODEL_COLORS.get(model),
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
            )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        _set_clean_axes(ax, ylabel="Rate among parsed outputs")
        ax.set_xticks(x)
        ax.set_xticklabels([EDIT_LABELS.get(e, e) for e in edits], rotation=0, ha="center", fontsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, len(labels))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.16)
    _save(fig, save_path)
    return summary, fig, axes


def make_all_graphs(data_dir: str | Path = DATA_DIR, graphs_dir: str | Path = GRAPHS_DIR) -> dict[str, pd.DataFrame]:
    """Generate the standard graph set and return the underlying tables."""
    graphs_dir = Path(graphs_dir)
    metrics = load_metrics(data_dir)
    outputs = {}
    outputs["flip_rate_by_edit_type"], _, _ = plot_flip_rate_by_edit_type(
        metrics,
        save_path=graphs_dir / "flip_rate_by_edit_type.png",
    )
    outputs["acknowledgement_rate_by_edit_type"], _, _ = plot_acknowledgement_rate_by_edit_type(
        metrics,
        save_path=graphs_dir / "acknowledgement_rate_by_edit_type.png",
        acknowledgement_source="both",
    )
    outputs["unacknowledged_flip_rate"], _, _ = plot_unacknowledged_flip_rate(
        metrics,
        save_path=graphs_dir / "unacknowledged_flip_rate.png",
        acknowledgement_source="both",
    )
    outputs["o3_obligation_faithfulness"], _, _ = plot_o3_obligation_faithfulness(
        metrics,
        save_path=graphs_dir / "o3_obligation_faithfulness.png",
        acknowledgement_source="both",
    )
    outputs["parse_status_by_model"], _, _ = plot_parse_status_by_model(
        metrics,
        save_path=graphs_dir / "parse_status_by_model.png",
    )
    outputs["counterfactual_counts_by_edit"], _, _ = plot_counterfactual_counts_by_edit(
        data_dir,
        save_path=graphs_dir / "counterfactual_counts_by_edit.png",
    )
    outputs["metric_dashboard"], _, _ = plot_metric_dashboard(
        metrics,
        save_path=graphs_dir / "metric_dashboard_keyword.png",
        acknowledgement_source="keyword",
    )
    outputs["metric_dashboard_llm"], _, _ = plot_metric_dashboard(
        metrics,
        save_path=graphs_dir / "metric_dashboard_llm.png",
        acknowledgement_source="llm",
    )
    return outputs
