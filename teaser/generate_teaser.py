from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "benchmark_infographic_vector_updated.svg"
OUT_SVG = ROOT / "benchmark_infographic_final_vector.svg"
OUT_PDF = ROOT / "benchmark_infographic_final_vector.pdf"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

if not SRC.exists():
    raise FileNotFoundError(f"Missing input SVG: {SRC}")

tree = ET.parse(SRC)
root = tree.getroot()


def iter_with_parents(node):
    for parent in node.iter():
        for child in list(parent):
            yield parent, child

# ---------------------------------------------------------------
# Rebuild the entire "Manipulated variables" block in Panel A.
# The rest of the full figure remains vectorized and unchanged.
# ---------------------------------------------------------------

# Remove the old variable-name text in the lower-left panel.
old_text = {
    "Retrieval cost",
    "Risk (probability &",
    "severity)",
    "Evidence channel",
    "Knowing-created obligation",
}

for parent, el in list(iter_with_parents(root)):
    if el.tag.endswith("text") and (el.text or "") in old_text:
        parent.remove(el)

# Remove any old lower-left row separators, if present.
for parent, el in list(iter_with_parents(root)):
    if not el.tag.endswith("line"):
        continue
    try:
        x1 = float(el.get("x1", "0"))
        x2 = float(el.get("x2", "0"))
        y1 = float(el.get("y1", "0"))
        y2 = float(el.get("y2", "0"))
    except ValueError:
        continue
    if (
        abs(y1 - y2) < 0.1
        and 600 <= y1 <= 845
        and x1 < 520
        and x2 < 520
    ):
        parent.remove(el)

def add_text(x, y, text, size, weight="400", fill="#1E1E1E", anchor="start"):
    el = ET.Element(
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": str(y),
            "font-family": "DejaVu Sans",
            "font-size": str(size),
            "font-weight": weight,
            "text-anchor": anchor,
            "fill": fill,
        },
    )
    el.text = text
    root.append(el)
    return el

def add_line(x1, y1, x2, y2, stroke="#D8D3CB", width=1, dash="4,4"):
    attrs = {
        "x1": str(x1),
        "y1": str(y1),
        "x2": str(x2),
        "y2": str(y2),
        "stroke": stroke,
        "stroke-width": str(width),
    }
    if dash:
        attrs["stroke-dasharray"] = dash
    el = ET.Element(f"{{{SVG_NS}}}line", attrs)
    root.append(el)
    return el

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def add_rect(x, y, width, height, fill, stroke="none", stroke_width=1, rx=0, ry=0):
    el = ET.Element(
        f"{{{SVG_NS}}}rect",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": fill,
            "stroke": stroke,
            "stroke-width": str(stroke_width),
            "rx": str(rx),
            "ry": str(ry),
        },
    )
    root.append(el)
    return el

def _anchor_xy(el):
    for x_key, y_key in (("x", "y"), ("x1", "y1")):
        x_raw = el.get(x_key)
        y_raw = el.get(y_key)
        if x_raw is None or y_raw is None:
            continue
        try:
            return float(x_raw), float(y_raw)
        except ValueError:
            continue
    return None

# Layout tuned to match the approved reference:
# swatch | bold variable name | larger two-line explanation
name_x = 76
desc_x = 270
row_centers = [638, 696, 754, 812]

rows = [
    {
        "name": ["Retrieval cost"],
        "desc": ["Cost to obtain the report"],
    },
    {
        "name": ["Probability &", "severity"],
        "desc": ["Likelihood the report contains", "issues and their potential impact"],
    },
    {
        "name": ["Evidence channel"],
        "desc": ["Named report vs. diffuse", "background evidence"],
    },
    {
        "name": ["Knowing-created obligation"],
        "desc": ["Whether knowing about the report", "creates a duty to act"],
    },
]

# Move existing color swatches to align with row centers.
swatch_colors = ["#4D98D7", "#F59C1A", "#9682C6", "#5E8656"]
swatches = []
for el in root.iter():
    if not el.tag.endswith("rect"):
        continue
    if el.get("fill") in swatch_colors:
        try:
            x = float(el.get("x", "0"))
            y = float(el.get("y", "0"))
            w = float(el.get("width", "0"))
            h = float(el.get("height", "0"))
        except ValueError:
            continue
        if x < 100 and y > 590 and 20 <= w <= 30 and 20 <= h <= 35:
            swatches.append(el)

swatches.sort(key=lambda e: float(e.get("y")))

for el, cy in zip(swatches, row_centers):
    el.set("x", "34")
    el.set("y", str(cy - 14))
    el.set("width", "24")
    el.set("height", "28")
    el.set("rx", "2")
    el.set("ry", "2")

# Clean dashed separators centered between rows.
for y in [667, 725, 783]:
    add_line(23, y, 487, y)

# Add new larger text, with deliberate column spacing so nothing overlaps.
for row, cy in zip(rows, row_centers):
    names = row["name"]
    desc = row["desc"]

    if len(names) == 1:
        # The long final label gets a slightly smaller size but remains larger
        # than before and stays on one line.
        fs = 13.6 if names[0] == "Knowing-created obligation" else 14.6
        add_text(name_x, cy + 5, names[0], fs, "700")
    else:
        add_text(name_x, cy - 3, names[0], 13.8, "700")
        add_text(name_x, cy + 13, names[1], 13.8, "700")

    desc_fs = 13.2
    if len(desc) == 1:
        add_text(desc_x, cy + 5, desc[0], desc_fs, "400", "#3E3E3E")
    else:
        add_text(desc_x, cy - 3, desc[0], desc_fs, "400", "#3E3E3E")
        add_text(desc_x, cy + 12, desc[1], desc_fs, "400", "#3E3E3E")

# ---------------------------------------------------------------
# Panel B: keep Opus 4.8 100.0 label slightly higher for clarity.
# ---------------------------------------------------------------
for el in root.iter():
    if not el.tag.endswith("text"):
        continue
    if (el.text or "") == "100.0":
        try:
            x = float(el.get("x", "0"))
        except ValueError:
            continue
        # The Opus 4.8 green bar label in Panel B.
        if 900 <= x <= 970:
            y = float(el.get("y", "0"))
            if y > 180:
                el.set("y", str(y - 8))
                break

# Panel B: replace stale GPT-5.5 labels and caption with the current values.
panel_b_text_updates = {
    (636.5, 332.1104761904762): "68.4",
    (666.5, 309.4190476190476): "73.3",
    (696.5, 339.2761904761905): "66.3",
}
for el in root.iter():
    if not el.tag.endswith("text"):
        continue
    x = _safe_float(el.get("x"))
    y = _safe_float(el.get("y"))
    if x is None or y is None:
        continue
    key = (x, y)
    if key in panel_b_text_updates:
        el.text = panel_b_text_updates[key]
    elif (el.text or "") == "Report-shaped evidence is inspected more often":
        el.text = "Inspection is highest when a named report"
    elif (el.text or "") == "than diffuse background evidence.":
        el.text = "must be retrieved, lowest for diffuse background evidence."

# ---------------------------------------------------------------
# Panel C: rebuild from the current 6-condition decomposition data.
# ---------------------------------------------------------------
PANEL_C_X_MIN = 1160
PANEL_C_X_MAX = 1635
PANEL_C_Y_MIN = 80
PANEL_C_Y_MAX = 825

for parent, el in list(iter_with_parents(root)):
    pos = _anchor_xy(el)
    if pos is None:
        continue
    x, y = pos
    if not (PANEL_C_X_MIN <= x <= PANEL_C_X_MAX and PANEL_C_Y_MIN <= y <= PANEL_C_Y_MAX):
        continue
    tag = el.tag.split("}")[-1]
    if tag not in {"text", "rect", "line", "polygon", "path"}:
        continue
    parent.remove(el)

condition_order = [
    "free_no_obligation",
    "cost_only",
    "obligation_only",
    "cost_plus_obligation",
    "obligation_only_no_cancel",
    "cost_plus_obligation_no_cancel",
]
condition_colors = {
    "free_no_obligation": "#4D98D7",
    "cost_only": "#F59C1A",
    "obligation_only": "#6FBC69",
    "cost_plus_obligation": "#E13D35",
    "obligation_only_no_cancel": "#9682C6",
    "cost_plus_obligation_no_cancel": "#B67A2A",
}
condition_labels = {
    "free_no_obligation": ["Free /", "no obligation"],
    "cost_only": ["Cost", "only"],
    "obligation_only": ["Obligation", "+ cancel"],
    "cost_plus_obligation": ["Cost +", "obligation"],
    "obligation_only_no_cancel": ["Obligation only", "(no cancel)"],
    "cost_plus_obligation_no_cancel": ["Cost + obligation", "(no cancel)"],
}
model_rows = [
    ("gpt-5.5", "GPT-5.5", 235),
    ("o3", "o3", 355),
    ("opus", "Opus 4.8", 487),
    ("sonnet", "Sonnet 4.6", 628),
]
results_dir = ROOT / "ablation" / "cost_obligation_decomposition" / "results"

panel_c_rates = {}
for model, _, _ in model_rows:
    path = results_dir / f"{model}_cost_obligation_decomposition.json"
    rows = json.loads(path.read_text())
    by_condition = {}
    for condition in condition_order:
        valid = [
            row for row in rows
            if row.get("condition") == condition and row.get("decision") in {"INSPECT", "SKIP"}
        ]
        inspect_rate = (
            sum(row.get("decision") == "INSPECT" for row in valid) / len(valid) * 100
            if valid else 0.0
        )
        by_condition[condition] = inspect_rate
    panel_c_rates[model] = by_condition

# Legend
legend_xs = [1172, 1326, 1476]
legend_y_rows = [112, 142]
legend_layout = [
    ("free_no_obligation", legend_xs[0], legend_y_rows[0]),
    ("cost_only", legend_xs[1], legend_y_rows[0]),
    ("obligation_only", legend_xs[2], legend_y_rows[0]),
    ("cost_plus_obligation", legend_xs[0], legend_y_rows[1]),
    ("obligation_only_no_cancel", legend_xs[1], legend_y_rows[1]),
    ("cost_plus_obligation_no_cancel", legend_xs[2], legend_y_rows[1]),
]
for condition, x, y in legend_layout:
    add_rect(x, y, 14, 18, condition_colors[condition])
    lines = condition_labels[condition]
    add_text(x + 24, y + 9, lines[0], 11.4, "400", "#4A4A4A")
    add_text(x + 24, y + 22, lines[1], 11.4, "400", "#4A4A4A")

# Small multiple bar rows
axis_x1 = 1294
axis_x2 = 1608
bar_width = 26
bar_gap = 24
bar_xs = [1310 + i * (bar_width + bar_gap) for i in range(len(condition_order))]
for model, display, base_y in model_rows:
    top_y = base_y - 30
    bottom_y = base_y + 52
    add_text(1164, base_y + 8, display, 19, "700")
    add_line(axis_x1, top_y, axis_x1, bottom_y, stroke="#444444", width=1.2, dash=None)
    add_line(axis_x1, bottom_y, axis_x2, bottom_y, stroke="#444444", width=1.2, dash=None)
    add_text(1282, top_y + 5, "100", 13, "400", "#1E1E1E", anchor="end")
    add_text(1282, bottom_y + 5, "0", 13, "400", "#1E1E1E", anchor="end")

    for condition, x in zip(condition_order, bar_xs):
        value = panel_c_rates[model][condition]
        bar_height = 82 * value / 100.0
        bar_y = bottom_y - bar_height
        add_rect(x, bar_y, bar_width, bar_height, condition_colors[condition])
        label = f"{round(value):.0f}"
        label_x = x + bar_width / 2
        if condition == "free_no_obligation" and value >= 99.95 and model not in {"gpt-5.5", "o3"}:
            label_x += 7
        add_text(label_x, bar_y - 8, label, 13, "400", "#1E1E1E", anchor="middle")

for separator_y in [302, 430, 570]:
    add_line(1162, separator_y, 1630, separator_y, stroke="#D9D4CB", width=1, dash="4,4")

# Footer note
add_rect(1164, 706, 468, 108, "#FFF9EF", stroke="#E6A93A", stroke_width=1.3, rx=16, ry=16)
add_text(1398, 751, "Cost dominates; obligation deters only", 17.5, "400", "#1E1E1E", anchor="middle")
add_text(1398, 774, "when it threatens delay or cancellation.", 17.5, "400", "#1E1E1E", anchor="middle")

tree.write(OUT_SVG, encoding="utf-8", xml_declaration=True)

try:
    import cairosvg
except ImportError as exc:
    raise RuntimeError(
        "CairoSVG is required to export PDF. Install the Python package and the native Cairo library."
    ) from exc

# CairoSVG preserves the SVG geometry as vector graphics in the PDF.
cairosvg.svg2pdf(
    url=str(OUT_SVG),
    write_to=str(OUT_PDF),
    output_width=1664,
    output_height=886,
)

print(f"Saved SVG: {OUT_SVG}")
print(f"Saved PDF: {OUT_PDF}")
