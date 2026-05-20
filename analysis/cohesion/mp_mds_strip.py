"""
1D MDS strip plot: one dot per MP, x = MDS position derived from pairwise kappas,
y = party band (beeswarm-style jitter to avoid overplotting).
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PERIOD     = sys.argv[1] if len(sys.argv) > 1 else "bundestag_2021_2025"
LIGHT_MODE = len(sys.argv) > 2 and sys.argv[2] == "light"
_theme  = "light" if LIGHT_MODE else "dark"

# ── Theme ─────────────────────────────────────────────────────────────────────
if LIGHT_MODE:
    T = dict(
        bg              = "#ffffff",
        text            = "#1a1a1a",
        subtext         = "#555555",
        axis_text       = "#666666",
        zero_line       = "#333333",
        zero_alpha      = 0.35,
        grid_alpha      = 0.10,
        band_color      = "#aaaaaa",   # neutral grey for off-white bands
        band_alpha_even = 0.10,
        ann_bg          = "#ffffff",
        ann_text        = "#1a1a1a",
        ann_detail      = "#555555",
        ann_line        = "#333333",
        ann_line_alpha  = 0.50,
    )
    # CDU/CSU default is near-white; FDP yellow too pale on white bg
    LIGHT_PARTY_OVERRIDES = {"CDU/CSU": "#3a3a3a", "FDP": "#f0c000"}
else:
    T = dict(
        bg              = "#0d1117",
        text            = "white",
        subtext         = "#888888",
        axis_text       = "#888888",
        zero_line       = "white",
        zero_alpha      = 0.22,
        grid_alpha      = 0.06,
        band_color      = "white",
        band_alpha_even = 0.03,
        ann_bg          = "#0d1117",
        ann_text        = "white",
        ann_detail      = "#aaaaaa",
        ann_line        = "white",
        ann_line_alpha  = 0.38,
    )
    LIGHT_PARTY_OVERRIDES = {}
BASE_DIR  = Path(__file__).parent.parent
OUTPUT    = BASE_DIR / "output"
IMG_DIR   = OUTPUT / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR  = OUTPUT / PERIOD
COLORS_F  = BASE_DIR / "config" / "party_colours.json"

PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "SPD", "BÜNDNIS 90/DIE GRÜNEN",
               "Die Linke", "BSW"]

# Notable MPs to annotate per period: (name, label, nudge_x, nudge_y)
# nudge_x/y shift the text label relative to the dot
HIGHLIGHTS = {
    "bundestag_2005_2009": [
        ("Wolfgang Wodarg",               "Wolfgang Wodarg\n(SPD rebel)",        -0.04,  0.65),
        ("Peter Gauweiler",               "Peter Gauweiler\n(CSU rebel)",         0.04,  0.65),
        ("Joseph Fischer",                "Joschka Fischer\n(closest opp.)",      0.0,   0.65),
    ],
    "bundestag_2009_2013": [
        ("Angela Merkel",                 "Angela Merkel\n(procedural abstainer)", 0.04, 0.65),
        ("Peter Gauweiler",               "Peter Gauweiler\n(serial rebel)",      0.04,  0.65),
        ("Angelica Schwall-Düren",        "Angelica Schwall-Düren\n(SPD, closest opp.)", 0.0, 0.65),
    ],
    "bundestag_2013_2017": [
        ("Sigmar Gabriel",                "Sigmar Gabriel\n(SPD rebel)",          0.04,  0.65),
        ("Annette Schavan",               "Annette Schavan\n(CDU, resigned)",    -0.04,  0.65),
        ("Tom Koenigs",                   "Tom Koenigs\n(Grüne, closest)",        0.0,   0.65),
    ],
    "bundestag_2017_2021": [
        ("Marie-Agnes Strack-Zimmermann", "Marie-Agnes Strack-Zimmermann\n(FDP, cross-aisle)", 0.04, 0.65),
        ("Andrea Nahles",                 "Andrea Nahles\n(SPD leader)",          0.04,  0.65),
        ("Ralf Brauksiepe",               "Ralf Brauksiepe\n(CDU rebel)",        -0.04,  0.65),
    ],
    "bundestag_2021_2025": [
        ("Heiko Maas",                    "Heiko Maas\n(SPD rebel)",             -0.04,  0.65),
        ("Melis Sekmen",                  "Melis Sekmen\n(CDU rebel)",             0.04,  0.65),
        ("Thomas Sattelberger",           "Thomas Sattelberger\n(FDP rebel)",     0.04,  0.65),
    ],
    "bundestag_2025_2029": [
        ("Annalena Baerbock",             "Annalena Baerbock\n(Grüne, closest)",  0.04,  0.65),
        ("Michael Frieser",               "Michael Frieser\n(CDU rebel)",        -0.04,  0.65),
        ("Hubertus Heil",                 "Hubertus Heil\n(SPD rebel)",           0.04,  0.65),
    ],
}

# ── Load ──────────────────────────────────────────────────────────────────────
nodes = pd.read_csv(DATA_DIR / "nodes.csv")
allpairs = DATA_DIR / "edges_allpairs.csv"
edges = pd.read_csv(allpairs if allpairs.exists() else DATA_DIR / "edges.csv")
print(f"Using {'edges_allpairs.csv' if allpairs.exists() else 'edges.csv'}")

with open(COLORS_F) as f:
    raw_colors = json.load(f)

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p):
    return ALIASES.get(p, p)

party_color = {}
for k, v in raw_colors.items():
    party_color[canon(k)] = v
# fallback
party_color.setdefault("fraktionslos", "#888888")
party_color.setdefault("BSW", "#a020f0")

nodes["party"] = nodes["party"].map(canon).fillna(nodes["party"])
party_color.update(LIGHT_PARTY_OVERRIDES)

# ── Build distance matrix ─────────────────────────────────────────────────────
node_ids = nodes["person_id"].tolist()
n        = len(node_ids)
id2idx   = {nid: i for i, nid in enumerate(node_ids)}

# default: kappa=0 → distance=1
D = np.ones((n, n), dtype=np.float32)
np.fill_diagonal(D, 0.0)

src_idx = edges["source"].map(id2idx)
tgt_idx = edges["target"].map(id2idx)
mask    = src_idx.notna() & tgt_idx.notna()
si      = src_idx[mask].astype(int).values
ti      = tgt_idx[mask].astype(int).values
w       = edges.loc[mask, "weight"].values.astype(np.float32)

D[si, ti] = 1.0 - w
D[ti, si] = 1.0 - w

print(f"MPs: {n}  edges loaded: {mask.sum()}")

# ── Classical 1D MDS ──────────────────────────────────────────────────────────
D2       = D.astype(np.float64) ** 2
row_mean = D2.mean(axis=1, keepdims=True)
col_mean = D2.mean(axis=0, keepdims=True)
B        = -0.5 * (D2 - row_mean - col_mean + D2.mean())

print("Running eigendecomposition…")
eigvals, eigvecs = np.linalg.eigh(B)
top              = np.argmax(eigvals)
pos1d            = eigvecs[:, top] * np.sqrt(max(eigvals[top], 0.0))

nodes["x"] = pos1d

# Orient consistently: coalition parties should have positive (right) mean.
# This gives "opposition ← → coalition" semantics across all periods.
with open(BASE_DIR / "config" / "coalitions.json") as f:
    coalitions_map = json.load(f)

# Match period to coalition key (e.g. "bundestag_2017_2021" → "Bundestag 2017 - 2021")
period_key = PERIOD.replace("bundestag_", "Bundestag ").replace("_", " - ", 1).replace("_", " - ")
# try both dash variants
coalition_parties = set()
for k, v in coalitions_map.items():
    if k.replace(" ", "").replace("-", "") == period_key.replace(" ", "").replace("-", ""):
        coalition_parties = set(v)
        break

if coalition_parties:
    coalition_mean = nodes[nodes["party"].isin(coalition_parties)]["x"].mean()
    if coalition_mean < 0:
        nodes["x"] = -nodes["x"]
    print(f"Coalition parties: {coalition_parties}  mean_x oriented to right")
else:
    # Fallback: CDU/CSU always present, use it as anchor (put on right)
    cdu_mean = nodes[nodes["party"] == "CDU/CSU"]["x"].mean()
    if cdu_mean < 0:
        nodes["x"] = -nodes["x"]
    print("Fallback orientation: CDU/CSU on right")

# Standardise to [-1, 1]
xmin, xmax = nodes["x"].min(), nodes["x"].max()
nodes["x"] = 2 * (nodes["x"] - xmin) / (xmax - xmin) - 1

# ── Beeswarm-style y jitter per party ─────────────────────────────────────────
# Sort each group by number of MPs ascending (smallest at bottom, largest at top)
mp_counts = nodes["party"].value_counts()
_all_present = [p for p in PARTY_ORDER if p in nodes["party"].values]
_opp   = sorted([p for p in _all_present if p not in coalition_parties], key=lambda p: mp_counts.get(p, 0))
_coal  = sorted([p for p in _all_present if p in coalition_parties],     key=lambda p: mp_counts.get(p, 0))
parties_present = _opp + _coal          # opposition bottom → coalition top

party_y = {}
for i, p in enumerate(_opp):
    party_y[p] = float(i)
for i, p in enumerate(_coal):
    party_y[p] = float(len(_opp)) + float(i)

y_min     = -0.5
y_top     = party_y[parties_present[-1]] if parties_present else 0.0
y_max     = y_top + 0.5

rng = np.random.default_rng(42)
nodes["y_base"] = nodes["party"].map(party_y)
nodes["y"]      = nodes["y_base"] + rng.uniform(-0.43, 0.43, size=n)

# ── Plot ─────────────────────────────────────────────────────────────────────
label_map = {
    "BÜNDNIS 90/DIE GRÜNEN": "Die Grünen",
}
def display(p): return label_map.get(p, p)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

# Party band backgrounds (alternating subtle shading)
for i, p in enumerate(parties_present):
    yb = party_y[p]
    alpha = T["band_alpha_even"] if i % 2 == 0 else 0.0
    ax.axhspan(yb - 0.5, yb + 0.5, color=T["band_color"], alpha=alpha, zorder=0)

# Coalition marker: vertical bar just left of the axes, next to the party label
_bar_trans = blended_transform_factory(ax.transAxes, ax.transData)
for p in parties_present:
    if p in coalition_parties:
        yb = party_y[p]
        ax.plot([0.003, 0.003], [yb - 0.48, yb + 0.48],
                transform=_bar_trans,
                color=party_color.get(p, "#888888"),
                linewidth=4, solid_capstyle="butt", clip_on=False, zorder=4)

# Zero line — clipped to band range
ax.plot([0, 0], [y_min, y_max],
        color=T["zero_line"], alpha=T["zero_alpha"],
        linewidth=0.9, linestyle="--", zorder=1)
ax.text(0, y_min - 0.12, "neutral vote", color=T["axis_text"], fontsize=9,
        ha="center", va="top", zorder=2)

# Scatter
for p in parties_present:
    sub = nodes[nodes["party"] == p]
    ax.scatter(sub["x"], sub["y"],
               s=28, color=party_color.get(p, "#888888"),
               alpha=0.75, linewidths=0, zorder=3)

# Highlighted MPs
_BG = dict(boxstyle="square,pad=0.15", facecolor=T["ann_bg"], alpha=1.0, edgecolor="none")

def _seg_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """True if segment AB strictly crosses segment CD."""
    def cp(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1, d2 = cp(cx, cy, dx, dy, ax, ay), cp(cx, cy, dx, dy, bx, by)
    d3, d4 = cp(ax, ay, bx, by, cx, cy), cp(ax, ay, bx, by, dx, dy)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def _clear_pos(hx, y_cands, dot_y, df, placed_boxes, placed_lines, hw=0.26, hh=0.22):
    """Search x over 61 candidates × all provided y candidates.
    y_cands should cover both gaps (above and below the band) so direction is chosen here.
    """
    x_cands = np.clip(hx + np.linspace(-0.9, 0.9, 61), -1.0 + hw, 1.0 - hw)
    best_pos, best_score = (hx, float(y_cands[0])), float("inf")
    for ty in y_cands:
        nearby = df[np.abs(df["y"] - ty) < hh + 0.15]
        for cx in x_cands:
            dot_n   = int((np.abs(nearby["x"] - cx) < hw).sum()) if not nearby.empty else 0
            ann_n   = sum(1 for (px, py, pw, ph) in placed_boxes
                          if abs(cx - px) < hw + pw and abs(ty - py) < hh + ph)
            own_dot = 1 if (abs(cx - hx) < hw and abs(ty - dot_y) < hh) else 0
            cross_n = sum(1 for (ldx, ldy, ltx, lty) in placed_lines
                          if _seg_cross(hx, dot_y, cx, ty, ldx, ldy, ltx, lty))
            dx = abs(cx - hx)
            dy = abs(ty - dot_y)
            score = dot_n + ann_n * 100 + own_dot * 50 + cross_n * 12 + dx * 12.0 + dy * 4.0
            if score < best_score:
                best_score, best_pos = score, (float(cx), float(ty))
    return best_pos

placed_boxes = []
placed_lines  = []
for name, label, nudge_x, nudge_y in HIGHLIGHTS.get(PERIOD, []):
    row = nodes[nodes["name"] == name]
    if row.empty:
        print(f"  Warning: '{name}' not found")
        continue
    row = row.iloc[0]
    dot_color = party_color.get(row["party"], "#888888")
    ax.scatter(row["x"], row["y"], s=70, color=dot_color,
               linewidths=0, zorder=5)

    # Build y candidates from BOTH gaps (above and below the band).
    # _clear_pos picks whichever gap yields the best placement, avoiding the
    # collision that happens when adjacent parties fight over the same gap.
    _yb = row["y_base"]
    # Cap at y_max-0.5 / y_min+0.5: label text (which grows beyond ty) stays in frame
    _gap_above = np.linspace(_yb + 0.52, min(_yb + 0.95, y_max - 0.5), 20)
    _gap_below = np.linspace(max(_yb - 0.95, y_min + 0.5), _yb - 0.52, 20)
    _gap_above = _gap_above[_gap_above >= _yb + 0.5]
    _gap_below = _gap_below[_gap_below <= _yb - 0.5]
    _y_cands   = np.concatenate([_gap_above, _gap_below])
    if len(_y_cands) == 0:
        _y_cands = np.array([_yb - 0.6])

    # Find the clearest (x, y) position across both gaps
    tx, ty = _clear_pos(row["x"], _y_cands, row["y"], nodes, placed_boxes, placed_lines)
    placed_boxes.append((tx, ty, 0.26, 0.22))
    placed_lines.append((row["x"], row["y"], tx, ty))

    # Infer direction from which gap was chosen
    direction = +1 if ty >= _yb else -1

    # Leader line from dot to just short of label — behind dots (zorder=2)
    line_end_y = ty - direction * 0.08
    ax.plot([row["x"], tx], [row["y"], line_end_y],
            color=T["ann_line"], alpha=T["ann_line_alpha"], linewidth=0.8, zorder=2)

    lines = label.split("\n")
    name_line   = lines[0]
    detail_line = lines[1] if len(lines) > 1 else ""
    va_name   = "bottom"   # name always visually above
    va_detail = "top"      # detail always visually below
    ax.text(tx, ty, name_line,
            color=T["ann_text"], fontsize=9.5, fontweight="bold",
            ha="center", va=va_name, bbox=_BG, zorder=7)
    if detail_line:
        ax.text(tx, ty, detail_line,
                color=T["ann_detail"], fontsize=8.5, fontweight="normal",
                ha="center", va=va_detail, bbox=_BG, zorder=7)

# Party labels — slightly smaller than title
ax.set_yticks([party_y[p] for p in parties_present])
ax.set_yticklabels([display(p) for p in parties_present],
                   color=T["text"], fontsize=12, fontweight="bold")
ax.tick_params(axis="y", length=0)
ax.set_ylim(y_min - 0.1, y_max + 0.1)

# Gridlines
for p in parties_present:
    ax.axhline(party_y[p], color=T["text"], alpha=T["grid_alpha"], linewidth=0.5, zorder=1)

# x axis labels
ax.set_xticks([])
ax.spines[:].set_visible(False)
ax.text(-1.0, y_min - 0.12, "← Voted with opposition", transform=ax.transData,
        color=T["axis_text"], fontsize=10, ha="left", va="top")
ax.text( 1.0, y_min - 0.12, "Voted with coalition →",  transform=ax.transData,
        color=T["axis_text"], fontsize=10, ha="right", va="top")

years = PERIOD.replace("bundestag_", "").replace("_", "–")

plt.tight_layout(pad=1.5)

# Find the leftmost x of the y-tick labels (party names) in axes-fraction coords,
# so the title block aligns with them rather than with the axes left spine.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
tick_labels = ax.get_yticklabels()
if tick_labels:
    min_x_disp = min(tl.get_window_extent(renderer).x0 for tl in tick_labels)
    title_x = ax.transAxes.inverted().transform((min_x_disp, 0))[0]
else:
    title_x = 0.0

# ── Economist-style title block
ax.plot([title_x, title_x + 0.032], [1.085, 1.085],
        color="#E3120B", linewidth=4, solid_capstyle="butt",
        transform=ax.transAxes, clip_on=False, zorder=10)
ax.text(title_x, 1.072, f"Bundestag {years}",
        transform=ax.transAxes, color=T["text"],
        fontsize=18, fontweight="bold", va="top", ha="left", clip_on=False)
_sub_base = "Voting similarity of MPs, measured by Cohen's κ"
if _coal:
    _sub_base += "   |   Coalition  "
_sub_t = ax.text(title_x, 1.028, _sub_base,
                 transform=ax.transAxes, color=T["subtext"],
                 fontsize=9, va="top", ha="left", clip_on=False)
if _coal:
    fig.canvas.draw()
    _x_disp = _sub_t.get_window_extent(renderer).x1
    # Coalition parties sorted largest-first for subtitle readability
    for i, p in enumerate(sorted(_coal, key=lambda p: -mp_counts.get(p, 0))):
        _x_ax = ax.transAxes.inverted().transform((_x_disp, 0))[0]
        _dot = ax.text(_x_ax, 1.028, "●",
                       transform=ax.transAxes,
                       color=party_color.get(p, "#888888"),
                       fontsize=7, va="top", ha="left", clip_on=False)
        fig.canvas.draw()
        _x_disp = _dot.get_window_extent(renderer).x1
        _sep = "  " if i < len(_coal) - 1 else ""
        _x_ax = ax.transAxes.inverted().transform((_x_disp, 0))[0]
        _nm = ax.text(_x_ax, 1.028, f" {display(p)}{_sep}",
                      transform=ax.transAxes, color=T["subtext"],
                      fontsize=9, va="top", ha="left", clip_on=False)
        fig.canvas.draw()
        _x_disp = _nm.get_window_extent(renderer).x1
ax.text(title_x, 1.004, "Parties ranked by number of seats within each group",
        transform=ax.transAxes, color=T["subtext"],
        fontsize=7.5, va="top", ha="left", clip_on=False)
out_path = IMG_DIR / f"{PERIOD}_mp_strip.png"
plt.savefig(out_path, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")

# Also save into the period's own data folder
period_out = BASE_DIR / "output" / PERIOD / f"mp_strip_{_theme}.png"
plt.savefig(period_out, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {period_out}")
