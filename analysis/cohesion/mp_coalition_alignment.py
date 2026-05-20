"""
mp_coalition_alignment.py

Multi-period strip plot: one dot per MP per parliament, y = coalition alignment
(1D MDS oriented so +1 = voted consistently with coalition, -1 = against).
X axis = parliamentary period. Dots coloured by party.

Usage:
    python mp_coalition_alignment.py           # dark mode
    python mp_coalition_alignment.py light     # light mode
"""

import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
COLORS_F  = BASE_DIR / "config" / "party_colours.json"
COAL_F    = BASE_DIR / "config" / "coalitions.json"

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]

PARTY_ORDER = [
    "AfD", "CDU/CSU", "FDP", "SPD",
    "BÜNDNIS 90/DIE GRÜNEN", "Die Linke", "BSW",
]

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(p, p)

# ── Theme ─────────────────────────────────────────────────────────────────────
LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(
        bg="#ffffff", text="#1a1a1a", subtext="#555555", axis_text="#666666",
        zero_line="#333333", zero_alpha=0.30, grid_alpha=0.08,
        band_color="#aaaaaa", band_alpha=0.07,
    )
    LIGHT_OVERRIDES = {"CDU/CSU": "#3a3a3a", "FDP": "#f0c000"}
else:
    T = dict(
        bg="#0d1117", text="white", subtext="#888888", axis_text="#888888",
        zero_line="white", zero_alpha=0.18, grid_alpha=0.05,
        band_color="white", band_alpha=0.03,
    )
    LIGHT_OVERRIDES = {}

# ── Colors ────────────────────────────────────────────────────────────────────
with open(COLORS_F) as f:
    raw_colors = json.load(f)
with open(COAL_F) as f:
    coalitions_raw = json.load(f)

party_color = {canon(k): v for k, v in raw_colors.items()}
party_color.setdefault("fraktionslos", "#888888")
party_color.setdefault("BSW", "#a020f0")
party_color.update(LIGHT_OVERRIDES)

# Normalise coalition keys to underscore format for lookup
def _period_key(k):
    k = k.lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"_+", "_", k)

coalitions_map = {_period_key(k): set(v) for k, v in coalitions_raw.items()}

# ── MDS helper ────────────────────────────────────────────────────────────────
def load_period(period_key):
    data_dir = BASE_DIR / "output" / period_key
    nodes_path = data_dir / "nodes.csv"
    allpairs   = data_dir / "edges_allpairs.csv"
    edges_path = allpairs if allpairs.exists() else data_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        print(f"  Skipping {period_key}: missing data")
        return None

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    nodes["party"] = nodes["party"].map(canon).fillna(nodes["party"])

    # Drop MPs with very few shared-vote records: they get D=1 with everyone,
    # collapsing to a degenerate point at the MDS extreme (the straight line artifact).
    edge_counts = pd.concat([edges["source"], edges["target"]]).value_counts()
    valid_ids   = set(edge_counts[edge_counts >= 20].index)
    nodes = nodes[nodes["person_id"].isin(valid_ids)].reset_index(drop=True)

    node_ids = nodes["person_id"].tolist()
    n        = len(node_ids)
    id2idx   = {nid: i for i, nid in enumerate(node_ids)}

    D = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    src_idx = edges["source"].map(id2idx)
    tgt_idx = edges["target"].map(id2idx)
    mask    = src_idx.notna() & tgt_idx.notna()
    si = src_idx[mask].astype(int).values
    ti = tgt_idx[mask].astype(int).values
    w  = edges.loc[mask, "weight"].values.astype(np.float32)
    D[si, ti] = 1.0 - w
    D[ti, si] = 1.0 - w

    D2       = D.astype(np.float64) ** 2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    B        = -0.5 * (D2 - row_mean - col_mean + D2.mean())
    eigvals, eigvecs = np.linalg.eigh(B)
    top   = np.argmax(eigvals)
    pos1d = eigvecs[:, top] * np.sqrt(max(eigvals[top], 0.0))

    nodes = nodes.copy()
    nodes["x"] = pos1d

    # Orient: coalition mean → positive
    coal = coalitions_map.get(_period_key(period_key), set())
    coal_mask = nodes["party"].isin(coal)
    if coal_mask.sum() > 0:
        if nodes.loc[coal_mask, "x"].mean() < 0:
            nodes["x"] = -nodes["x"]
    else:
        if nodes[nodes["party"] == "CDU/CSU"]["x"].mean() < 0:
            nodes["x"] = -nodes["x"]

    # Normalise to [-1, 1]
    xmin, xmax = nodes["x"].min(), nodes["x"].max()
    nodes["x"] = 2 * (nodes["x"] - xmin) / (xmax - xmin) - 1

    nodes["coalition"] = nodes["party"].isin(coal)
    nodes["period"]    = period_key
    return nodes[["name", "party", "x", "coalition", "period"]]

# ── Load all periods ──────────────────────────────────────────────────────────
frames = []
for pk, label in PERIODS:
    print(f"Loading {pk}…")
    df = load_period(pk)
    if df is not None:
        df["label"] = label
        df["xi"]    = [i for i, (p, _) in enumerate(PERIODS) if p == pk][0]
        frames.append(df)

all_data = pd.concat(frames, ignore_index=True)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

rng = np.random.default_rng(42)
n_periods = len(PERIODS)

# Alternating column shading
for i in range(n_periods):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color=T["band_color"], alpha=T["band_alpha"], zorder=0, lw=0)

# Zero line
ax.plot([-0.5, n_periods - 0.5], [0, 0],
        color=T["zero_line"], alpha=T["zero_alpha"],
        linewidth=0.9, linestyle="--", zorder=1)

# Dots — one pass per party to get clean legend ordering
parties_seen = [p for p in PARTY_ORDER if p in all_data["party"].values]
for p in parties_seen:
    sub = all_data[all_data["party"] == p].copy()
    jitter = rng.uniform(-0.45, 0.45, size=len(sub))
    ax.scatter(sub["xi"] + jitter, sub["x"],
               s=18, color=party_color.get(p, "#888888"),
               alpha=0.65, linewidths=0, zorder=3)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.6, n_periods - 0.4)
ax.set_ylim(-1.15, 1.15)
ax.set_xticks(range(n_periods))
ax.set_xticklabels([lbl for _, lbl in PERIODS],
                   color=T["text"], fontsize=11, fontweight="bold")
ax.tick_params(axis="x", length=0, pad=8)
ax.set_yticks([])
ax.spines[:].set_visible(False)

# Gridlines at ±0.5
for y in [-0.5, 0.5]:
    ax.axhline(y, color=T["text"], alpha=T["grid_alpha"], linewidth=0.5, zorder=1)

# ── Layout: axes flush to left edge (Economist style) ─────────────────────────
fig.subplots_adjust(left=0.005, right=0.99, top=0.88, bottom=0.07)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# Shared x reference: left edge of first column's dots — aligns labels, title, legend
first_seg_disp = ax.transData.transform((-0.45, 0))[0]
title_x = ax.transAxes.inverted().transform((first_seg_disp, 0))[0]

# Axis direction labels inside data area, left-aligned with title
ax.text(title_x, 0.972, "↑ Voted with coalition",
        transform=ax.transAxes, color=T["axis_text"], fontsize=7.5,
        ha="left", va="top", clip_on=True)
ax.text(title_x, 0.028, "↓ Voted with opposition",
        transform=ax.transAxes, color=T["axis_text"], fontsize=7.5,
        ha="left", va="bottom", clip_on=True)

ax.text(title_x, 1.14, "Bundestag 2005–2029",
        transform=ax.transAxes, color=T["text"],
        fontsize=18, fontweight="bold", va="top", ha="left", clip_on=False)
ax.text(title_x, 1.065, "Coalition alignment of individual MPs by parliament · measured by Cohen's κ",
        transform=ax.transAxes, color=T["subtext"],
        fontsize=9, va="top", ha="left", clip_on=False)

# ── Economist-style party legend ──────────────────────────────────────────────
_label_map = {"BÜNDNIS 90/DIE GRÜNEN": "Die Grünen"}
_legend_parties = sorted(
    [p for p in all_data["party"].unique() if p in party_color],
    key=lambda p: _label_map.get(p, p)
)

fig.canvas.draw()
_lx_disp = ax.transData.transform((-0.45, 0))[0]   # start aligned with title
for p in _legend_parties:
    _lx_ax = ax.transAxes.inverted().transform((_lx_disp, 0))[0]
    _dot = ax.text(_lx_ax, 1.030, "●",
                   transform=ax.transAxes,
                   color=party_color.get(p, "#888888"),
                   fontsize=7, va="top", ha="left", clip_on=False)
    fig.canvas.draw()
    _lx_disp = _dot.get_window_extent(renderer).x1
    _lx_ax = ax.transAxes.inverted().transform((_lx_disp, 0))[0]
    _nm = ax.text(_lx_ax, 1.030, f" {_label_map.get(p, p)}  ",
                  transform=ax.transAxes, color=T["subtext"],
                  fontsize=8, va="top", ha="left", clip_on=False)
    fig.canvas.draw()
    _lx_disp = _nm.get_window_extent(renderer).x1

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = IMG_DIR / f"mp_coalition_alignment.png"
plt.savefig(out_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.15,
            facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
