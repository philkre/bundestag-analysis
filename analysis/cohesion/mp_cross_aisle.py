"""
mp_cross_aisle.py

Small-multiple heatmaps showing mean pairwise voting similarity (Cohen's κ)
between every party pair per parliament period.
Diagonal = intra-party cohesion. Off-diagonal = cross-aisle similarity.
Widening gap between diagonal and off-diagonal = increasing polarisation.

Usage:
    python mp_cross_aisle.py           # dark mode
    python mp_cross_aisle.py light     # light mode
"""

import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
COLORS_F = BASE_DIR / "config" / "party_colours.json"

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]

# Display order on heatmap axes (left-to-right / top-to-bottom = right to left politically)
PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD", "BÜNDNIS 90/DIE GRÜNEN", "Die Linke"]

DISPLAY = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
    "CDU/CSU":               "CDU/CSU",
    "Die Linke":             "Linke",
    "BSW":                   "BSW",
    "AfD":                   "AfD",
    "FDP":                   "FDP",
    "SPD":                   "SPD",
    "fraktionslos":          "fraktl.",
}

ALIASES = {
    "DIE GRÜNEN":  "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":   "Die Linke",
    "Die Linke.":  "Die Linke",
}
def canon(p): return ALIASES.get(p, p)

# ── Theme ─────────────────────────────────────────────────────────────────────
LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(
        bg="#ffffff", text="#1a1a1a", subtext="#555555",
        cell_text="#1a1a1a", na_color="#e8e8e8",
        grid_color="#ffffff",
    )
    LIGHT_OVERRIDES = {"CDU/CSU": "#3a3a3a", "FDP": "#f0c000"}
else:
    T = dict(
        bg="#0d1117", text="white", subtext="#888888",
        cell_text="white", na_color="#1e2530",
        grid_color="#0d1117",
    )
    LIGHT_OVERRIDES = {}

# ── Colors ────────────────────────────────────────────────────────────────────
with open(COLORS_F) as f:
    raw_colors = json.load(f)

party_color = {canon(k): v for k, v in raw_colors.items()}
party_color.setdefault("fraktionslos", "#888888")
party_color.setdefault("BSW", "#a020f0")
party_color.update(LIGHT_OVERRIDES)

# ── Diverging colormap: red (−1) → white (0) → blue (+1) ─────────────────────
if LIGHT_MODE:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "kappa", ["#c0392b", "#f5f5f5", "#2471a3"], N=256
    )
else:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "kappa", ["#c0392b", "#1e2530", "#2471a3"], N=256
    )

# ── Load & aggregate ──────────────────────────────────────────────────────────
def load_period(period_key):
    d = BASE_DIR / "output" / period_key
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(lambda p: canon(str(p)))
    ep = d / "edges_allpairs.csv"
    edges = pd.read_csv(ep if ep.exists() else d / "edges.csv")
    return nodes, edges

def party_kappa_matrix(nodes, edges):
    """Return DataFrame: mean κ per (party_A, party_B) pair."""
    pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))
    edges = edges.copy()
    edges["party_s"] = edges["source"].map(pid_to_party)
    edges["party_t"] = edges["target"].map(pid_to_party)
    edges = edges.dropna(subset=["party_s", "party_t"])
    # Normalise pair order so (A,B) == (B,A)
    edges["pa"] = np.where(edges["party_s"] <= edges["party_t"],
                           edges["party_s"], edges["party_t"])
    edges["pb"] = np.where(edges["party_s"] <= edges["party_t"],
                           edges["party_t"], edges["party_s"])
    agg = edges.groupby(["pa", "pb"])["weight"].mean().reset_index()
    # Build symmetric matrix
    parties = sorted(set(edges["pa"]) | set(edges["pb"]))
    mat = pd.DataFrame(np.nan, index=parties, columns=parties)
    for _, row in agg.iterrows():
        mat.loc[row["pa"], row["pb"]] = row["weight"]
        mat.loc[row["pb"], row["pa"]] = row["weight"]
    return mat

print("Loading periods…")
matrices = {}
for pk, lbl in PERIODS:
    print(f"  {lbl}")
    try:
        nodes, edges = load_period(pk)
        matrices[lbl] = party_kappa_matrix(nodes, edges)
    except FileNotFoundError:
        print(f"    skipped (no data)")

# ── Determine union of parties across all periods (in display order) ──────────
all_parties = []
for p in PARTY_ORDER:
    if any(p in m.index for m in matrices.values()):
        all_parties.append(p)
# Append any unlisted parties (fraktionslos etc.) at the end
for m in matrices.values():
    for p in m.index:
        if p not in all_parties:
            all_parties.append(p)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_periods = len(matrices)
n_parties = len(all_parties)
tick_labels = [DISPLAY.get(p, p) for p in all_parties]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor(T["bg"])

vmin, vmax = -0.3, 1.0   # κ range across all periods

for ax, (lbl, mat) in zip(axes.flat, matrices.items()):
    ax.set_facecolor(T["bg"])

    # Build display matrix in all_parties order
    disp = np.full((n_parties, n_parties), np.nan)
    for i, pi in enumerate(all_parties):
        for j, pj in enumerate(all_parties):
            if pi in mat.index and pj in mat.columns:
                v = mat.loc[pi, pj]
                if not np.isnan(v):
                    disp[i, j] = v

    # Draw NA cells first
    na_arr = np.where(np.isnan(disp), 1.0, np.nan)
    ax.imshow(na_arr, aspect="auto", cmap=mcolors.ListedColormap([T["na_color"]]),
              vmin=0, vmax=1, interpolation="nearest")

    # Draw kappa values
    im = ax.imshow(disp, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Annotate cells with κ value
    for i in range(n_parties):
        for j in range(n_parties):
            v = disp[i, j]
            if not np.isnan(v):
                # Choose white or dark text based on background
                norm_v = (v - vmin) / (vmax - vmin)
                cell_bg = cmap(norm_v)
                lum = 0.299*cell_bg[0] + 0.587*cell_bg[1] + 0.114*cell_bg[2]
                txt_color = "#111111" if lum > 0.5 else "#eeeeee"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=txt_color, fontweight="normal")

    # Grid lines between cells
    for k in range(n_parties + 1):
        ax.axhline(k - 0.5, color=T["grid_color"], linewidth=0.8)
        ax.axvline(k - 0.5, color=T["grid_color"], linewidth=0.8)

    # Coloured party name labels on x and y axes
    ax.set_xticks(range(n_parties))
    ax.set_yticks(range(n_parties))
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.tick_params(length=0, pad=2)

    # Colour each tick label by party
    for tick, p in zip(ax.get_xticklabels(), all_parties):
        tick.set_color(party_color.get(p, T["text"]))
    for tick, p in zip(ax.get_yticklabels(), all_parties):
        tick.set_color(party_color.get(p, T["text"]))

    ax.spines[:].set_visible(False)

    # Period label
    ax.set_title(lbl, color=T["text"], fontsize=12,
                 fontweight="bold", pad=8)

# Hide unused panels
for ax in axes.flat[n_periods:]:
    ax.set_visible(False)

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.55])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
cb = fig.colorbar(sm, cax=cbar_ax)
cb.ax.yaxis.set_tick_params(color=T["subtext"], labelsize=7)
cb.outline.set_visible(False)
for lbl_t in cb.ax.get_yticklabels():
    lbl_t.set_color(T["subtext"])
cbar_ax.text(0.5, 1.04, "κ = 1\n(always\nagreed)", transform=cbar_ax.transAxes,
             color=T["subtext"], fontsize=6, ha="center", va="bottom")
cbar_ax.text(0.5, -0.04, "κ = −1\n(always\ndisagreed)", transform=cbar_ax.transAxes,
             color=T["subtext"], fontsize=6, ha="center", va="top")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.03, 0.97, "Cross-aisle voting similarity, Bundestag 2005–2029",
         color=T["text"], fontsize=16, fontweight="bold", va="top", ha="left")
fig.text(0.03, 0.935, "Mean pairwise Cohen's κ between party groups per legislature  ·  "
         "Diagonal = within-party cohesion  ·  Off-diagonal = cross-aisle similarity",
         color=T["subtext"], fontsize=8.5, va="top", ha="left")

fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.10,
                    hspace=0.45, wspace=0.35)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = IMG_DIR / f"mp_cross_aisle.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2,
            facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
