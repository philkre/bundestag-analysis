"""
mp_kappa_dist_split.py

2×3 grid: one subplot per Bundestag period.
Each subplot shows cross-party (solid fill) and intra-party (dashed)
κ distributions, plus coalition annotation.

Usage:
    python analysis/mp_kappa_dist_split.py        # dark
    python analysis/mp_kappa_dist_split.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09", "Bundestag 2005 - 2009"),
    ("bundestag_2009_2013", "2009–13", "Bundestag 2009 - 2013"),
    ("bundestag_2013_2017", "2013–17", "Bundestag 2013 - 2017"),
    ("bundestag_2017_2021", "2017–21", "Bundestag 2017 - 2021"),
    ("bundestag_2021_2025", "2021–25", "Bundestag 2021 - 2025"),
    ("bundestag_2025_2029", "2025–29", "Bundestag 2025 - 2029"),
]

ALIASES = {"DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
           "DIE LINKE": "Die Linke", "Die Linke.": "Die Linke"}
def canon(p): return ALIASES.get(str(p), str(p))

SHORT = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
    "CDU/CSU": "CDU/CSU", "Die Linke": "Linke",
    "BSW": "BSW", "AfD": "AfD", "FDP": "FDP",
    "SPD": "SPD", "fraktionslos": "fraktl.",
}

with open(BASE_DIR / "config" / "coalitions.json") as f:
    coalitions_map = json.load(f)

with open(BASE_DIR / "config" / "party_colours.json") as f:
    raw_colors = json.load(f)
party_color = {canon(k): v for k, v in raw_colors.items()}
party_color.setdefault("BSW", "#a020f0")

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#ffffff" if LIGHT_MODE else "#0d1117"
TEXT    = "#1a1a1a" if LIGHT_MODE else "white"
SUBTEXT = "#666666" if LIGHT_MODE else "#888888"
GRID    = "#e0e0e0" if LIGHT_MODE else "#1a2030"
CROSS_C = "#2a2a2a" if LIGHT_MODE else "white"
INTRA_C = "#555555" if LIGHT_MODE else "#aaaaaa"

if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"
    party_color["FDP"]     = "#f5d800"

x = np.linspace(-1, 1, 600)

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.patch.set_facecolor(BG)

for ax, (pk, lbl, coal_key) in zip(axes.flat, PERIODS):
    d      = BASE_DIR / "output" / pk
    nodes  = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    edges  = pd.read_csv(d / "edges_allpairs.csv")
    pid2p  = dict(zip(nodes["person_id"], nodes["party"]))
    edges["pa"] = edges["source"].map(pid2p)
    edges["pb"] = edges["target"].map(pid2p)
    edges  = edges.dropna(subset=["pa", "pb"])

    coal_set = set(canon(p) for p in coalitions_map.get(coal_key, []))
    edges["pa_coal"] = edges["pa"].isin(coal_set)
    edges["pb_coal"] = edges["pb"].isin(coal_set)

    coal_e = edges[ edges["pa_coal"] &  edges["pb_coal"]]["weight"].values
    oppo_e = edges[~edges["pa_coal"] & ~edges["pb_coal"]]["weight"].values

    kde_c  = gaussian_kde(coal_e, bw_method=0.06)(x)
    kde_i  = gaussian_kde(oppo_e, bw_method=0.06)(x)

    # Normalise both to same peak so they're visually comparable
    kde_c  = kde_c / kde_c.max()
    kde_i  = kde_i / kde_i.max()

    ax.set_facecolor(BG)

    # Cross-party: filled + solid line
    ax.fill_between(x, kde_c, alpha=0.12 if LIGHT_MODE else 0.15, color=CROSS_C)
    ax.plot(x, kde_c, color=CROSS_C, lw=1.8)

    # Intra-party: dashed, no fill
    ax.plot(x, kde_i, color=INTRA_C, lw=1.4, ls="--", alpha=0.80)

    # κ = 0 reference
    ax.axvline(0, color=SUBTEXT, lw=0.8, ls="--", alpha=0.4)

    # Styling
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.28)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=7.5)
    ax.yaxis.set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.4, alpha=0.8)
    ax.set_xlabel("Cohen's κ", color=SUBTEXT, fontsize=8.5, labelpad=6)

    # Period title at top-left in axes coords
    ax.text(0.0, 1.13, lbl,
            transform=ax.transAxes, color=TEXT,
            fontsize=15, fontweight="bold", va="bottom", ha="left")

    # Coalition parties directly under title — coloured party names
    coal_parties = [canon(p) for p in coalitions_map.get(coal_key, [])]
    if coal_parties:
        CHAR_W = 0.0115
        cursor = 0.0
        for i, p in enumerate(coal_parties):
            if i > 0:
                ax.text(cursor, 1.055, " · ",
                        transform=ax.transAxes, color=SUBTEXT,
                        fontsize=8, va="bottom", zorder=10)
                cursor += 3 * CHAR_W
            col  = party_color.get(p, SUBTEXT)
            name = SHORT.get(p, p)
            ax.text(cursor, 1.055, name,
                    transform=ax.transAxes, color=col,
                    fontsize=8, fontweight="bold", va="bottom", zorder=10)
            cursor += len(name) * CHAR_W


# Supertitle
fig.text(0.5, 0.985,
         "Distribution of pairwise Cohen's κ  ·  all Bundestag periods",
         ha="center", va="top", color=TEXT,
         fontsize=16, fontweight="bold")
fig.text(0.5, 0.960,
         "Kernel density estimates of all pairwise Cohen's κ values, peak-normalised.  "
         "——  coalition pairs    - - -  opposition pairs",
         ha="center", va="top", color=SUBTEXT, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.955])
plt.subplots_adjust(hspace=0.38, wspace=0.08)

out = IMG_DIR / "kappa_dist_split_coaloppo.png"
plt.savefig(out, dpi=400, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
