"""
mp_kappa_dots_govoppo.py

Dot strip plot of government–opposition κ values.
One dot per MP pair, colored by opposition party, y-jittered.

Usage:
    python analysis/mp_kappa_dots_govoppo.py        # dark
    python analysis/mp_kappa_dots_govoppo.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
rng = np.random.default_rng(42)

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
    "SPD": "SPD",
}
def short(p): return SHORT.get(p, p)

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
ALPHA   = 0.18 if LIGHT_MODE else 0.22

if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"
    party_color["FDP"]     = "#f5d800"

_GRID = np.linspace(-1, 1, 512)

def density_jitter(kappa, max_width=0.80, bw=0.04):
    """Density-proportional upward jitter. KDE evaluated on grid, interpolated."""
    kde  = gaussian_kde(kappa, bw_method=bw)
    dens = kde(_GRID)
    dens /= dens.max()
    interp  = interp1d(_GRID, dens, bounds_error=False, fill_value=0.0)
    d_at_pt = np.clip(interp(kappa), 0, 1)
    return rng.uniform(0, 1, size=len(kappa)) * d_at_pt * max_width


fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.patch.set_facecolor(BG)

for ax, (pk, lbl, coal_key) in zip(axes.flat, PERIODS):
    d     = BASE_DIR / "output" / pk
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    edges = pd.read_csv(d / "edges_allpairs.csv")
    pid2p = dict(zip(nodes["person_id"], nodes["party"]))
    edges["pa"] = edges["source"].map(pid2p)
    edges["pb"] = edges["target"].map(pid2p)
    edges = edges.dropna(subset=["pa", "pb"])

    coal_set = set(canon(p) for p in coalitions_map.get(coal_key, []))
    edges["pa_coal"] = edges["pa"].isin(coal_set)
    edges["pb_coal"] = edges["pb"].isin(coal_set)

    cross = edges[edges["pa_coal"] != edges["pb_coal"]].copy()

    # Vectorised opp_party assignment
    cross["opp_party"] = np.where(cross["pa_coal"], cross["pb"], cross["pa"])

    # Drop fraktionslos
    cross = cross[cross["opp_party"] != "fraktionslos"]

    all_kappa  = cross["weight"].values
    opp_labels = cross["opp_party"].values

    # Jitter (fast: KDE on grid + interpolate)
    y_jitter = density_jitter(all_kappa)

    # Build RGBA color array in one pass — single scatter call
    rgba = np.array([to_rgba(party_color.get(p, "#888888")) for p in opp_labels])
    rgba[:, 3] = ALPHA

    ax.scatter(all_kappa, y_jitter, s=0.25, c=rgba,
               linewidths=0, rasterized=True)

    # κ = 0 reference
    ax.axvline(0, color=SUBTEXT, lw=0.8, ls="--", alpha=0.45, zorder=5)

    # Mean line
    mean_val = all_kappa.mean()
    ax.axvline(mean_val, color=SUBTEXT, lw=1.1, alpha=0.75, zorder=5)
    ax.text(mean_val + 0.02, 0.97, f"μ={mean_val:+.2f}",
            color=SUBTEXT, fontsize=7.5, va="top",
            transform=ax.get_xaxis_transform())

    # Styling
    ax.set_xlim(-1, 0.5)
    ax.set_ylim(-0.05, 0.90)
    ax.set_facecolor(BG)
    ax.yaxis.set_visible(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.4, alpha=0.8, zorder=0)
    ax.set_xlabel("Cohen's κ", color=SUBTEXT, fontsize=8.5, labelpad=5)

    # Period title
    ax.text(0.0, 1.13, lbl, transform=ax.transAxes, color=TEXT,
            fontsize=14, fontweight="bold", va="bottom", ha="left")

    # Coalition labels
    coal_parties = [canon(p) for p in coalitions_map.get(coal_key, [])]
    CHAR_W = 0.0115; cursor = 0.0
    for i, p in enumerate(coal_parties):
        if i > 0:
            ax.text(cursor, 1.055, " · ", transform=ax.transAxes,
                    color=SUBTEXT, fontsize=8, va="bottom")
            cursor += 3 * CHAR_W
        col  = party_color.get(p, SUBTEXT)
        name = short(p)
        ax.text(cursor, 1.055, name, transform=ax.transAxes, color=col,
                fontsize=8, fontweight="bold", va="bottom")
        cursor += len(name) * CHAR_W

    # Legend — parties sorted by mean κ
    opp_parties = sorted(set(opp_labels),
                         key=lambda p: cross[cross["opp_party"] == p]["weight"].mean())
    handles = [mpatches.Patch(color=party_color.get(p, "#888888"), label=short(p))
               for p in opp_parties]
    ax.legend(handles=handles[::-1], loc="upper left",
              fontsize=7, frameon=False,
              labelcolor=TEXT if LIGHT_MODE else "white",
              handlelength=1.0, handleheight=0.9,
              borderpad=0.3, labelspacing=0.25)

    print(f"  {lbl}: {len(cross):,} pairs")

fig.text(0.5, 0.985,
         "Government–opposition Cohen's κ  ·  all Bundestag periods",
         ha="center", va="top", color=TEXT, fontsize=16, fontweight="bold")
fig.text(0.5, 0.960,
         "Each dot is one government–opposition MP pair, positioned by Cohen's κ and colored by opposition party.  "
         "Height is density-proportional.  Vertical line marks the mean.",
         ha="center", va="top", color=SUBTEXT, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.955])
plt.subplots_adjust(hspace=0.42, wspace=0.10)

out = IMG_DIR / "kappa_dots_govoppo.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
