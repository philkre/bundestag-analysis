"""
uk_commons_ridgeline.py

Ridgeline plot of MP coupling distributions per UK parliament period,
with the Prime Minister marked as a tick on each ridge.

Usage
-----
  python analysis/ising/uk_commons_ridgeline.py
  python analysis/ising/uk_commons_ridgeline.py light
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
CACHE    = BASE_DIR / "output" / "uk_ising_field_cache.csv"

# Oldest at bottom, newest at top
PERIOD_ORDER = ["2017–19", "2019–24", "2024–"]

# PMs per period — Truss (44 days) included; Starmer too few votes as PM
PM = {
    "2017–19": [("Theresa May",      "Conservative")],
    "2019–24": [("Boris Johnson",    "Conservative"),
                ("Elizabeth Truss",  "Conservative"),
                ("Rishi Sunak",      "Conservative")],
    "2024–":   [],   # Starmer: 36/529 votes (7%) — below PLM threshold
}
PM_NOTE = {
    "2024–": "Starmer: 7% participation — below PLM threshold",
}

PARTY_COLORS = {
    "Conservative":      "#0087DC",
    "Labour":            "#E4003B",
    "Liberal Democrat":  "#FAA61A",
    "SNP":               "#009DD1",
    "Green":             "#02A95B",
    "Plaid Cymru":       "#005B54",
    "DUP":               "#D46A4B",
    "Reform UK":         "#12B6CF",
    "Independent":       "#888888",
}
def party_color(p: str) -> str:
    for key, col in PARTY_COLORS.items():
        if key.lower() in p.lower():
            return col
    return "#888888"

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme / "ising"
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", sub="#555555", ax="#cccccc")
    ridge_line = "#ffffff"
    tint_mix   = 0.55
else:
    T = dict(bg="#0d1117", text="white", sub="#888888", ax="#333333")
    ridge_line = "#0d1117"
    tint_mix   = 0.45

if not CACHE.exists():
    sys.exit(f"Cache missing: {CACHE}\nRun uk_commons_plm_cache.py first.")

all_df = pd.read_csv(CACHE)
all_df["pct"] = all_df.groupby("period")["coupling"].rank(pct=True, method="first") * 100

_BG_RGB = to_rgba(T["bg"])[:3]
def tint(hex_color: str, mix: float) -> tuple:
    r, g, b = to_rgba(hex_color)[:3]
    return (r + (_BG_RGB[0] - r) * mix,
            g + (_BG_RGB[1] - g) * mix,
            b + (_BG_RGB[2] - b) * mix)

def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

xmax = np.nanpercentile(all_df["coupling"], 99.5) * 1.05
xgrid = np.linspace(0, xmax, 300)

XMIN   = max(0, all_df["coupling"].min() * 0.8)
overlap = 1.7
row_gap = 1.6

# Tint by governing party (Conservatives = blue, Labour = red)
GOVERN_PARTY = {
    "2017–19": "Conservative",
    "2019–24": "Conservative",
    "2024–":   "Labour",
}

for yi, period in enumerate(PERIOD_ORDER):
    sub = all_df[all_df["period"] == period]
    if sub.empty:
        continue
    base = yi * row_gap
    vals = sub["coupling"].dropna().values

    kde  = gaussian_kde(vals)
    dens = kde(xgrid)
    dens = dens / dens.max() * overlap

    fill_c = tint(party_color(GOVERN_PARTY[period]), tint_mix)
    ax.fill_between(xgrid, base, base + dens,
                    color=fill_c, alpha=0.92, zorder=yi * 2,
                    edgecolor=ridge_line, linewidth=1.2)
    ax.plot([XMIN, xmax], [base, base], color=T["ax"], lw=0.6,
            alpha=0.5, zorder=yi * 2 - 1)

    # PM ticks — stagger label heights when multiple PMs in same period
    pm_list = PM.get(period, [])
    for pi, (pm_name, pm_party) in enumerate(pm_list):
        crow = sub[sub["name"] == pm_name]
        if crow.empty:
            print(f"  Warning: {pm_name} not found in {period}")
            continue
        cx  = float(crow["coupling"].iloc[0])
        cpc = float(crow["pct"].iloc[0])
        col = party_color(pm_party)
        ch_h = float(kde(cx)[0] / kde(xgrid).max() * overlap)
        ax.plot([cx, cx], [base, base + ch_h], color=col, lw=2.4,
                zorder=yi * 2 + 1)
        ax.scatter([cx], [base + ch_h], s=60, c=col, zorder=yi * 2 + 1,
                   linewidths=1.2, edgecolors=T["bg"])
        surname = pm_name.split()[-1]
        # Stagger vertically: 0.12 base gap + 0.28 per PM index
        label_y_offset = 0.12 + pi * 0.30
        t = ax.text(cx, base + ch_h + label_y_offset,
                    f"{surname} · {ordinal(round(cpc))} pct",
                    fontsize=9, fontweight="bold", color=T["text"],
                    ha="center", va="bottom", zorder=100)
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=T["bg"])])

    ax.text(XMIN - xmax * 0.01, base, period, fontsize=11, color=T["text"],
            ha="right", va="bottom")

    # Note for periods where PM couldn't be estimated
    if period in PM_NOTE:
        ax.text(XMIN + xmax * 0.02, base + overlap * 0.5,
                PM_NOTE[period], fontsize=8, color=T["sub"],
                ha="left", va="center", style="italic", zorder=100)

ax.set_xlim(XMIN, xmax)
ax.set_ylim(-0.2, (len(PERIOD_ORDER) - 1) * row_gap + overlap + 0.8)
ax.set_yticks([])
ax.set_xlabel("Voting coupling strength  —  mean |J_ij| (×10³)",
              color=T["sub"], fontsize=11)
ax.tick_params(axis="x", colors=T["sub"], labelsize=9, length=0)
ax.set_axisbelow(True)
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(T["ax"])

ax.set_title("UK Prime Ministers rank in the bottom 3–4% of voting coupling",
             color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.09,
        "Each ridge = one parliament's MP coupling distribution, coloured by governing party  ·  "
        "tick = the PM  ·  Starmer voted in only 7% of divisions as PM — below PLM estimation threshold  ·  "
        "coupling = mean |J_ij| from regularised PLM inverse Ising",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "uk_commons_ridgeline.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"Saved → {out}")
