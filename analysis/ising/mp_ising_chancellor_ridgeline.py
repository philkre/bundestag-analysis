"""
mp_ising_chancellor_ridgeline.py

Ridgeline (joy plot): coupling distribution per parliamentary period,
with the chancellor marked as a vertical tick on each ridge.

Reads cache from mp_ising_field computation.

Usage
-----
  python analysis/ising/mp_ising_chancellor_ridgeline.py
  python analysis/ising/mp_ising_chancellor_ridgeline.py light
"""

from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
CACHE    = BASE_DIR / "output" / "mp_ising_field_cache.csv"

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

# Oldest at bottom → newest at top; newest drawn on top of overlaps
PERIOD_ORDER = ["2005–09", "2009–13", "2013–17", "2017–21", "2021–25", "2025–29"]
CHANCELLOR = {
    "2005–09": ("Angela Merkel",  "CDU/CSU"),
    "2009–13": ("Angela Merkel",  "CDU/CSU"),
    "2013–17": ("Angela Merkel",  "CDU/CSU"),
    "2017–21": ("Angela Merkel",  "CDU/CSU"),
    "2021–25": ("Olaf Scholz",    "SPD"),
    "2025–29": ("Friedrich Merz", "CDU/CSU"),
}

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme / "ising"
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", sub="#555555", grid="#dddddd", ax="#cccccc")
    ridge_line = "#ffffff"
    tint_mix = 0.55   # blend toward white for fill
else:
    T = dict(bg="#0d1117", text="white", sub="#888888", grid="#1e2530", ax="#333333")
    ridge_line = "#0d1117"
    tint_mix = 0.45   # blend toward bg for fill

with open(BASE_DIR / "config" / "party_colours.json") as f:
    _raw = json.load(f)
party_color = {canon(k): v for k, v in _raw.items()}
party_color["CDU/CSU"] = "#3a3a3a" if LIGHT_MODE else "#dddddd"

if not CACHE.exists():
    sys.exit(f"Cache missing: {CACHE}\nRun mp_ising_chancellor_path.py first.")

all_df = pd.read_csv(CACHE)
all_df["party"] = all_df["party"].map(canon)
all_df["pct"] = all_df.groupby("period")["coupling"].rank(
    pct=True, method="first") * 100


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


_BG_RGB = to_rgba(T["bg"])[:3]
def tint(hex_color, mix):
    """Blend color toward background by `mix` (0=full colour, 1=bg)."""
    r, g, b = to_rgba(hex_color)[:3]
    return (r + (_BG_RGB[0] - r) * mix,
            g + (_BG_RGB[1] - g) * mix,
            b + (_BG_RGB[2] - b) * mix)


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9.5))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

xmax = np.nanpercentile(all_df["coupling"], 99.5) * 1.05
xgrid = np.linspace(0, xmax, 300)

XMIN = 2.5        # x-axis start

overlap = 1.7     # ridge height (in row units)
row_gap = 1.6     # vertical spacing between ridge baselines

n_periods = len(PERIOD_ORDER)

for yi, period in enumerate(PERIOD_ORDER):
    sub = all_df[all_df["period"] == period]
    if sub.empty:
        continue
    base = yi * row_gap
    vals = sub["coupling"].dropna().values

    kde = gaussian_kde(vals)
    dens = kde(xgrid)
    dens = dens / dens.max() * overlap   # normalise height

    # Fill = governing chancellor's party, tinted toward background
    _, ch_party = CHANCELLOR[period]
    fill_c = tint(party_color.get(ch_party, "#888888"), tint_mix)

    ax.fill_between(xgrid, base, base + dens,
                    color=fill_c, alpha=0.92, zorder=yi * 2,
                    edgecolor=ridge_line, linewidth=1.2)
    # baseline
    ax.plot([XMIN, xmax], [base, base], color=T["ax"], lw=0.6,
            alpha=0.5, zorder=yi * 2 - 1)

    # Chancellor tick
    name, party = CHANCELLOR[period]
    crow = sub[sub["name"] == name]
    if not crow.empty:
        cx  = float(crow["coupling"].iloc[0])
        cpc = float(crow["pct"].iloc[0])
        col = party_color.get(party, "#888888")
        ch_h = float(kde(cx)[0] / kde(xgrid).max() * overlap)
        ax.plot([cx, cx], [base, base + ch_h], color=col, lw=2.4,
                zorder=yi * 2 + 1)
        ax.scatter([cx], [base + ch_h], s=60, c=col, zorder=yi * 2 + 1,
                   linewidths=1.2, edgecolors=T["bg"])
        t = ax.text(cx, base + ch_h + 0.12,
                    f"{name.split()[-1]} · {ordinal(round(cpc))} pct",
                    fontsize=9, fontweight="bold", color=T["text"],
                    ha="center", va="bottom", zorder=100)
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=T["bg"])])

    # Period label on left
    ax.text(XMIN - xmax * 0.01, base, period, fontsize=11, color=T["text"],
            ha="right", va="bottom")

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

ax.set_title("Every chancellor sits in the low tail of voting coupling",
             color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.09,
        "Each ridge = one period's distribution of MP coupling, coloured by governing party  ·  "
        "tick = the chancellor  ·  coupling = mean |J_ij| from regularised PLM inverse Ising",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "mp_ising_chancellor_ridgeline.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"Saved → {out}")
