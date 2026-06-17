"""
mp_ising_chancellor_strip.py

Strip plot: every MP's coupling percentile per period, chancellor highlighted.
Shows the chancellor sitting at the bottom of the full parliament distribution.

  x  coupling percentile within period (0 = least coupled, 100 = most)
  y  period row (jittered for the MP cloud)

Reads cache from mp_ising_field computation.

Usage
-----
  python analysis/ising/mp_ising_chancellor_strip.py
  python analysis/ising/mp_ising_chancellor_strip.py light
"""

from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
CACHE    = BASE_DIR / "output" / "mp_ising_field_cache.csv"

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

# Period order (bottom → top) and chancellor per period
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
    cloud = "#bbbbbb"
else:
    T = dict(bg="#0d1117", text="white", sub="#888888", grid="#1e2530", ax="#333333")
    cloud = "#3a4452"

with open(BASE_DIR / "config" / "party_colours.json") as f:
    _raw = json.load(f)
party_color = {canon(k): v for k, v in _raw.items()}
party_color.setdefault("BSW", "#a020f0")
party_color.setdefault("fraktionslos", "#888888")
if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"; party_color["FDP"] = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"; party_color["FDP"] = "#f5d800"

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

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7.5))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

rng = np.random.default_rng(0)
row_h = 0.62   # jitter band height

xmax = np.nanpercentile(all_df["coupling"], 99.5) * 1.05
parties_seen = set()

for yi, period in enumerate(PERIOD_ORDER):
    sub = all_df[all_df["period"] == period]
    if sub.empty:
        continue
    jitter = rng.uniform(-row_h/2, row_h/2, len(sub))
    cols = [party_color.get(p, "#888888") for p in sub["party"]]
    parties_seen.update(sub["party"].unique())
    ax.scatter(sub["coupling"], np.full(len(sub), yi) + jitter,
               c=cols, s=13, alpha=0.55, linewidths=0, zorder=2)

    # period median reference tick
    med = sub["coupling"].median()
    ax.plot([med, med], [yi - row_h/2, yi + row_h/2],
            color=T["ax"], lw=1.0, ls="--", alpha=0.6, zorder=1)

    # Chancellor
    name, party = CHANCELLOR[period]
    crow = sub[sub["name"] == name]
    if crow.empty:
        continue
    cx  = float(crow["coupling"].iloc[0])
    cpc = float(crow["pct"].iloc[0])
    col = party_color.get(party, "#888888")
    ax.scatter([cx], [yi], s=170, c=col, zorder=5,
               linewidths=1.8, edgecolors=T["bg"])
    lbl = f"{name.split()[-1]}  ·  {ordinal(round(cpc))} pct"
    t = ax.text(cx + xmax * 0.012, yi, lbl, fontsize=10, fontweight="bold",
                color=T["text"], va="center", ha="left", zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=T["bg"])])

ax.set_yticks(range(len(PERIOD_ORDER)))
ax.set_yticklabels(PERIOD_ORDER, color=T["text"], fontsize=11)
ax.set_ylim(-0.6, len(PERIOD_ORDER) - 0.1)
ax.set_xlim(0, xmax)
ax.set_xlabel("Voting coupling strength  —  mean |J_ij| (×10³)  ·  dashed = period median",
              color=T["sub"], fontsize=11)
ax.tick_params(axis="x", colors=T["sub"], labelsize=9, length=0)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=T["grid"], lw=0.5, ls="--", zorder=0)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color(T["ax"])

# Party legend
PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
               "BÜNDNIS 90/DIE GRÜNEN", "Die Linke", "fraktionslos"]
DISPLAY = {"BÜNDNIS 90/DIE GRÜNEN": "Grüne", "fraktionslos": "fraktl."}
handles = [
    plt.Line2D([0], [0], marker="o", color="none",
               markerfacecolor=party_color.get(p, "#888"), markersize=7,
               label=DISPLAY.get(p, p))
    for p in PARTY_ORDER if p in parties_seen
]
ax.legend(handles=handles, loc="upper center", frameon=False,
          fontsize=8.5, labelcolor=T["sub"], ncol=len(handles),
          handlelength=0.6, handletextpad=0.3, columnspacing=1.0,
          bbox_to_anchor=(0.5, -0.10))

ax.set_title("Every chancellor ranks among the least coupled MPs in parliament",
             color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.155,
        "Each dot = one MP, coloured by party  ·  large dot = the chancellor  ·  "
        "coupling = mean |J_ij| from regularised PLM inverse Ising",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "mp_ising_chancellor_strip.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"Saved → {out}")
