"""
mp_ising_bump.py

Bar chart: average PLM coupling percentile rank of notable MPs across all
parliamentary periods they served in.

Usage
-----
  python analysis/ising/mp_ising_bump.py
  python analysis/ising/mp_ising_bump.py light
"""

from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
CACHE    = BASE_DIR / "output" / "mp_ising_longitudinal_cache.csv"

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

# Chancellors, vice chancellors, and Söder (2005–present)
NOTABLE = [
    # Chancellors
    "Angela Merkel",        # 2005–2021
    "Olaf Scholz",          # 2021–2025 (also VC 2018–2021)
    "Friedrich Merz",       # 2025–
    # Vice Chancellors
    "Franz Müntefering",    # 2005–2007
    "Frank-Walter Steinmeier",  # 2007–2009
    "Guido Westerwelle",    # 2009–2011
    "Philipp Rösler",       # 2011–2013
    "Sigmar Gabriel",       # 2013–2018
    "Robert Habeck",        # 2021–2025
    "Lars Klingbeil",       # 2025–
    # CSU leader
    "Markus Söder",
]

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme / "ising"
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", sub="#555555", grid="#dddddd", ax="#cccccc")
else:
    T = dict(bg="#0d1117", text="white", sub="#888888", grid="#1e2530", ax="#333333")

with open(BASE_DIR / "config" / "party_colours.json") as f:
    _raw = json.load(f)
party_color = {canon(k): v for k, v in _raw.items()}
party_color.setdefault("BSW", "#a020f0")
party_color.setdefault("fraktionslos", "#888888")
if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"
    party_color["FDP"]     = "#f5d800"


# ── Load & compute ─────────────────────────────────────────────────────────────
all_df = pd.read_csv(CACHE)
all_df["party"] = all_df["party"].map(canon)

# Percentile rank within each period (100 = highest coupling)
all_df["pct"] = all_df.groupby("period")["coupling"].rank(pct=True) * 100

df_top = all_df[all_df["name"].isin(NOTABLE)].copy()
found  = df_top["name"].unique().tolist()
missing = [n for n in NOTABLE if n not in found]
if missing:
    print(f"Not in data: {missing}")

last_party = (df_top.sort_values("period")
              .groupby("name")["party"].last()
              .to_dict())

# Average percentile + period count per MP
summary = (df_top.groupby("name")
           .agg(mean_pct=("pct", "mean"), n_periods=("period", "count"))
           .reset_index())
summary["party"] = summary["name"].map(last_party)
summary = summary.sort_values("mean_pct", ascending=True)  # lowest at bottom

print("Notable MPs:")
for _, row in summary.iterrows():
    print(f"  {row['name']:30s}  pct={row['mean_pct']:.1f}  n={row['n_periods']}")


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

colors = [party_color.get(p, "#888888") for p in summary["party"]]
bars = ax.barh(summary["name"], summary["mean_pct"],
               color=colors, alpha=0.85, height=0.65, zorder=3)

# Period-count annotation (n=X) right of bar
for bar, val, n in zip(bars, summary["mean_pct"], summary["n_periods"]):
    ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            fontsize=7.5, color=T["sub"], va="center", ha="left", zorder=4)

# Median reference
ax.axvline(50, color=T["ax"], lw=1.0, ls="--", alpha=0.6, zorder=2)
ax.text(50.6, len(summary) - 0.5, "parliament median",
        fontsize=8, color=T["sub"], va="top", ha="left")

ax.set_xlim(0, 108)
ax.set_xlabel(
    "Average coupling percentile across all periods served  (100 = most coupled)",
    color=T["sub"], fontsize=10)
ax.tick_params(axis="y", colors=T["text"], labelsize=9.5, length=0)
ax.tick_params(axis="x", colors=T["sub"], labelsize=9)
ax.xaxis.grid(True, color=T["grid"], lw=0.5, ls="--", zorder=0)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color(T["ax"])

# Party legend
PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
               "BÜNDNIS 90/DIE GRÜNEN", "Die Linke", "fraktionslos"]
DISPLAY = {"BÜNDNIS 90/DIE GRÜNEN": "Grüne", "fraktionslos": "fraktl."}
seen = set(summary["party"].unique())
handles = [
    plt.Line2D([0], [0], marker="s", color="none",
               markerfacecolor=party_color.get(p, "#888"), markersize=9,
               label=DISPLAY.get(p, p))
    for p in PARTY_ORDER if p in seen
]
ax.legend(handles=handles, loc="lower right", frameon=False,
          fontsize=9, labelcolor=T["sub"],
          handlelength=0.6, handletextpad=0.5)

n_total = int(all_df.groupby("period")["name"].count().mean())
ax.set_title("Who drives Bundestag votes?  —  PLM coupling of notable MPs",
             color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.06,
        f"Average coupling percentile across all periods served  ·  ~{n_total} MPs per period  ·  "
        "coupling = mean |J_ij| from regularised inverse Ising (PLM)",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "mp_ising_bump.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")
