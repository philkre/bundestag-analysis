"""
mp_ising_chancellor_coupling.py

Coupling strength of every chancellor while in office, one bar per term.
Coupling shown as percentile rank within that period's parliament
(100 = most coupled MP, 0 = least).

Reads cache from mp_ising_chancellor_path.py / mp_ising_field computation.

Usage
-----
  python analysis/ising/mp_ising_chancellor_coupling.py
  python analysis/ising/mp_ising_chancellor_coupling.py light
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
CACHE    = BASE_DIR / "output" / "mp_ising_field_cache.csv"

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

# Chancellor terms within the data window
CHANCELLOR_TERMS = [
    ("Angela Merkel",  "CDU/CSU", "2005–09"),
    ("Angela Merkel",  "CDU/CSU", "2009–13"),
    ("Angela Merkel",  "CDU/CSU", "2013–17"),
    ("Angela Merkel",  "CDU/CSU", "2017–21"),
    ("Olaf Scholz",    "SPD",     "2021–25"),
    ("Friedrich Merz", "CDU/CSU", "2025–29"),
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
if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
else:
    party_color["CDU/CSU"] = "#dddddd"

if not CACHE.exists():
    sys.exit(f"Cache missing: {CACHE}\nRun mp_ising_chancellor_path.py first.")

all_df = pd.read_csv(CACHE)
all_df["party"] = all_df["party"].map(canon)
# Coupling as percent of the strongest coupler that period
all_df["pct"] = all_df.groupby("period")["coupling"].transform(
    lambda x: x / x.max() * 100)

# Build rows for each chancellor term
rows = []
for name, party, period in CHANCELLOR_TERMS:
    sub = all_df[(all_df["name"] == name) & (all_df["period"] == period)]
    if sub.empty:
        print(f"  missing: {name} {period}")
        continue
    rows.append({
        "label": f"{name.split()[-1]}\n{period}",
        "name": name, "party": party, "period": period,
        "pct": float(sub["pct"].iloc[0]),
        "coupling": float(sub["coupling"].iloc[0]),
    })
plot_df = pd.DataFrame(rows)

print("Chancellor coupling (percentile within period):")
for _, r in plot_df.iterrows():
    print(f"  {r['name']:16s} {r['period']}  pct={r['pct']:5.1f}  coupling={r['coupling']:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

x = np.arange(len(plot_df))
colors = [party_color.get(p, "#888888") for p in plot_df["party"]]
bars = ax.bar(x, plot_df["pct"], color=colors, alpha=0.88, width=0.62, zorder=3)

# value labels on top
for bar, val in zip(bars, plot_df["pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=9, color=T["text"], zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["label"], color=T["text"], fontsize=10)
ax.set_ylim(0, 100)
ax.set_ylabel("Coupling as % of the strongest coupler that period",
              color=T["sub"], fontsize=11)
ax.tick_params(axis="y", colors=T["sub"], labelsize=9, length=0)
ax.tick_params(axis="x", length=0)
ax.yaxis.grid(True, color=T["grid"], lw=0.5, ls="--", zorder=0)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color(T["ax"])

ax.set_title("Coupling strength of German chancellors while in office",
             color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.16,
        "Each bar = one chancellor term  ·  coupling = mean |J_ij| from regularised PLM inverse Ising  ·  "
        "as % of the most-coupled MP that period",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "mp_ising_chancellor_coupling.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")
