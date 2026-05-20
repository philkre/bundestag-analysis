"""
mp_cross_aisle_simple.py

Line chart: mean pairwise κ within coalition, within opposition,
and between coalition and opposition — per parliament period.
The widening gap between within-group and cross-group lines = polarisation.

Usage:
    python mp_cross_aisle_simple.py           # dark mode
    python mp_cross_aisle_simple.py light     # light mode
"""

import json, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]

ALIASES = {
    "DIE GRÜNEN":  "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":   "Die Linke",
    "Die Linke.":  "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

def _period_key(k):
    k = k.lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"_+", "_", k)

# ── Theme ─────────────────────────────────────────────────────────────────────
LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", subtext="#555555",
             grid="#dddddd", zero="#999999")
    C = dict(coal="#2471a3", opp="#c0392b", cross="#888888")
else:
    T = dict(bg="#0d1117", text="white", subtext="#888888",
             grid="#1e2530", zero="#444444")
    C = dict(coal="#4a9fd4", opp="#e05c4a", cross="#888888")

# ── Load coalitions ───────────────────────────────────────────────────────────
with open(BASE_DIR / "config" / "coalitions.json") as f:
    coalitions_raw = json.load(f)
coalitions_map = {_period_key(k): set(v) for k, v in coalitions_raw.items()}

# ── Compute κ stats per period ────────────────────────────────────────────────
records = []

for pk, lbl in PERIODS:
    d = BASE_DIR / "output" / pk
    ep = d / "edges_allpairs.csv"
    ef = ep if ep.exists() else d / "edges.csv"
    if not (d / "nodes.csv").exists():
        continue

    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    edges = pd.read_csv(ef)

    coal_parties = coalitions_map.get(_period_key(pk), set())
    nodes["group"] = nodes["party"].apply(
        lambda p: "coalition" if p in coal_parties else "opposition"
    )
    pid_to_group = dict(zip(nodes["person_id"], nodes["group"]))

    edges = edges.copy()
    edges["g_s"] = edges["source"].map(pid_to_group)
    edges["g_t"] = edges["target"].map(pid_to_group)
    edges = edges.dropna(subset=["g_s", "g_t"])

    # Within coalition
    wc = edges[(edges["g_s"] == "coalition") & (edges["g_t"] == "coalition")]["weight"]
    # Within opposition
    wo = edges[(edges["g_s"] == "opposition") & (edges["g_t"] == "opposition")]["weight"]
    # Cross-aisle
    cx = edges[edges["g_s"] != edges["g_t"]]["weight"]

    records.append(dict(
        period=lbl,
        within_coal=wc.mean(),
        within_opp=wo.mean(),
        cross=cx.mean(),
        n_coal_mps=(nodes["group"] == "coalition").sum(),
        n_opp_mps=(nodes["group"] == "opposition").sum(),
    ))

df = pd.DataFrame(records)
xi = np.arange(len(df))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

lw = 2.2
ms = 7

ax.plot(xi, df["within_coal"],  color=C["coal"],  lw=lw, marker="o", ms=ms,
        label="Within coalition", zorder=4)
ax.plot(xi, df["within_opp"],   color=C["opp"],   lw=lw, marker="o", ms=ms,
        label="Within opposition", zorder=4)
ax.plot(xi, df["cross"],        color=C["cross"],  lw=lw, marker="o", ms=ms,
        linestyle="--", label="Coalition ↔ opposition", zorder=4)

# Fill gap between within-group average and cross-aisle
within_avg = (df["within_coal"] + df["within_opp"]) / 2
ax.fill_between(xi, within_avg, df["cross"],
                color=C["opp"], alpha=0.07, zorder=2)

# Annotate final values
for col, c in [("within_coal", C["coal"]), ("within_opp", C["opp"]), ("cross", C["cross"])]:
    v = df[col].iloc[-1]
    ax.text(xi[-1] + 0.12, v, f"{v:.2f}",
            color=c, fontsize=8, va="center", clip_on=False)

# y = 0 reference
ax.axhline(0, color=T["zero"], lw=0.8, linestyle="--", alpha=0.5, zorder=1)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(xi)
ax.set_xticklabels(df["period"], color=T["text"], fontsize=11, fontweight="bold")
ax.tick_params(axis="x", length=0, pad=8)
ax.tick_params(axis="y", colors=T["subtext"], labelsize=8, length=0)
ax.set_xlim(-0.5, len(df) - 0.3)
ax.set_ylim(-0.25, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

# Horizontal gridlines
for y in np.arange(-0.2, 1.05, 0.2):
    ax.axhline(y, color=T["grid"], lw=0.5, zorder=0)

ax.spines[:].set_visible(False)

# ── Coalition size annotation ─────────────────────────────────────────────────
for i, row in df.iterrows():
    ax.text(i, -0.22,
            f"{row['n_coal_mps']} / {row['n_opp_mps']}",
            ha="center", va="top", fontsize=6.5, color=T["subtext"])
ax.text(-0.45, -0.22, "Coal / Opp MPs:", ha="left", va="top",
        fontsize=6.5, color=T["subtext"], clip_on=False)

# ── Legend ────────────────────────────────────────────────────────────────────
leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right",
                labelcolor=[C["coal"], C["opp"], C["cross"]])
for t in leg.get_texts():
    t.set_color(T["subtext"])

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(0, 1.13, "Coalition vs. opposition voting cohesion, Bundestag 2005–2029",
        transform=ax.transAxes, color=T["text"], fontsize=15,
        fontweight="bold", va="top", ha="left", clip_on=False)
ax.text(0, 1.065, "Mean pairwise Cohen's κ  ·  Higher = voted more similarly  ·  "
        "Gap between within-group and cross-aisle lines = polarisation",
        transform=ax.transAxes, color=T["subtext"], fontsize=8.5,
        va="top", ha="left", clip_on=False)

fig.subplots_adjust(left=0.06, right=0.93, top=0.83, bottom=0.12)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = IMG_DIR / f"mp_cross_aisle_simple.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2,
            facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
