"""
mp_umap.py

UMAP embedding of MPs based on their full pairwise κ profile.
Each MP = row vector of κ values with all other MPs in the same period.
2×3 grid, one subplot per Bundestag period.

Usage:
    python analysis/mp_umap.py        # dark
    python analysis/mp_umap.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
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
    "SPD": "SPD", "fraktionslos": "fraktl.",
}
def short(p): return SHORT.get(p, p)

with open(BASE_DIR / "config" / "coalitions.json") as f:
    coalitions_map = json.load(f)

with open(BASE_DIR / "config" / "party_colours.json") as f:
    raw_colors = json.load(f)
party_color = {canon(k): v for k, v in raw_colors.items()}
party_color.setdefault("BSW", "#a020f0")
party_color.setdefault("fraktionslos", "#888888")

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#ffffff" if LIGHT_MODE else "#0d1117"
TEXT    = "#1a1a1a" if LIGHT_MODE else "white"
SUBTEXT = "#666666" if LIGHT_MODE else "#888888"

if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"
    party_color["FDP"]     = "#f5d800"


def build_kappa_matrix(nodes, edges):
    """Build symmetric N×N κ matrix. Missing pairs → 0."""
    ids  = nodes["person_id"].values
    id2i = {pid: i for i, pid in enumerate(ids)}
    N    = len(ids)
    mat  = np.zeros((N, N), dtype=np.float32)

    src = edges["source"].map(id2i)
    tgt = edges["target"].map(id2i)
    w   = edges["weight"].values

    valid = src.notna() & tgt.notna()
    si = src[valid].astype(int).values
    ti = tgt[valid].astype(int).values
    wi = w[valid]

    mat[si, ti] = wi
    mat[ti, si] = wi   # symmetrise
    return mat, ids


fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.patch.set_facecolor(BG)

for ax, (pk, lbl, coal_key) in zip(axes.flat, PERIODS):
    print(f"Processing {lbl}…", flush=True)
    d     = BASE_DIR / "output" / pk
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    edges = pd.read_csv(d / "edges_allpairs.csv")

    mat, ids = build_kappa_matrix(nodes, edges)
    print(f"  Matrix: {mat.shape[0]} MPs, {(mat != 0).sum()//2:,} pairs with data", flush=True)

    # UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(mat)

    # Map back to party labels
    id2party = dict(zip(nodes["person_id"], nodes["party"]))
    parties  = np.array([id2party.get(pid, "fraktionslos") for pid in ids])

    # Plot — one scatter call per party for z-ordering
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    unique_parties = sorted(set(parties))
    for party in unique_parties:
        mask = parties == party
        col  = party_color.get(party, "#888888")
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=6, color=col,
                   alpha=0.7 if LIGHT_MODE else 0.8,
                   linewidths=0, rasterized=True,
                   label=short(party))

    # Period title
    ax.set_title(lbl, color=TEXT, fontsize=15, fontweight="bold",
                 loc="left", pad=10)

    # Coalition annotation
    coal_parties = [canon(p) for p in coalitions_map.get(coal_key, [])]
    if coal_parties:
        CHAR_W = 0.0115; cursor = 0.0
        for i, p in enumerate(coal_parties):
            if i > 0:
                ax.text(cursor, 1.01, " · ", transform=ax.transAxes,
                        color=SUBTEXT, fontsize=8, va="bottom")
                cursor += 3 * CHAR_W
            col  = party_color.get(p, SUBTEXT)
            name = short(p)
            ax.text(cursor, 1.01, name, transform=ax.transAxes, color=col,
                    fontsize=8, fontweight="bold", va="bottom")
            cursor += len(name) * CHAR_W

    # Legend
    handles = [mpatches.Patch(color=party_color.get(p, "#888888"), label=short(p))
               for p in unique_parties]
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=False,
              labelcolor=TEXT if LIGHT_MODE else "white",
              handlelength=1.0, handleheight=0.9,
              borderpad=0.3, labelspacing=0.25)

    print(f"  {lbl}: done", flush=True)

fig.text(0.5, 0.98,
         "UMAP of MPs by voting similarity  ·  all Bundestag periods",
         ha="center", va="top", color=TEXT, fontsize=16, fontweight="bold")
fig.text(0.5, 0.96,
         "Each dot is one MP. Position reflects pairwise Cohen's κ profile across all voting pairs. Color indicates party.",
         ha="center", va="top", color=SUBTEXT, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.955])
plt.subplots_adjust(hspace=0.30, wspace=0.08)

out = IMG_DIR / "mp_umap.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
