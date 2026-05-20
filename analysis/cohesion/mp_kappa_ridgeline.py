"""
mp_kappa_ridgeline.py

Ridgeline (joy plot) of Cohen's κ distributions per party pair,
one figure per Bundestag period.

Usage:
    python analysis/mp_kappa_ridgeline.py        # all periods, dark
    python analysis/mp_kappa_ridgeline.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.stats import gaussian_kde
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
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

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
BG      = "#ffffff" if LIGHT_MODE else "#0d1117"
TEXT    = "#1a1a1a" if LIGHT_MODE else "white"
SUBTEXT = "#555555" if LIGHT_MODE else "#888888"
GRID    = "#dddddd" if LIGHT_MODE else "#1a2030"

with open(BASE_DIR / "config" / "party_colours.json") as f:
    raw_colors = json.load(f)
party_color = {canon(k): v for k, v in raw_colors.items()}
party_color.setdefault("fraktionslos", "#888888")
party_color.setdefault("BSW", "#a020f0")
if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"
    party_color["FDP"]     = "#f5d800"


def hex_blend(ca, cb, t=0.5):
    def h2r(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])
    m = h2r(ca) * (1 - t) + h2r(cb) * t
    return "#{:02x}{:02x}{:02x}".format(*[int(v * 255) for v in m])


def gradient_fill(ax, x, y_base, yk, col_a, col_b, alpha=0.28, zorder=2):
    """Fill beneath a KDE with a horizontal gradient from col_a → col_b."""
    verts = ([(x[0], y_base)] +
             list(zip(x, y_base + yk)) +
             [(x[-1], y_base), (x[0], y_base)])
    codes = ([MPath.MOVETO] + [MPath.LINETO] * len(x) +
             [MPath.LINETO, MPath.CLOSEPOLY])
    path  = MPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor="none", zorder=zorder)
    ax.add_patch(patch)

    cmap = LinearSegmentedColormap.from_list("pair", [col_a, col_b])
    y_top = y_base + yk.max()
    img = ax.imshow(
        np.linspace(0, 1, 256).reshape(1, -1),
        extent=[x[0], x[-1], y_base, y_top + 0.01],
        aspect="auto", cmap=cmap, alpha=alpha,
        origin="lower", zorder=zorder,
        clip_path=patch, clip_on=True,
    )
    return patch, img


def render_period(pk, lbl):
    d = BASE_DIR / "output" / pk
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    edges = pd.read_csv(d / "edges_allpairs.csv")
    pid2party = dict(zip(nodes["person_id"], nodes["party"]))
    edges["pa"] = edges["source"].map(pid2party)
    edges["pb"] = edges["target"].map(pid2party)

    # Canonical pair key
    edges = edges.dropna(subset=["pa", "pb"])
    edges["key"] = [
        (a, b) if a <= b else (b, a)
        for a, b in zip(edges["pa"], edges["pb"])
    ]

    MIN_EDGES = 200
    pairs = []
    for key, grp in edges.groupby("key"):
        if len(grp) < MIN_EDGES:
            continue
        pa, pb = key
        if pa == "fraktionslos" or pb == "fraktionslos":
            continue
        pairs.append(dict(
            pa=pa, pb=pb,
            intra=(pa == pb),
            weights=grp["weight"].values,
            mean=grp["weight"].mean(),
            n=len(grp),
        ))

    pairs.sort(key=lambda p: -p["mean"])
    n_pairs = len(pairs)
    if n_pairs == 0:
        print(f"  {lbl}: no pairs"); return

    # ── Layout ────────────────────────────────────────────────────────────────
    ROW_H     = 0.52
    RIDGE_MAX = 0.88     # max ridge height in y-units
    OVERLAP   = 0.06     # ridges may poke slightly into row above

    fig_h = max(7, n_pairs * ROW_H + 2.8)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x   = np.linspace(-1.0, 1.0, 512)
    gap = 1.0   # y-spacing between rows

    # Find split between intra and cross sections
    n_intra = sum(1 for p in pairs if p["intra"])

    for rank, p in enumerate(pairs):
        y_base = (n_pairs - 1 - rank) * gap
        w      = p["weights"]
        col_a  = party_color.get(p["pa"], "#888888")
        col_b  = party_color.get(p["pb"], "#888888")
        is_intra = p["intra"]
        col_mid = col_a if is_intra else hex_blend(col_a, col_b)

        # KDE
        try:
            bw  = max(0.05, 1.06 * w.std() * len(w) ** (-1/5))
            kde = gaussian_kde(w, bw_method=bw / (w.std() + 1e-9))
            yk  = kde(x)
        except Exception:
            continue
        yk = yk / yk.max() * (RIDGE_MAX + OVERLAP)

        # ── Clip out-of-range region for cleanliness ──────────────────────────
        yk = np.clip(yk, 0, None)

        # ── Background mask (hides row below) ────────────────────────────────
        ax.fill_between(x, y_base, y_base + yk, color=BG, zorder=rank * 4 + 1)

        # ── Gradient fill or solid fill ────────────────────────────────────────
        if is_intra:
            ax.fill_between(x, y_base, y_base + yk,
                            color=col_a,
                            alpha=0.30 if LIGHT_MODE else 0.28,
                            zorder=rank * 4 + 2)
        else:
            gradient_fill(ax, x, y_base, yk, col_a, col_b,
                          alpha=0.35 if LIGHT_MODE else 0.28,
                          zorder=rank * 4 + 2)

        # ── KDE outline ───────────────────────────────────────────────────────
        lw   = 1.8 if is_intra else 1.3
        alph = 0.95 if is_intra else 0.80
        ax.plot(x, y_base + yk, color=col_mid, lw=lw, alpha=alph,
                zorder=rank * 4 + 3)

        # ── Baseline ──────────────────────────────────────────────────────────
        ax.plot([-1.0, 1.0], [y_base, y_base],
                color=GRID, lw=0.5, alpha=0.7, zorder=rank * 4)

        # ── Mean tick ────────────────────────────────────────────────────────
        tick_h = RIDGE_MAX * 0.55
        ax.plot([p["mean"], p["mean"]], [y_base, y_base + tick_h],
                color=col_mid, lw=1.2, alpha=0.75, zorder=rank * 4 + 4)

        # ── Label: two-tone for cross-party ───────────────────────────────────
        # t1 is right-aligned at LBL_A; t2 is left-aligned just after it.
        LBL_A = -1.06   # right edge of first label
        LBL_B = -1.04   # left edge of second label (fixed gap)
        y_lbl = y_base + RIDGE_MAX * 0.28
        if is_intra:
            ax.text(LBL_A, y_lbl, short(p["pa"]),
                    color=col_a, fontsize=8, fontweight="bold",
                    ha="right", va="center", zorder=rank * 4 + 4)
        else:
            sa, sb = short(p["pa"]), short(p["pb"])
            ax.text(LBL_A, y_lbl, sa,
                    color=col_a, fontsize=7.5, fontweight="normal",
                    ha="right", va="center", zorder=rank * 4 + 4)
            ax.text(LBL_B, y_lbl, f"· {sb}",
                    color=col_b, fontsize=7.5, fontweight="normal",
                    ha="left", va="center", zorder=rank * 4 + 4)

        # ── Mean value (right margin) ─────────────────────────────────────────
        ax.text(1.04, y_lbl, f"{p['mean']:+.2f}",
                color=SUBTEXT, fontsize=7, ha="left", va="center",
                zorder=rank * 4 + 4)

    # ── Separator between intra and cross sections ───────────────────────────
    if n_intra > 0 and n_intra < n_pairs:
        sep_y = (n_pairs - n_intra) * gap - 0.18
        ax.axhline(sep_y, color=SUBTEXT, lw=0.6, alpha=0.35,
                   ls="--", zorder=1000)
        ax.text(0, sep_y + 0.08, "← intra-party above",
                color=SUBTEXT, fontsize=6.5, ha="center", va="bottom",
                zorder=1001)

    # ── κ = 0 reference ───────────────────────────────────────────────────────
    ax.axvline(0, color=SUBTEXT, lw=0.9, alpha=0.40, ls="--", zorder=0)

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(-1.52, 1.18)
    ax.set_ylim(-0.6, n_pairs * gap + 0.4)
    ax.set_xticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=8)
    ax.yaxis.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    ax.set_xlabel(
        "Cohen's κ  ·  −1 = always opposed  ·  0 = independent  ·  +1 = always agreed",
        color=SUBTEXT, fontsize=8.5, labelpad=8,
    )

    ax.set_title(
        f"Bundestag {lbl}  ·  κ distribution by party pair  ({n_pairs} pairs shown)\n"
        "sorted by mean κ ↓  ·  tick = mean  ·  bold = intra-party  ·  gradient = cross-party",
        color=TEXT, fontsize=11, fontweight="bold", loc="left", pad=14,
    )

    plt.tight_layout(pad=1.5)

    IMG_DIR = BASE_DIR / "output" / "img" / _theme
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_img    = IMG_DIR / f"{pk}_kappa_dist.png"
    out_period = d / f"kappa_dist_{_theme}.png"

    for out in [out_img, out_period]:
        plt.savefig(out, dpi=200, bbox_inches="tight",
                    pad_inches=0.3, facecolor=BG)
    plt.close()
    print(f"  {lbl}: {n_pairs} pairs → {out_img.name}")


for pk, lbl in PERIODS:
    print(f"Processing {lbl}…")
    render_period(pk, lbl)
print("Done.")
