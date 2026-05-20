"""
mp_kappa_hist_govoppo.py

Stacked histogram of government–opposition κ values, broken by party.
Generates three variants:
  (A) colored by opposition party
  (B) colored by coalition party
  (C) colored by party pair (blended color)

Usage:
    python analysis/mp_kappa_hist_govoppo.py        # dark
    python analysis/mp_kappa_hist_govoppo.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
GRID    = "#e0e0e0" if LIGHT_MODE else "#1a2030"

if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"
    party_color["FDP"]     = "#f5d800"

BINS = np.linspace(-1, 1, 51)   # 50 bins
BW   = BINS[1] - BINS[0]
CTRS = (BINS[:-1] + BINS[1:]) / 2


def hex_blend(ca, cb, t=0.5):
    def h2r(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])
    m = h2r(ca) * (1 - t) + h2r(cb) * t
    return "#{:02x}{:02x}{:02x}".format(*[int(v * 255) for v in m])


def load_period(pk, coal_key):
    d = BASE_DIR / "output" / pk
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
    # Normalise so coal_party / opp_party are consistent columns
    cross["coal_party"] = cross.apply(
        lambda r: r["pa"] if r["pa_coal"] else r["pb"], axis=1)
    cross["opp_party"]  = cross.apply(
        lambda r: r["pb"] if r["pa_coal"] else r["pa"], axis=1)
    return cross, coal_set


def stacked_hist(ax, df, group_col, lbl, coal_key, title_suffix):
    """Draw a stacked bar histogram coloured by group_col."""
    groups = df[group_col].unique()
    # Sort by mean κ so legend order matches visual stack order
    means  = {g: df[df[group_col] == g]["weight"].mean() for g in groups}
    groups = sorted(groups, key=lambda g: means[g])

    bottom = np.zeros(len(CTRS))
    handles = []
    for g in groups:
        sub = df[df[group_col] == g]["weight"].values
        counts, _ = np.histogram(sub, bins=BINS)
        col = party_color.get(g, "#888888")
        ax.bar(CTRS, counts, width=BW * 0.92,
               bottom=bottom, color=col,
               alpha=0.85 if LIGHT_MODE else 0.80,
               linewidth=0)
        bottom += counts
        handles.append(mpatches.Patch(color=col, label=short(g)))

    ax.axvline(0, color=SUBTEXT, lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=7.5)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.4, alpha=0.8)
    ax.set_xlabel("Cohen's κ", color=SUBTEXT, fontsize=8.5, labelpad=5)

    # Period title
    ax.text(0.0, 1.13, lbl, transform=ax.transAxes, color=TEXT,
            fontsize=14, fontweight="bold", va="bottom", ha="left")

    # Coalition labels
    coal_parties = [canon(p) for p in coalitions_map.get(coal_key, [])]
    CHAR_W = 0.0115
    cursor = 0.0
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

    # Legend inside panel (top-left)
    ax.legend(handles=handles[::-1], loc="upper left",
              fontsize=6.5, frameon=False,
              labelcolor=TEXT if LIGHT_MODE else "white",
              handlelength=1.0, handleheight=0.9,
              borderpad=0.3, labelspacing=0.25)


def make_figure(variant, suffix, supertitle, subtitle):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor(BG)

    for ax, (pk, lbl, coal_key) in zip(axes.flat, PERIODS):
        cross, coal_set = load_period(pk, coal_key)
        stacked_hist(ax, cross, variant, lbl, coal_key, suffix)

    fig.text(0.5, 0.985, supertitle,
             ha="center", va="top", color=TEXT, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.960, subtitle,
             ha="center", va="top", color=SUBTEXT, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.subplots_adjust(hspace=0.42, wspace=0.12)

    out = IMG_DIR / f"kappa_hist_govoppo_{suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved → {out}")


# ── Variant A: colored by opposition party ────────────────────────────────────
make_figure(
    "opp_party", "by_opp",
    "Government–opposition κ  ·  colored by opposition party",
    "Stacked histogram of pairwise Cohen's κ for all government–opposition MP pairs.  Each segment = one opposition party.",
)

# ── Variant B: colored by coalition party ─────────────────────────────────────
make_figure(
    "coal_party", "by_coal",
    "Government–opposition κ  ·  colored by coalition party",
    "Stacked histogram of pairwise Cohen's κ for all government–opposition MP pairs.  Each segment = one coalition party.",
)

# ── Variant C: colored by party pair (blended) ───────────────────────────────
# Pre-compute blend colors per unique pair, then assign
def make_pair_variant():
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor(BG)

    for ax, (pk, lbl, coal_key) in zip(axes.flat, PERIODS):
        cross, coal_set = load_period(pk, coal_key)
        cross["pair_label"] = cross["coal_party"] + " × " + cross["opp_party"]

        # Build blended color map for each unique pair
        pair_colors = {}
        for _, row in cross[["coal_party", "opp_party", "pair_label"]].drop_duplicates().iterrows():
            ca = party_color.get(row["coal_party"], "#888888")
            cb = party_color.get(row["opp_party"],  "#888888")
            pair_colors[row["pair_label"]] = hex_blend(ca, cb, 0.5)

        pairs = cross["pair_label"].unique()
        means = {p: cross[cross["pair_label"] == p]["weight"].mean() for p in pairs}
        pairs = sorted(pairs, key=lambda p: means[p])

        bottom = np.zeros(len(CTRS))
        handles = []
        for pair in pairs:
            sub = cross[cross["pair_label"] == pair]["weight"].values
            counts, _ = np.histogram(sub, bins=BINS)
            col = pair_colors[pair]
            # Short label: "CDU/CSU × AfD" → "CDU × AfD"
            parts = pair.split(" × ")
            slabel = short(parts[0]) + " × " + short(parts[1])
            ax.bar(CTRS, counts, width=BW * 0.92,
                   bottom=bottom, color=col,
                   alpha=0.85 if LIGHT_MODE else 0.80,
                   linewidth=0)
            bottom += counts
            handles.append(mpatches.Patch(color=col, label=slabel))

        ax.axvline(0, color=SUBTEXT, lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-1, 1)
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(axis="x", colors=SUBTEXT, labelsize=7.5)
        ax.tick_params(axis="y", colors=SUBTEXT, labelsize=6.5)
        ax.grid(axis="x", color=GRID, lw=0.4, alpha=0.8)
        ax.set_xlabel("Cohen's κ", color=SUBTEXT, fontsize=8.5, labelpad=5)

        ax.text(0.0, 1.13, lbl, transform=ax.transAxes, color=TEXT,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
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

        ax.legend(handles=handles[::-1], loc="upper left",
                  fontsize=6, frameon=False,
                  labelcolor=TEXT if LIGHT_MODE else "white",
                  handlelength=1.0, handleheight=0.9,
                  borderpad=0.3, labelspacing=0.2)

    fig.text(0.5, 0.985,
             "Government–opposition κ  ·  colored by party pair",
             ha="center", va="top", color=TEXT, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.960,
             "Stacked histogram of pairwise Cohen's κ for all government–opposition MP pairs.  "
             "Each segment = one coalition × opposition party pair (blended color).",
             ha="center", va="top", color=SUBTEXT, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.subplots_adjust(hspace=0.42, wspace=0.12)

    out = IMG_DIR / "kappa_hist_govoppo_by_pair.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved → {out}")

make_pair_variant()
print("Done.")
