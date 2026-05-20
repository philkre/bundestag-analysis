"""
mp_influence.py

Computes MP-level influence on the chamber majority vote outcome,
inspired by Lee et al. 2017 "Statistical mechanics of the US Supreme Court"
(their Fig 4C and Table I).

Method
──────
For each MP i and each poll v where σ_i is defined:
  - σ_i ∈ {+1, −1}  (yes / no; abstain / no_show → NaN)
  - γ^v = sign(Σ σ_j)  majority vote direction
  - I(σ_i; γ) = mutual information in bits (2×2 joint table)
  - c_i = ⟨σ_i γ⟩   Pearson correlation (simpler proxy, x-axis)

Usage:
    python mp_influence.py           # dark mode
    python mp_influence.py light     # light mode
"""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text
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

VOTE_MAP = {"yes": 1.0, "no": -1.0}  # abstain / no_show → NaN

# ── Party colours ──────────────────────────────────────────────────────────────
LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_DIR / "config" / "party_colours.json") as f:
    party_color: dict = json.load(f)

# Normalise aliases in colour dict too
for alias, canonical in ALIASES.items():
    if alias in party_color and canonical not in party_color:
        party_color[canonical] = party_color[alias]

party_color.setdefault("fraktionslos", "#888888")
party_color.setdefault("BSW", "#a020f0")

if LIGHT_MODE:
    party_color["CDU/CSU"] = "#3a3a3a"
    party_color["FDP"]     = "#f0c000"

# ── Theme ─────────────────────────────────────────────────────────────────────
if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", subtext="#555555")
else:
    T = dict(bg="#0d1117", text="white",   subtext="#888888")


# ══════════════════════════════════════════════════════════════════════════════
# Core computation
# ══════════════════════════════════════════════════════════════════════════════

def _safe_mi(p11, p1n1, pn11, pn1n1):
    """
    Mutual information in bits for a 2×2 table.
    Cell entries are probabilities (sum to 1).
    Returns I = Σ p_xy log2(p_xy / p_x p_y).
    Cells with p_xy = 0 contribute 0 (0 log 0 = 0 convention).
    """
    table = np.array([p11, p1n1, pn11, pn1n1])
    # marginals
    p_sigma_pos = p11 + p1n1   # P(σ=+1)
    p_sigma_neg = pn11 + pn1n1  # P(σ=−1)
    p_gam_pos   = p11 + pn11   # P(γ=+1)
    p_gam_neg   = p1n1 + pn1n1  # P(γ=−1)

    marginals_sigma = np.array([p_sigma_pos, p_sigma_pos, p_sigma_neg, p_sigma_neg])
    marginals_gamma = np.array([p_gam_pos,   p_gam_neg,   p_gam_pos,  p_gam_neg])

    mi = 0.0
    for p_xy, p_x, p_y in zip(table, marginals_sigma, marginals_gamma):
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi


def compute_period(period_key):
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)

    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)

    # ── Build vote matrix S: MP × poll ────────────────────────────────────────
    poll_ids = [p["id"] for p in raw["polls"]]
    poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
    mp_ids   = nodes["person_id"].tolist()
    mp_idx   = {mid: i for i, mid in enumerate(mp_ids)}

    n_mp, n_poll = len(mp_ids), len(poll_ids)
    S = np.full((n_mp, n_poll), np.nan, dtype=np.float32)

    for v in raw["votes"]:
        val = VOTE_MAP.get(v["vote"])
        if val is None:
            continue
        mi = mp_idx.get(v["mandate"]["id"])
        pi = poll_idx.get(v["poll"]["id"])
        if mi is not None and pi is not None:
            S[mi, pi] = val

    # ── Filter near-unanimous polls ────────────────────────────────────────────
    yes_frac = np.nanmean(S == 1, axis=0)
    keep     = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S        = S[:, keep]
    n_poll_kept = keep.sum()
    print(f"  polls kept: {n_poll_kept} / {n_poll}")

    # ── Drop MPs with < 10 % participation in kept polls ──────────────────────
    participation = (~np.isnan(S)).sum(axis=1)
    min_votes     = max(3, 0.10 * n_poll_kept)
    active        = participation >= min_votes
    S             = S[active]
    nodes         = nodes[active].reset_index(drop=True)
    n_mp_active   = nodes.shape[0]
    print(f"  active MPs: {n_mp_active}")

    # ── Majority vote γ^v = sign(Σ σ) per poll ────────────────────────────────
    col_sums = np.nansum(S, axis=0)           # (n_poll_kept,)
    gamma    = np.sign(col_sums).astype(np.float32)   # +1 or −1 (0 if all NaN, rare)
    # Polls with exact ties (col_sum=0) are uninformative; treat as NaN for MI
    gamma[gamma == 0] = np.nan

    # ── Per-MP mutual information I(σ_i; γ) and correlation c_i ──────────────
    results = []
    for i in range(n_mp_active):
        sigma = S[i]                          # (n_poll_kept,)

        # Valid polls: σ defined AND γ defined
        valid = (~np.isnan(sigma)) & (~np.isnan(gamma))
        n_valid = valid.sum()

        if n_valid < 10:
            continue

        s = sigma[valid]   # ±1
        g = gamma[valid]   # ±1

        # 2×2 joint counts → probabilities
        n11  = np.sum((s ==  1) & (g ==  1))
        n1n1 = np.sum((s ==  1) & (g == -1))
        nn11 = np.sum((s == -1) & (g ==  1))
        nn1n1= np.sum((s == -1) & (g == -1))
        total = n11 + n1n1 + nn11 + nn1n1   # == n_valid

        p11   = n11   / total
        p1n1  = n1n1  / total
        pn11  = nn11  / total
        pn1n1 = nn1n1 / total

        mi = _safe_mi(p11, p1n1, pn11, pn1n1)

        # Correlation c_i = ⟨σ_i γ⟩
        c_i = float(np.mean(s * g))

        row = nodes.iloc[i]
        results.append({
            "person_id":    row["person_id"],
            "name":         row["name"],
            "party":        row["party"],
            "mi":           mi,
            "corr":         c_i,
            "n_valid":      n_valid,
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Main loop + stdout report
# ══════════════════════════════════════════════════════════════════════════════

period_dfs = []
for pk, lbl in PERIODS:
    print(f"\n{lbl}")
    try:
        df = compute_period(pk)
        period_dfs.append((lbl, df))

        top10 = df.nlargest(10, "mi")
        print(f"  Top 10 by mutual information I(σ_i; γ):")
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"    {rank:2d}. {row['name']:<30s}  {row['party']:<30s}  "
                  f"I = {row['mi']:.4f} bits  c = {row['corr']:+.3f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        period_dfs.append((lbl, pd.DataFrame()))
        print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation: 2×3 grid, one panel per period
# ══════════════════════════════════════════════════════════════════════════════

DISPLAY_NAME = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne", "CDU/CSU": "CDU/CSU",
    "Die Linke": "Linke", "BSW": "BSW", "AfD": "AfD",
    "FDP": "FDP", "SPD": "SPD", "fraktionslos": "fraktl.",
}

CHANCELLORS = {
    "2005–09": "Angela Merkel",
    "2009–13": "Angela Merkel",
    "2013–17": "Angela Merkel",
    "2017–21": "Angela Merkel",
    "2021–25": "Olaf Scholz",
    "2025–29": "Friedrich Merz",
}

# Fraktionsvorsitzende per period (co-chairs listed where applicable)
FRAKTIONSVORSITZENDE = {
    "2005–09": [
        "Volker Kauder",           # CDU/CSU
        "Peter Struck",            # SPD
        "Guido Westerwelle",       # FDP
        "Renate Künast",           # Grüne
        "Fritz Kuhn",              # Grüne
        "Gregor Gysi",             # Linke
        "Oskar Lafontaine",        # Linke
    ],
    "2009–13": [
        "Volker Kauder",           # CDU/CSU
        "Frank-Walter Steinmeier", # SPD
        "Birgit Homburger",        # FDP (until 2011)
        "Rainer Brüderle",         # FDP (from 2011)
        "Renate Künast",           # Grüne
        "Jürgen Trittin",          # Grüne
        "Gregor Gysi",             # Linke
    ],
    "2013–17": [
        "Volker Kauder",           # CDU/CSU
        "Thomas Oppermann",        # SPD
        "Katrin Göring-Eckardt",   # Grüne
        "Anton Hofreiter",         # Grüne
        "Gregor Gysi",             # Linke (until 2015)
        "Sahra Wagenknecht",       # Linke (from 2015)
        "Dietmar Bartsch",         # Linke (from 2015)
    ],
    "2017–21": [
        "Volker Kauder",           # CDU/CSU (until 2018)
        "Ralph Brinkhaus",         # CDU/CSU (from 2018)
        "Andrea Nahles",           # SPD (until 2019)
        "Rolf Mützenich",          # SPD (from 2019)
        "Alexander Gauland",       # AfD
        "Alice Weidel",            # AfD
        "Christian Lindner",       # FDP
        "Katrin Göring-Eckardt",   # Grüne
        "Anton Hofreiter",         # Grüne
        "Sahra Wagenknecht",       # Linke (until 2019)
        "Dietmar Bartsch",         # Linke
        "Amira Mohamed Ali",       # Linke (from 2019)
    ],
    "2021–25": [
        "Rolf Mützenich",          # SPD
        "Katharina Dröge",         # Grüne
        "Britta Haßelmann",        # Grüne
        "Christian Dürr",          # FDP
        "Friedrich Merz",          # CDU/CSU (from 2022)
        "Ralph Brinkhaus",         # CDU/CSU (until 2022)
        "Alice Weidel",            # AfD
        "Tino Chrupalla",          # AfD
        "Amira Mohamed Ali",       # Linke/BSW
        "Dietmar Bartsch",         # Linke
        "Heidi Reichinnek",        # Linke (from 2023)
    ],
    "2025–29": [
        "Jens Spahn",              # CDU/CSU
        "Rolf Mützenich",          # SPD
        "Katharina Dröge",         # Grüne
        "Britta Haßelmann",        # Grüne
        "Alice Weidel",            # AfD
        "Tino Chrupalla",          # AfD
        "Heidi Reichinnek",        # Linke
    ],
}

fig, axes = plt.subplots(2, 3, figsize=(26, 18),
                         sharex=True, sharey=True)
fig.patch.set_facecolor(T["bg"])

grid_color = "#1e2530" if not LIGHT_MODE else "#dddddd"
zero_color = "#444444" if not LIGHT_MODE else "#aaaaaa"

# Collect all parties for shared legend
all_parties_seen = {}   # party -> color

for idx, (ax, (lbl, df)) in enumerate(zip(axes.flat, period_dfs)):
    ax_row, ax_col = divmod(idx, 3)
    ax.set_facecolor(T["bg"])
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=T["subtext"], labelsize=11, length=0)

    # Period label as proper subplot title — sits above the frame, never inside data
    ax.set_title(lbl, color=T["text"], fontsize=16, fontweight="bold",
                 loc="left", pad=8)

    if df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                color=T["subtext"], ha="center", va="center")
        continue

    # Scatter
    for party, grp in df.groupby("party"):
        color = party_color.get(party, "#888888")
        all_parties_seen[party] = color
        ax.scatter(grp["corr"], grp["mi"],
                   c=color, s=18, alpha=0.70,
                   linewidths=0, zorder=3, label=party)

    # ── Chancellor + Fraktionsvorsitzende highlights ──────────────────────────
    hl_color  = "#111111" if LIGHT_MODE else "white"
    hl_stroke = T["bg"]

    hl_texts = []   # Text objects for adjustText
    hl_xs, hl_ys = [], []

    # Chancellor
    chancellor_name = CHANCELLORS.get(lbl)
    if chancellor_name:
        ch_row = df[df["name"] == chancellor_name]
        if not ch_row.empty:
            cx, cy = float(ch_row["corr"].iloc[0]), float(ch_row["mi"].iloc[0])
            ch_color = party_color.get(ch_row["party"].iloc[0], "#888888")
            ax.scatter(cx, cy, s=130, facecolors="none",
                       edgecolors=hl_color, linewidths=1.8, zorder=6)
            ax.scatter(cx, cy, s=60, c=ch_color, linewidths=0, zorder=7)
            t = ax.text(cx, cy, chancellor_name,
                        fontsize=11, fontweight="bold", color=hl_color,
                        va="center", ha="center", clip_on=False, zorder=8)
            hl_texts.append(t)
            hl_xs.append(cx); hl_ys.append(cy)

    # Fraktionsvorsitzende
    for fv_name in FRAKTIONSVORSITZENDE.get(lbl, []):
        if fv_name == chancellor_name:
            continue
        fv_row = df[df["name"] == fv_name]
        if fv_row.empty:
            continue
        fx, fy = float(fv_row["corr"].iloc[0]), float(fv_row["mi"].iloc[0])
        fv_color = party_color.get(fv_row["party"].iloc[0], "#888888")
        ax.scatter(fx, fy, s=75, facecolors="none",
                   edgecolors=hl_color, linewidths=1.2, zorder=6, alpha=0.85)
        ax.scatter(fx, fy, s=35, c=fv_color, linewidths=0, zorder=7)
        t = ax.text(fx, fy, fv_name.split()[-1],
                    fontsize=10, fontweight="normal", color=hl_color,
                    va="center", ha="center", clip_on=False, zorder=8)
        hl_texts.append(t)
        hl_xs.append(fx); hl_ys.append(fy)

    # Let adjustText find non-overlapping positions and draw connectors
    if hl_texts:
        adjust_text(
            hl_texts,
            x=np.array(hl_xs), y=np.array(hl_ys),
            ax=ax,
            expand=(1.4, 1.6),
            force_text=(0.4, 0.6),
            force_points=(0.3, 0.5),
            arrowprops=dict(arrowstyle="-", color=hl_color, lw=0.6, alpha=0.75),
        )
        for t in hl_texts:
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=hl_stroke)])

    # Axis labels only on border panels
    if ax_row == 1:
        ax.set_xlabel("← Consistent opposition  —  Consistent coalition →",
                      color=T["subtext"], fontsize=13)
    if ax_col == 0:
        ax.set_ylabel("← Less predictive  —  More predictive →",
                      color=T["subtext"], fontsize=13)

    ax.grid(True, color=grid_color, lw=0.4, zorder=0)
    ax.axvline(0, color=zero_color, lw=0.7, ls="--", alpha=0.5, zorder=1)

# ── Shared legend ─────────────────────────────────────────────────────────────
PARTY_ORDER_DISP = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
                    "BÜNDNIS 90/DIE GRÜNEN", "Die Linke", "fraktionslos"]
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="none",
               markerfacecolor=all_parties_seen.get(p, party_color.get(p, "#888")),
               markersize=6,
               label=DISPLAY_NAME.get(p, p))
    for p in PARTY_ORDER_DISP if p in all_parties_seen
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 0.018), ncol=len(legend_handles),
           frameon=False, fontsize=12,
           handlelength=0.7, handletextpad=0.4, columnspacing=1.4,
           labelcolor=T["subtext"])

# ── Title + subtitle ──────────────────────────────────────────────────────────
fig.text(0.5, 0.997,
         "How predictable was each MP's vote? Bundestag 2005–2029",
         ha="center", va="top",
         color=T["text"], fontsize=21, fontweight="bold")
fig.text(0.5, 0.968,
         "Each dot is one MP, coloured by party  ·  MPs at the top voted more predictably"
         "  ·  chancellor (bold ring) and Fraktionsvorsitzende (thin ring) highlighted",
         ha="center", va="top",
         color=T["subtext"], fontsize=13)

fig.subplots_adjust(left=0.07, right=0.995, top=0.925, bottom=0.055,
                    wspace=0.06, hspace=0.22)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = IMG_DIR / f"mp_influence.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15,
            facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
