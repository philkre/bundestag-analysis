"""
mp_ising_field.py

2D scatter: per-MP external field strength |h_i| vs PLM coupling strength.

  X  |h_i|                      personal field — bias to vote yes/no regardless of peers
  Y  mean_j |J_ij|              average PLM coupling to all other MPs

Two orthogonal channels of influence:
  high |h|, low J : unconditional conviction / institutional determinism (chancellors)
  low |h|, high J : conditional influence — vote depends on & predicts peers (rebels, bloc soldiers)

Usage
-----
  python analysis/ising/mp_ising_field.py
  python analysis/ising/mp_ising_field.py light
"""

from __future__ import annotations
import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

VOTE_MAP = {"yes": 1.0, "no": -1.0}

PLM_C  = 0.05
N_JOBS = -1

CHANCELLORS = {
    "2005–09": "Angela Merkel",
    "2009–13": "Angela Merkel",
    "2013–17": "Angela Merkel",
    "2017–21": "Angela Merkel",
    "2021–25": "Olaf Scholz",
    "2025–29": "Friedrich Merz",
}

FRAKTIONSVORSITZENDE = {
    "2005–09": ["Volker Kauder","Peter Struck","Guido Westerwelle",
                "Renate Künast","Fritz Kuhn","Gregor Gysi","Oskar Lafontaine"],
    "2009–13": ["Volker Kauder","Frank-Walter Steinmeier","Birgit Homburger",
                "Rainer Brüderle","Renate Künast","Jürgen Trittin","Gregor Gysi"],
    "2013–17": ["Volker Kauder","Thomas Oppermann","Katrin Göring-Eckardt",
                "Anton Hofreiter","Gregor Gysi","Sahra Wagenknecht","Dietmar Bartsch"],
    "2017–21": ["Volker Kauder","Ralph Brinkhaus","Andrea Nahles","Rolf Mützenich",
                "Alexander Gauland","Alice Weidel","Christian Lindner",
                "Katrin Göring-Eckardt","Anton Hofreiter",
                "Sahra Wagenknecht","Dietmar Bartsch","Amira Mohamed Ali"],
    "2021–25": ["Rolf Mützenich","Katharina Dröge","Britta Haßelmann","Christian Dürr",
                "Friedrich Merz","Ralph Brinkhaus","Alice Weidel","Tino Chrupalla",
                "Amira Mohamed Ali","Dietmar Bartsch","Heidi Reichinnek"],
    "2025–29": ["Jens Spahn","Rolf Mützenich","Katharina Dröge","Britta Haßelmann",
                "Alice Weidel","Tino Chrupalla","Heidi Reichinnek"],
}

# ── Theme ──────────────────────────────────────────────────────────────────────
LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme / "ising"
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", sub="#555555",
             grid="#dddddd", ax="#cccccc")
else:
    T = dict(bg="#0d1117", text="white", sub="#888888",
             grid="#1e2530", ax="#333333")

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


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_vote_matrix(period_key: str):
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)

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

    yes_frac = np.nanmean(S == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S = S[:, keep]

    n_poll_k = keep.sum()
    active = (~np.isnan(S)).sum(axis=1) >= max(3, 0.1 * n_poll_k)
    S     = S[active]
    nodes = nodes[active].reset_index(drop=True)
    return S, nodes


# ══════════════════════════════════════════════════════════════════════════════
# PLM — return both J_i and intercept h_i
# ══════════════════════════════════════════════════════════════════════════════

def _fit_one_spin(i, S, m):
    N = S.shape[0]
    obs = ~np.isnan(S[i])
    y   = S[i, obs]
    X   = S[:, obs].T.copy()
    for j in range(N):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = m[j]
    X[:, i] = 0.0
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.zeros(N), 0.0
    y01 = (y + 1) / 2
    clf = LogisticRegression(C=PLM_C, penalty="l2", solver="lbfgs",
                             max_iter=500, fit_intercept=True)
    clf.fit(X, y01)
    J_i = clf.coef_[0].copy()
    J_i[i] = 0.0
    h_i = float(clf.intercept_[0])
    return J_i, h_i


def fit_plm(S):
    """Return (coupling per MP, field h per MP)."""
    N = S.shape[0]
    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_one_spin)(i, S, m) for i in range(N)
    )
    J_rows = np.array([r[0] for r in results])      # (N, N)
    h      = np.array([r[1] for r in results])      # (N,)
    J = (J_rows + J_rows.T) / 2.0
    np.fill_diagonal(J, 0.0)
    coupling = np.abs(J).sum(axis=1) / (N - 1)
    return coupling, h


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

period_dfs = []
for period_key, lbl in PERIODS:
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        print(f"  {lbl}: no data, skipping")
        period_dfs.append((lbl, pd.DataFrame()))
        continue

    print(f"\n{lbl}: loading…", flush=True)
    S, nodes = load_vote_matrix(period_key)
    N, M = S.shape
    print(f"  {N} MPs × {M} polls", flush=True)

    print(f"  fitting PLM…", flush=True)
    coupling, h = fit_plm(S)

    df = nodes[["person_id", "name", "party"]].copy()
    df["coupling"] = coupling * 1000   # ×10³ for readability
    df["h"]        = h
    df["abs_h"]    = np.abs(h)

    period_dfs.append((lbl, df))
    print(f"  done. |h| range: [{df['abs_h'].min():.3f}, {df['abs_h'].max():.3f}]", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

DISPLAY_NAME = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne", "CDU/CSU": "CDU/CSU",
    "Die Linke": "Linke", "BSW": "BSW", "AfD": "AfD",
    "FDP": "FDP", "SPD": "SPD", "fraktionslos": "fraktl.",
}
PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
               "BÜNDNIS 90/DIE GRÜNEN", "Die Linke", "fraktionslos"]

hl_color  = "#111111" if LIGHT_MODE else "white"
hl_stroke = T["bg"]


def draw_panel(ax, lbl, df, *, big=False):
    """Draw one period's |h| vs coupling scatter onto ax."""
    ax.set_facecolor(T["bg"])
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=T["sub"], labelsize=10, length=0)

    if df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                color=T["sub"], ha="center", va="center")
        return {}

    parties_seen = {}
    for party, grp in df.groupby("party"):
        color = party_color.get(party, "#888888")
        parties_seen[party] = color
        ax.scatter(grp["abs_h"], grp["coupling"],
                   c=color, s=(22 if big else 18), alpha=0.72,
                   linewidths=0, zorder=3)

    ax.yaxis.grid(True, color=T["grid"], lw=0.4, zorder=0)
    ax.xaxis.grid(True, color=T["grid"], lw=0.4, zorder=0)

    hl_texts, hl_xs, hl_ys = [], [], []

    chancellor_name = CHANCELLORS.get(lbl)
    if chancellor_name:
        row = df[df["name"] == chancellor_name]
        if not row.empty:
            cx = float(row["abs_h"].iloc[0])
            cy = float(row["coupling"].iloc[0])
            ch_color = party_color.get(row["party"].iloc[0], "#888888")
            ax.scatter(cx, cy, s=(160 if big else 130), facecolors="none",
                       edgecolors=hl_color, linewidths=2.0, zorder=6)
            ax.scatter(cx, cy, s=(70 if big else 60), c=ch_color,
                       linewidths=0, zorder=7)
            t = ax.text(cx, cy, chancellor_name,
                        fontsize=(11 if big else 10), fontweight="bold",
                        color=hl_color, va="center", ha="center", zorder=8)
            hl_texts.append(t); hl_xs.append(cx); hl_ys.append(cy)

    for fv_name in FRAKTIONSVORSITZENDE.get(lbl, []):
        if fv_name == chancellor_name:
            continue
        row = df[df["name"] == fv_name]
        if row.empty:
            continue
        fx = float(row["abs_h"].iloc[0])
        fy = float(row["coupling"].iloc[0])
        fv_color = party_color.get(row["party"].iloc[0], "#888888")
        ax.scatter(fx, fy, s=(90 if big else 75), facecolors="none",
                   edgecolors=hl_color, linewidths=1.3, zorder=6, alpha=0.85)
        ax.scatter(fx, fy, s=(42 if big else 35), c=fv_color,
                   linewidths=0, zorder=7)
        t = ax.text(fx, fy, fv_name.split()[-1],
                    fontsize=(10 if big else 9), color=hl_color,
                    va="center", ha="center", zorder=8)
        hl_texts.append(t); hl_xs.append(fx); hl_ys.append(fy)

    if hl_texts:
        adjust_text(
            hl_texts,
            x=np.array(hl_xs), y=np.array(hl_ys),
            ax=ax,
            expand=(1.4, 1.6),
            force_text=(0.4, 0.6),
            force_points=(0.3, 0.5),
            arrowprops=dict(arrowstyle="-", color=hl_color, lw=0.7, alpha=0.6),
        )
        for t in hl_texts:
            t.set_path_effects([pe.withStroke(linewidth=2.3, foreground=hl_stroke)])

    xmax = np.nanpercentile(df["abs_h"], 99) * 1.10
    ymax = np.nanpercentile(df["coupling"], 99) * 1.15
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    return parties_seen


# ══════════════════════════════════════════════════════════════════════════════
# Combined 2×3 grid
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(26, 18))
fig.patch.set_facecolor(T["bg"])

all_parties_seen = {}
for idx, (ax, (lbl, df)) in enumerate(zip(axes.flat, period_dfs)):
    ax_row, ax_col = divmod(idx, 3)
    ax.set_title(lbl, color=T["text"], fontsize=16, fontweight="bold",
                 loc="left", pad=8)
    seen = draw_panel(ax, lbl, df, big=False)
    all_parties_seen.update(seen)
    if ax_row == 1:
        ax.set_xlabel("Field strength |h|  →  more peer-independent vote",
                      color=T["sub"], fontsize=12)
    if ax_col == 0:
        ax.set_ylabel("← Less coupling  —  More peer influence →",
                      color=T["sub"], fontsize=12)

legend_handles = [
    plt.Line2D([0], [0], marker="o", color="none",
               markerfacecolor=all_parties_seen.get(p, party_color.get(p, "#888")),
               markersize=6, label=DISPLAY_NAME.get(p, p))
    for p in PARTY_ORDER if p in all_parties_seen
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 0.018), ncol=len(legend_handles),
           frameon=False, fontsize=12,
           handlelength=0.7, handletextpad=0.4, columnspacing=1.4,
           labelcolor=T["sub"])

fig.text(0.5, 0.997,
         "Two channels of influence: personal field vs peer coupling · Bundestag 2005–2029",
         ha="center", va="top", color=T["text"], fontsize=20, fontweight="bold")
fig.text(0.5, 0.968,
         "X: |h_i| field strength — bias to vote yes/no regardless of peers  ·  "
         "Y: mean |J_ij| peer coupling — both from regularised PLM inverse Ising  ·  "
         "chancellor (bold ring) and Fraktionsvorsitzende (thin ring) highlighted",
         ha="center", va="top", color=T["sub"], fontsize=11)

fig.subplots_adjust(left=0.07, right=0.995, top=0.925, bottom=0.055,
                    wspace=0.10, hspace=0.22)

out = IMG_DIR / "mp_ising_field.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-period individual plots → output/bundestag_{key}/
# ══════════════════════════════════════════════════════════════════════════════

print("\nSaving per-period plots…")
for (period_key, lbl), (lbl2, df) in zip(PERIODS, period_dfs):
    out_path = BASE_DIR / "output" / period_key / f"plm_field_{_theme}.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(T["bg"])
    seen = draw_panel(ax, lbl, df, big=True)

    if df.empty:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=T["bg"])
        plt.close()
        print(f"  {lbl}: no data → {out_path}")
        continue

    ax.set_xlabel("Field strength |h|  →  more peer-independent vote",
                  color=T["sub"], fontsize=12)
    ax.set_ylabel("← Less coupling  —  More peer influence →",
                  color=T["sub"], fontsize=12)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=seen.get(p, party_color.get(p, "#888")),
                   markersize=7, label=DISPLAY_NAME.get(p, p))
        for p in PARTY_ORDER if p in seen
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=10, labelcolor=T["sub"],
              handlelength=0.6, handletextpad=0.4, columnspacing=1.2)

    ax.set_title(
        f"Personal field vs peer coupling  ·  Bundestag {lbl}",
        color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
    ax.text(0, -0.10,
            "X: |h_i| field strength (peer-independent vote bias)  ·  "
            "Y: mean |J_ij| peer coupling  ·  both from regularised PLM inverse Ising  ·  "
            "chancellor (bold ring), Fraktionsvorsitzende (thin ring)",
            transform=ax.transAxes, color=T["sub"], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=T["bg"])
    plt.close()
    print(f"  Saved → {out_path}")
