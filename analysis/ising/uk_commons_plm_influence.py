"""
uk_commons_plm_influence.py

2D scatter: per-MP PLM coupling vs majority alignment for UK House of Commons.

  X  c_i = ⟨σ_i · γ⟩          alignment with chamber majority (government = right)
  Y  mean_j |J_ij|             PLM coupling strength

Refits PLM from scratch per period so coupling and alignment
share the exact same filtered vote matrix.

Usage
-----
  python analysis/ising/uk_commons_plm_influence.py
  python analysis/ising/uk_commons_plm_influence.py light
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
    ("uk_commons_2017_2019", "2017–19"),
    ("uk_commons_2019_2024", "2019–24"),
    ("uk_commons_2024_2029", "2024–"),
]

VOTE_MAP = {"yes": 1.0, "no": -1.0}
PLM_C    = 0.05
N_JOBS   = -1

# Primary PM per period (for bold-ring highlight)
PM = {
    "2017–19": "Theresa May",
    "2019–24": "Boris Johnson",
    "2024–":   "Keir Starmer",
}

# Notable party leaders to label (equivalent of Fraktionsvorsitzende)
LEADERS = {
    "2017–19": ["Jeremy Corbyn", "Vince Cable", "Ian Blackford"],
    "2019–24": ["Keir Starmer", "Ed Davey", "Ian Blackford",
                "Stephen Flynn", "Rishi Sunak"],
    "2024–":   ["Kemi Badenoch", "Ed Davey", "Stephen Flynn", "Nigel Farage"],
}

# Chief Whips per period — government + main opposition
WHIPS = {
    "2017–19": ["Julian Smith", "Nick Brown", "Patrick Grady"],
    "2019–24": ["Mark Spencer", "Alan Campbell", "Owen Thompson"],
    "2024–":   ["Alan Campbell", "Stuart Andrew", "Wendy Chamberlain", "Kirsty Blackman"],
}

PARTY_COLORS = {
    "Conservative":      "#0087DC",
    "Labour":            "#E4003B",
    "Liberal Democrat":  "#FAA61A",
    "SNP":               "#009DD1",
    "Green":             "#02A95B",
    "Plaid Cymru":       "#005B54",
    "DUP":               "#D46A4B",
    "Reform UK":         "#12B6CF",
    "Alba":              "#005EB8",
    "Independent":       "#888888",
}
def pc(p: str) -> str:
    for key, col in PARTY_COLORS.items():
        if key.lower() in str(p).lower():
            return col
    return "#888888"

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme = "light" if LIGHT_MODE else "dark"
IMG_DIR = BASE_DIR / "output" / "img" / _theme / "ising"
IMG_DIR.mkdir(parents=True, exist_ok=True)

if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", sub="#555555", grid="#dddddd", ax="#cccccc")
else:
    T = dict(bg="#0d1117", text="white", sub="#888888", grid="#1e2530", ax="#333333")

hl_color  = "#111111" if LIGHT_MODE else "white"
hl_stroke = T["bg"]

def load_vote_matrix(period_key: str):
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")

    poll_ids = [p["id"] for p in raw["polls"]]
    poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
    mp_ids   = nodes["person_id"].tolist()
    mp_idx   = {mid: i for i, mid in enumerate(mp_ids)}

    S = np.full((len(mp_ids), len(poll_ids)), np.nan, dtype=np.float32)
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
    nodes = nodes[active].reset_index(drop=True)
    S     = S[active]
    return S, nodes


def _fit_one_spin(i, S, m):
    N = S.shape[0]
    obs = ~np.isnan(S[i])
    y   = S[i, obs]
    X   = S[:, obs].T.copy()
    for j in range(N):
        nm = np.isnan(X[:, j])
        X[nm, j] = m[j]
    X[:, i] = 0.0
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.zeros(N)
    clf = LogisticRegression(C=PLM_C, penalty="l2", solver="lbfgs",
                             max_iter=500, fit_intercept=True)
    clf.fit(X, (y + 1) / 2)
    J_i = clf.coef_[0].copy()
    J_i[i] = 0.0
    return J_i


def fit_and_align(S, nodes):
    N = S.shape[0]
    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    J_rows = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_one_spin)(i, S, m) for i in range(N))
    J_raw = np.array(J_rows)
    J = (J_raw + J_raw.T) / 2.0
    np.fill_diagonal(J, 0.0)
    coupling = np.abs(J).sum(axis=1) / (N - 1) * 1000

    gamma = np.sign(np.nansum(S, axis=0)).astype(np.float32)
    gamma[gamma == 0] = np.nan
    alignment = np.full(N, np.nan)
    for i in range(N):
        valid = (~np.isnan(S[i])) & (~np.isnan(gamma))
        if valid.sum() >= 10:
            alignment[i] = float(np.mean(S[i, valid] * gamma[valid]))

    df = nodes[["person_id", "name", "party"]].copy()
    df["coupling"]  = coupling
    df["alignment"] = alignment
    return df.dropna(subset=["alignment"])


# ── Build per-period dataframes ───────────────────────────────────────────────
period_dfs = []

for period_key, lbl in PERIODS:
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        print(f"  {lbl}: no data, skipping")
        period_dfs.append((lbl, pd.DataFrame()))
        continue

    print(f"\n{lbl}: loading…", flush=True)
    S, nodes = load_vote_matrix(period_key)
    print(f"  {S.shape[0]} MPs × {S.shape[1]} polls — fitting PLM…", flush=True)
    df = fit_and_align(S, nodes)
    print(f"  done. {len(df)} MPs with coupling + alignment")
    period_dfs.append((lbl, df))


# ── Combined 3-panel plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(26, 9))
fig.patch.set_facecolor(T["bg"])

for idx, (ax, (lbl, df)) in enumerate(zip(axes.flat, period_dfs)):
    ax.set_facecolor(T["bg"])
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=T["sub"], labelsize=10, length=0)
    ax.set_title(lbl, color=T["text"], fontsize=16, fontweight="bold",
                 loc="left", pad=8)

    if df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                color=T["sub"], ha="center", va="center")
        continue

    for party, grp in df.groupby("party"):
        ax.scatter(grp["alignment"], grp["coupling"],
                   c=pc(party), s=18, alpha=0.70, linewidths=0, zorder=3)

    ax.axvline(0, color=T["ax"], lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax.yaxis.grid(True, color=T["grid"], lw=0.4, zorder=0)

    hl_texts, hl_xs, hl_ys = [], [], []

    pm_name = PM.get(lbl)
    if pm_name:
        row = df[df["name"] == pm_name]
        if not row.empty:
            cx, cy = float(row["alignment"].iloc[0]), float(row["coupling"].iloc[0])
            col = pc(row["party"].iloc[0])
            ax.scatter(cx, cy, s=130, facecolors="none",
                       edgecolors=hl_color, linewidths=1.8, zorder=6)
            ax.scatter(cx, cy, s=60, c=col, linewidths=0, zorder=7)
            t = ax.text(cx, cy, pm_name.split()[-1],
                        fontsize=10, fontweight="bold", color=hl_color,
                        va="center", ha="center", zorder=8)
            hl_texts.append(t); hl_xs.append(cx); hl_ys.append(cy)
        else:
            ax.text(0.02, 0.97, f"{pm_name.split()[-1]}: below threshold",
                    transform=ax.transAxes, color=T["sub"], fontsize=8,
                    style="italic", va="top")

    for name in LEADERS.get(lbl, []):
        if name == pm_name:
            continue
        row = df[df["name"] == name]
        if row.empty:
            continue
        fx, fy = float(row["alignment"].iloc[0]), float(row["coupling"].iloc[0])
        col = pc(row["party"].iloc[0])
        ax.scatter(fx, fy, s=75, facecolors="none",
                   edgecolors=hl_color, linewidths=1.2, zorder=6, alpha=0.85)
        ax.scatter(fx, fy, s=35, c=col, linewidths=0, zorder=7)
        t = ax.text(fx, fy, name.split()[-1],
                    fontsize=9, color=hl_color, va="center", ha="center", zorder=8)
        hl_texts.append(t); hl_xs.append(fx); hl_ys.append(fy)

    # Whips — diamond marker, labelled with surname + "W"
    for name in WHIPS.get(lbl, []):
        if name in (LEADERS.get(lbl, []) + [pm_name]):
            continue
        row = df[df["name"] == name]
        if row.empty:
            print(f"  whip not found: {name} ({lbl})")
            continue
        wx, wy = float(row["alignment"].iloc[0]), float(row["coupling"].iloc[0])
        col = pc(row["party"].iloc[0])
        ax.scatter(wx, wy, s=110, marker="D", c=col,
                   linewidths=1.4, edgecolors=hl_color, zorder=7, alpha=0.9)
        t = ax.text(wx, wy, f"{name.split()[-1]} ◆",
                    fontsize=8.5, color=hl_color, va="center", ha="center", zorder=8)
        hl_texts.append(t); hl_xs.append(wx); hl_ys.append(wy)

    if hl_texts:
        adjust_text(
            hl_texts,
            x=np.array(hl_xs), y=np.array(hl_ys),
            ax=ax,
            expand=(1.4, 1.6),
            force_text=(0.4, 0.6),
            force_points=(0.3, 0.5),
            arrowprops=dict(arrowstyle="-", color=hl_color, lw=0.6, alpha=0.6),
        )
        for t in hl_texts:
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=hl_stroke)])

    ax.set_xlim(-1.05, 1.05)
    ymax = np.nanpercentile(df["coupling"], 99) * 1.15
    ax.set_ylim(0, ymax)
    ax.set_xlabel("← Consistent opposition  —  Consistent government →",
                  color=T["sub"], fontsize=11)
    if idx == 0:
        ax.set_ylabel("← Less coupled  —  More coupled →",
                      color=T["sub"], fontsize=11)

fig.text(0.5, 0.997,
         "Parliamentary coupling vs majority alignment · UK House of Commons 2017–",
         ha="center", va="top", color=T["text"], fontsize=18, fontweight="bold")
fig.text(0.5, 0.963,
         "Y: mean |J_ij| ×10³ from regularised PLM inverse Ising  ·  "
         "X: ⟨σ_i · γ⟩ correlation with chamber majority  ·  "
         "PM (bold ring), party leaders (thin ring), chief whips (◆ diamond) highlighted",
         ha="center", va="top", color=T["sub"], fontsize=10)

fig.subplots_adjust(left=0.05, right=0.995, top=0.89, bottom=0.10,
                    wspace=0.10)

out = IMG_DIR / "uk_commons_plm_influence.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")
