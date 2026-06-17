"""
mp_ising_longitudinal.py

Track individual MPs across parliamentary periods.
For each period: compute coupling (mean |J_ij|) and alignment (c_i = <σ_i γ>).
Z-score both axes within each period to make cross-period comparison meaningful.
Plot trajectories for MPs appearing in 3+ periods.

Usage
-----
  python analysis/ising/mp_ising_longitudinal.py
  python analysis/ising/mp_ising_longitudinal.py light
"""

from __future__ import annotations
import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent.parent
CACHE    = BASE_DIR / "output" / "mp_ising_longitudinal_cache.csv"

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]
PERIOD_LBLS = [lbl for _, lbl in PERIODS]

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))
VOTE_MAP = {"yes": 1.0, "no": -1.0}

PLM_C  = 0.05
N_JOBS = -1

MIN_PERIODS = 3   # only track MPs with this many periods or more

# ── Theme ──────────────────────────────────────────────────────────────────────
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

# ── Notable MPs to label ───────────────────────────────────────────────────────
LABEL_NAMES = {
    # Chancellors
    "Angela Merkel", "Olaf Scholz",
    # Long-serving faction leaders
    "Gregor Gysi", "Dietmar Bartsch",
    "Volker Kauder", "Rolf Mützenich", "Ralph Brinkhaus",
    "Katrin Göring-Eckardt", "Anton Hofreiter",
    "Renate Künast", "Jürgen Trittin",
    # Senior ministers / party leaders
    "Wolfgang Schäuble", "Sahra Wagenknecht",
    "Andrea Nahles", "Sigmar Gabriel",
    "Jens Spahn", "Karl Lauterbach",
    "Lars Klingbeil", "Saskia Esken",
    "Hubertus Heil", "Claudia Roth",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading + PLM
# ══════════════════════════════════════════════════════════════════════════════

def load_and_compute(period_key: str, lbl: str) -> pd.DataFrame:
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
        if val is None: continue
        mi = mp_idx.get(v["mandate"]["id"])
        pi = poll_idx.get(v["poll"]["id"])
        if mi is not None and pi is not None:
            S[mi, pi] = val

    yes_frac = np.nanmean(S == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S = S[:, keep]
    n_poll_k = keep.sum()
    active = (~np.isnan(S)).sum(axis=1) >= max(3, 0.1 * n_poll_k)
    S = S[active]; nodes = nodes[active].reset_index(drop=True)
    N = S.shape[0]

    # PLM coupling
    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    def _fit(i):
        obs = ~np.isnan(S[i]); y = S[i, obs]
        X = S[:, obs].T.copy()
        for j in range(N):
            nan_mask = np.isnan(X[:, j]); X[nan_mask, j] = m[j]
        X[:, i] = 0.0
        if len(y) < 5 or len(np.unique(y)) < 2: return np.zeros(N)
        y01 = (y + 1) / 2
        clf = LogisticRegression(C=PLM_C, penalty="l2", solver="lbfgs",
                                 max_iter=500, fit_intercept=True)
        clf.fit(X, y01)
        J_i = clf.coef_[0].copy(); J_i[i] = 0.0
        return J_i
    J_rows = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(_fit)(i) for i in range(N))
    J_raw = np.array(J_rows)
    J = (J_raw + J_raw.T) / 2.0; np.fill_diagonal(J, 0.0)
    coupling = np.abs(J).sum(axis=1) / (N - 1)

    # Alignment
    gamma = np.sign(np.nansum(S, axis=0)).astype(np.float32)
    gamma[gamma == 0] = np.nan
    alignment = np.array([
        float(np.mean(S[i, (~np.isnan(S[i])) & (~np.isnan(gamma))] *
                      gamma[(~np.isnan(S[i])) & (~np.isnan(gamma))]))
        if ((~np.isnan(S[i])) & (~np.isnan(gamma))).sum() >= 10
        else np.nan
        for i in range(N)
    ])

    df = nodes[["person_id", "name", "party"]].copy()
    df["period"]    = lbl
    df["coupling"]  = coupling
    df["alignment"] = alignment
    return df.dropna(subset=["alignment"])


# ══════════════════════════════════════════════════════════════════════════════
# Load or compute
# ══════════════════════════════════════════════════════════════════════════════

if CACHE.exists():
    print("Loading cached results…")
    all_df = pd.read_csv(CACHE)
else:
    dfs = []
    for period_key, lbl in PERIODS:
        d = BASE_DIR / "output" / period_key
        if not (d / "raw.json").exists():
            print(f"  {lbl}: no data, skipping"); continue
        print(f"{lbl}: fitting PLM…", flush=True)
        dfs.append(load_and_compute(period_key, lbl))
        print(f"  done", flush=True)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(CACHE, index=False)
    print(f"Cached → {CACHE}")

# ── Percentile rank within each period (0–100) ────────────────────────────────
all_df["coupling_z"] = all_df.groupby("period")["coupling"].transform(
    lambda x: x.rank(pct=True) * 100
)

# ── MPs in MIN_PERIODS+ periods ────────────────────────────────────────────────
counts = all_df.groupby("name")["period"].nunique()
multi  = counts[counts >= MIN_PERIODS].index
df_m   = all_df[all_df["name"].isin(multi)].copy()

# Most recent party per MP
last_party = (df_m.sort_values("period")
              .groupby("name")["party"].last()
              .to_dict())

print(f"\n{len(multi)} MPs in {MIN_PERIODS}+ periods")


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])

period_x = {lbl: i for i, lbl in enumerate(PERIOD_LBLS)}

label_texts = []

# ── Pass 1: faint background lines (all non-labelled MPs) ─────────────────────
for name, grp in df_m.groupby("name"):
    if name in LABEL_NAMES:
        continue
    grp = grp.sort_values("period")
    if len(grp) < MIN_PERIODS:
        continue
    xs = [period_x[p] for p in grp["period"]]
    ys = grp["coupling_z"].values
    ax.plot(xs, ys, color=T["sub"], alpha=0.04, lw=0.7, zorder=1)

# ── Pass 2: highlighted MPs on top ────────────────────────────────────────────
for name, grp in df_m.groupby("name"):
    if name not in LABEL_NAMES:
        continue
    grp = grp.sort_values("period")
    if len(grp) < MIN_PERIODS:
        continue

    party  = last_party[name]
    color  = party_color.get(party, "#888888")
    xs     = [period_x[p] for p in grp["period"]]
    ys     = grp["coupling_z"].values

    ax.plot(xs, ys, color=color, alpha=0.9, lw=2.2, zorder=4)
    ax.scatter(xs, ys, c=color, s=35, alpha=1.0, linewidths=0.6,
               edgecolors="white" if not LIGHT_MODE else "#333", zorder=5)

    # Label at last appearance
    lx, ly = xs[-1], ys[-1]
    t = ax.text(lx + 0.08, ly, name.split()[-1],
                fontsize=8.5, color=T["text"], va="center",
                ha="left", zorder=7,
                path_effects=[pe.withStroke(linewidth=2, foreground=T["bg"])])
    label_texts.append(t)

# Axes
ax.set_xticks(range(len(PERIOD_LBLS)))
ax.set_xticklabels(PERIOD_LBLS, color=T["sub"], fontsize=11)
ax.set_ylabel("Voting influence beyond party line  (percentile within period)",
              color=T["sub"], fontsize=11)
ax.tick_params(axis="y", colors=T["sub"], labelsize=10, length=0)
ax.yaxis.grid(True, color=T["grid"], lw=0.5, ls="--")
ax.axhline(50, color=T["ax"], lw=0.7, ls="--", alpha=0.4)   # median reference
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color(T["ax"])

# Party legend
PARTY_ORDER = ["AfD","CDU/CSU","FDP","BSW","SPD","BÜNDNIS 90/DIE GRÜNEN","Die Linke","fraktionslos"]
DISPLAY     = {"BÜNDNIS 90/DIE GRÜNEN":"Grüne","fraktionslos":"fraktl."}
seen_labelled = df_m[df_m["name"].isin(LABEL_NAMES)]["party"].unique()
handles = [
    plt.Line2D([0],[0], marker="o", color="none",
               markerfacecolor=party_color.get(p,"#888"), markersize=6,
               label=DISPLAY.get(p,p))
    for p in PARTY_ORDER if p in seen_labelled
]
ax.legend(handles=handles, loc="upper left", frameon=False,
          fontsize=9, labelcolor=T["sub"],
          handlelength=0.6, handletextpad=0.4, columnspacing=1.2, ncol=2)

ax.set_title("MP voting influence across Bundestag periods  —  multi-term MPs only",
             color=T["text"], fontsize=14, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.10,
        f"Each line = one MP serving {MIN_PERIODS}+ periods  ·  "
        "Y axis: percentile rank within each period (robust to outliers)  ·  "
        "colour = most recent party",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "mp_ising_longitudinal.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")
