"""
mp_ising_chancellor_path.py

Trajectory of chancellors through the (|h|, J) plane across parliamentary terms.

  X  |h_i|              personal field — bias to vote yes/no regardless of peers
  Y  mean_j |J_ij|      average PLM peer coupling

Each chancellor = one connected path; dots = terms, arrow = time direction.
Faint grey background = all MPs pooled (context cloud).

Reuses cache from compute; writes output/mp_ising_field_cache.csv.

Usage
-----
  python analysis/ising/mp_ising_chancellor_path.py
  python analysis/ising/mp_ising_chancellor_path.py light
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
CACHE    = BASE_DIR / "output" / "mp_ising_field_cache.csv"

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]
PERIOD_ORDER = [lbl for _, lbl in PERIODS]

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

VOTE_MAP = {"yes": 1.0, "no": -1.0}
PLM_C  = 0.05
N_JOBS = -1

# Chancellors to trace
CHANCELLOR_PATHS = {
    "Angela Merkel":   "CDU/CSU",
    "Olaf Scholz":     "SPD",
    "Friedrich Merz":  "CDU/CSU",
}

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
    party_color["CDU/CSU"] = "#3a3a3a"; party_color["FDP"] = "#f0c000"
else:
    party_color["CDU/CSU"] = "#dddddd"; party_color["FDP"] = "#f5d800"


# ══════════════════════════════════════════════════════════════════════════════
# Compute (cached)
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
    return S, nodes


def _fit_one_spin(i, S, m):
    N = S.shape[0]
    obs = ~np.isnan(S[i]); y = S[i, obs]
    X = S[:, obs].T.copy()
    for j in range(N):
        nan_mask = np.isnan(X[:, j]); X[nan_mask, j] = m[j]
    X[:, i] = 0.0
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.zeros(N), 0.0
    y01 = (y + 1) / 2
    clf = LogisticRegression(C=PLM_C, penalty="l2", solver="lbfgs",
                             max_iter=500, fit_intercept=True)
    clf.fit(X, y01)
    J_i = clf.coef_[0].copy(); J_i[i] = 0.0
    return J_i, float(clf.intercept_[0])


def fit_plm(S):
    N = S.shape[0]
    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    res = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_one_spin)(i, S, m) for i in range(N))
    J_rows = np.array([r[0] for r in res])
    h = np.array([r[1] for r in res])
    J = (J_rows + J_rows.T) / 2.0; np.fill_diagonal(J, 0.0)
    coupling = np.abs(J).sum(axis=1) / (N - 1)
    return coupling, h


if CACHE.exists():
    print("Loading cached field results…")
    all_df = pd.read_csv(CACHE)
else:
    dfs = []
    for period_key, lbl in PERIODS:
        d = BASE_DIR / "output" / period_key
        if not (d / "raw.json").exists():
            print(f"  {lbl}: no data, skipping"); continue
        print(f"{lbl}: fitting PLM…", flush=True)
        S, nodes = load_vote_matrix(period_key)
        coupling, h = fit_plm(S)
        df = nodes[["person_id", "name", "party"]].copy()
        df["period"]   = lbl
        df["coupling"] = coupling * 1000
        df["h"]        = h
        df["abs_h"]    = np.abs(h)
        dfs.append(df)
        print(f"  done", flush=True)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(CACHE, index=False)
    print(f"Cached → {CACHE}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])
ax.spines[:].set_visible(False)
ax.tick_params(colors=T["sub"], labelsize=10, length=0)

# Background context cloud — all MPs pooled
ax.scatter(all_df["abs_h"], all_df["coupling"],
           c=T["sub"], s=6, alpha=0.10, linewidths=0, zorder=1)

ax.xaxis.grid(True, color=T["grid"], lw=0.4, zorder=0)
ax.yaxis.grid(True, color=T["grid"], lw=0.4, zorder=0)

# axis limits from population (clip extremes)
xmax = np.nanpercentile(all_df["abs_h"], 99.5) * 1.05
ymax = np.nanpercentile(all_df["coupling"], 99.5) * 1.10
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)

label_texts = []

for name, party in CHANCELLOR_PATHS.items():
    sub = all_df[all_df["name"] == name].copy()
    if sub.empty:
        print(f"  {name}: not found"); continue
    sub["ord"] = sub["period"].map({p: i for i, p in enumerate(PERIOD_ORDER)})
    sub = sub.sort_values("ord")
    color = party_color.get(party, "#888888")

    xs = sub["abs_h"].values
    ys = sub["coupling"].values
    periods = sub["period"].values

    # Path line
    ax.plot(xs, ys, color=color, lw=2.0, alpha=0.85, zorder=4,
            solid_capstyle="round")
    # Arrows between consecutive points (time direction)
    for k in range(len(xs) - 1):
        ax.annotate("", xy=(xs[k+1], ys[k+1]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.6, alpha=0.9,
                                    shrinkA=8, shrinkB=8),
                    zorder=5)
    # Dots, sized by term order (later = bigger)
    sizes = np.linspace(50, 150, len(xs))
    ax.scatter(xs, ys, c=color, s=sizes, zorder=6, linewidths=1.2,
               edgecolors=T["bg"])

    # Period label at each dot
    for k in range(len(xs)):
        t = ax.text(xs[k], ys[k], f" {periods[k]}",
                    fontsize=7.5, color=T["sub"], va="bottom", ha="left",
                    zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground=T["bg"])])
        label_texts.append(t)

    # Name label at last point
    t = ax.text(xs[-1], ys[-1], name,
                fontsize=11, fontweight="bold", color=T["text"],
                va="center", ha="left", zorder=9)
    t.set_position((xs[-1] + xmax * 0.012, ys[-1] - ymax * 0.03))
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=T["bg"])])
    label_texts.append(t)

ax.set_xlabel("Field strength |h|  →  more peer-independent vote",
              color=T["sub"], fontsize=12)
ax.set_ylabel("← Less coupling  —  More peer influence →",
              color=T["sub"], fontsize=12)

ax.set_title("How chancellors move through influence space  ·  Bundestag 2005–2029",
             color=T["text"], fontsize=15, fontweight="bold", loc="left", pad=10)
ax.text(0, -0.09,
        "Each path = one chancellor across terms  ·  arrow = time  ·  larger dot = later term  ·  "
        "grey cloud = all MPs  ·  axes from regularised PLM inverse Ising",
        transform=ax.transAxes, color=T["sub"], fontsize=8.5)

plt.tight_layout()
out = IMG_DIR / "mp_ising_chancellor_path.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")
