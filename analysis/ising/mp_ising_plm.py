"""
mp_ising_plm.py

Regularised pseudo-likelihood maximisation (PLM) inverse Ising model.

For each Bundestag period:
  1. Build MP × poll vote matrix  (+1 / −1 / NaN)
  2. Filter near-unanimous polls and low-participation MPs
  3. For each spin i: fit L2-regularised logistic regression
         P(σ_i | σ_{-i}) = sigmoid(h_i + Σ_j J_ij σ_j)
     Missing σ_j imputed with empirical mean m_j.
  4. Symmetrise:  J_ij ← (J_ij + J_ji) / 2
  5. Party-level J_ab = mean J_ij over (i∈a, j∈b) pairs
  6. T_eff from Curie-Weiss self-consistency on J_ab
  7. T_c = λ_max(J_ab)

Output
------
  output/img/{theme}/mp_ising_plm.png    — J heatmaps + T_eff/T_c per period

Usage
-----
  python analysis/ising/mp_ising_plm.py
  python analysis/ising/mp_ising_plm.py light
"""

from __future__ import annotations
import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq

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

PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
               "BÜNDNIS 90/DIE GRÜNEN", "Die Linke"]

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

VOTE_MAP = {"yes": 1.0, "no": -1.0}

# PLM regularisation: C = 1/λ; small C = heavy regularisation
PLM_C   = 0.05
N_JOBS  = -1      # use all cores

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
    _raw_col = json.load(f)
party_color = {canon(k): v for k, v in _raw_col.items()}
party_color.setdefault("BSW", "#a020f0")
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
    """Return (S, nodes, n_poll_raw) where S is (n_mp, n_poll) ±1/NaN."""
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

    # Filter near-unanimous polls
    yes_frac = np.nanmean(S == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S = S[:, keep]

    # Filter low-participation MPs
    n_poll_k = keep.sum()
    active = (~np.isnan(S)).sum(axis=1) >= max(3, 0.1 * n_poll_k)
    S     = S[active]
    nodes = nodes[active].reset_index(drop=True)

    return S, nodes, n_poll


# ══════════════════════════════════════════════════════════════════════════════
# PLM
# ══════════════════════════════════════════════════════════════════════════════

def _fit_one_spin(i: int, S: np.ndarray, m: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Fit logistic regression for spin i conditioned on all others.
    Returns (h_i, J_i) where J_i has length N (J_ii set to 0).
    """
    N, M = S.shape
    # Polls where spin i was observed
    obs = ~np.isnan(S[i])
    y = S[i, obs]               # ±1 labels → logistic needs 0/1 but we use {-1,+1} trick below

    # Features: all other spins on those polls, NaN → impute with m_j
    X = S[:, obs].T.copy()      # (n_obs, N)
    for j in range(N):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = m[j]
    X[:, i] = 0.0               # zero out self

    if len(y) < 5 or len(np.unique(y)) < 2:
        return 0.0, np.zeros(N)

    # Convert ±1 → 0/1 for sklearn
    y01 = (y + 1) / 2

    clf = LogisticRegression(
        C=PLM_C, penalty="l2", solver="lbfgs",
        max_iter=500, fit_intercept=True, warm_start=False,
    )
    clf.fit(X, y01)

    h_i = float(clf.intercept_[0])
    J_i = clf.coef_[0].copy()   # length N; J_i[i] meaningless but zeroed above
    J_i[i] = 0.0
    return h_i, J_i


def fit_plm(S: np.ndarray, n_jobs: int = N_JOBS):
    """
    Full PLM: fit all N spins, symmetrise J.
    Returns h (N,), J (N,N) symmetric, zero diagonal.
    """
    N = S.shape[0]
    m = np.nanmean(S, axis=1)
    m = np.nan_to_num(m, nan=0.0)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_one_spin)(i, S, m) for i in range(N)
    )

    h = np.array([r[0] for r in results])
    J_raw = np.array([r[1] for r in results])   # (N, N), row i = J_i·

    # Symmetrise
    J = (J_raw + J_raw.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return h, J


# ══════════════════════════════════════════════════════════════════════════════
# Party aggregation + T_eff
# ══════════════════════════════════════════════════════════════════════════════

def party_J(J: np.ndarray, nodes: pd.DataFrame, parties: list[str]) -> np.ndarray:
    """Mean J_ij over MP pairs (a, b) → party-level matrix."""
    n = len(parties)
    Jab = np.zeros((n, n))
    for ai, a in enumerate(parties):
        ia = nodes.index[nodes["party"] == a].tolist()
        for bi, b in enumerate(parties):
            ib = nodes.index[nodes["party"] == b].tolist()
            if not ia or not ib:
                continue
            if ai == bi:
                # Intra-party: off-diagonal only
                block = J[np.ix_(ia, ib)]
                mask  = ~np.eye(len(ia), dtype=bool)
                vals  = block[mask] if mask.any() else block.ravel()
            else:
                vals = J[np.ix_(ia, ib)].ravel()
            Jab[ai, bi] = vals.mean() if len(vals) else 0.0
    return Jab


def fit_teff(Jab: np.ndarray, M: np.ndarray) -> float:
    """
    Curie-Weiss self-consistency: M_a = tanh(β Σ_b J_ab M_b).
    Returns T_eff = 1/β_eff, or NaN if no solution found.
    """
    def residual(beta):
        M_pred = np.tanh(beta * Jab @ M)
        return np.mean((M_pred - M) ** 2)

    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(residual, bounds=(0.01, 100), method="bounded")
        return 1.0 / res.x if res.x > 0 else np.nan
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

results_all = []

for period_key, lbl in PERIODS:
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        print(f"  {lbl}: no raw.json, skipping")
        continue

    print(f"\n{lbl}: loading…", flush=True)
    S, nodes, n_poll_raw = load_vote_matrix(period_key)
    N, M = S.shape
    print(f"  {N} MPs × {M} polls", flush=True)

    print(f"  fitting PLM (C={PLM_C})…", flush=True)
    h, J = fit_plm(S)

    # Party-level
    parties = [p for p in PARTY_ORDER if p in nodes["party"].values]
    Jab = party_J(J, nodes, parties)

    # Empirical party magnetisations
    m = np.nanmean(S, axis=1)
    party_m = np.array([
        m[nodes["party"] == p].mean() if (nodes["party"] == p).any() else 0.0
        for p in parties
    ])

    # T_eff, T_c
    T_eff = fit_teff(Jab, party_m)
    eigs  = np.linalg.eigvalsh(Jab)
    T_c   = float(eigs.max())

    print(f"  T_eff={T_eff:.3f}  T_c={T_c:.3f}  ratio={T_eff/T_c:.3f}", flush=True)

    results_all.append(dict(
        lbl=lbl, parties=parties, Jab=Jab,
        h=h, J=J, nodes=nodes, S=S,
        T_eff=T_eff, T_c=T_c,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation  —  2×3 grid: J_ab heatmap per period
# ══════════════════════════════════════════════════════════════════════════════

n_periods = len(results_all)
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.patch.set_facecolor(T["bg"])

# Symmetric diverging colormap centred at 0
vmax = max(abs(r["Jab"]).max() for r in results_all) * 0.9
cmap_name = "RdBu_r" if LIGHT_MODE else "coolwarm"
cmap = plt.get_cmap(cmap_name)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

for ax, res in zip(axes.flat, results_all):
    parties = res["parties"]
    Jab     = res["Jab"]
    n       = len(parties)

    im = ax.imshow(Jab, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = {"BÜNDNIS 90/DIE GRÜNEN": "Grüne", "CDU/CSU": "CDU/CSU",
             "Die Linke": "Linke", "fraktionslos": "fraktl."}
    labels = [short.get(p, p) for p in parties]
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=8, color=T["sub"])
    ax.set_yticklabels(labels, fontsize=8, color=T["sub"])

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{Jab[i,j]:+.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if abs(Jab[i,j]) > vmax * 0.5 else T["text"])

    ax.set_facecolor(T["bg"])
    for sp in ax.spines.values():
        sp.set_color(T["ax"])

    ratio = res["T_eff"] / res["T_c"] if res["T_c"] > 0 else np.nan
    ordered = "ordered" if ratio < 1 else "disordered"
    ax.set_title(
        f"{res['lbl']}   T_eff/T_c = {ratio:.2f}  ({ordered})",
        color=T["text"], fontsize=11, fontweight="bold",
        loc="left", pad=8,
    )

# Colorbar
cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, cax=cb_ax)
cb.set_label("J_ab  (coupling strength)", color=T["sub"], fontsize=9)
cb.ax.yaxis.set_tick_params(color=T["sub"])
plt.setp(cb.ax.yaxis.get_ticklabels(), color=T["sub"])

# Hide unused axes
for ax in axes.flat[n_periods:]:
    ax.set_visible(False)

fig.text(0.5, 0.985,
         "Inverse Ising — party coupling matrix  ·  regularised PLM",
         ha="center", va="top", color=T["text"],
         fontsize=15, fontweight="bold")
fig.text(0.5, 0.962,
         f"L2-regularised pseudo-likelihood (C={PLM_C}).  "
         "J_ab > 0: parties tend to align.  J_ab < 0: tend to oppose.  "
         "T_eff/T_c < 1: ordered (polarised) regime.",
         ha="center", va="top", color=T["sub"], fontsize=8.5)

plt.subplots_adjust(left=0.06, right=0.90, top=0.94, bottom=0.08,
                    hspace=0.38, wspace=0.28)

out = IMG_DIR / "mp_ising_plm.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=T["bg"])
plt.close()
print(f"\nSaved → {out}")
