"""
mp_ising_free_energy.py

3-D mean-field free-energy surface  F(τ, m)

    τ = T / T_c   (normalised temperature)
    m = magnetisation ∈ (−1, 1)

    F_norm(m, τ) = −½ m²  +  τ · [(1+m)/2 · ln((1+m)/2) + (1−m)/2 · ln((1−m)/2)]

    Below T_c  →  double-well (ordered, two degenerate ground states)
    Above T_c  →  single well at m = 0 (disordered)

Each Bundestag period is shown as a coloured slice on the surface at its
empirically fitted τ_eff = T_eff / T_c.

Usage:
    python analysis/mp_ising_free_energy.py        # dark (default)
    python analysis/mp_ising_free_energy.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from pathlib import Path
from scipy.optimize import minimize_scalar

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
PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
               "BÜNDNIS 90/DIE GRÜNEN", "Die Linke"]
ALIASES = {"DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
           "DIE LINKE": "Die Linke", "Die Linke.": "Die Linke"}
def canon(p): return ALIASES.get(str(p), str(p))
VOTE_MAP = {"yes": 1.0, "no": -1.0}

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)
BG      = "#ffffff" if LIGHT_MODE else "#0d1117"
TEXT    = "#1a1a1a" if LIGHT_MODE else "white"
SUBTEXT = "#555555" if LIGHT_MODE else "#888888"
GRID    = "#cccccc" if LIGHT_MODE else "#1e2530"

SLICE_COLORS = ["#e05c4a", "#f5a623", "#f7dc6f", "#82e0aa", "#5dade2", "#a569bd"]

# ── Fit mean-field Ising per period ──────────────────────────────────────────

def fit_period(period_key):
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        return None, None
    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))

    poll_ids = [p["id"] for p in raw["polls"]]
    poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
    mp_ids   = nodes["person_id"].tolist()
    mp_idx   = {mid: i for i, mid in enumerate(mp_ids)}

    S = np.full((len(mp_ids), len(poll_ids)), np.nan, dtype=np.float32)
    for v in raw["votes"]:
        val = VOTE_MAP.get(v["vote"])
        if val is None: continue
        mi = mp_idx.get(v["mandate"]["id"])
        pi = poll_idx.get(v["poll"]["id"])
        if mi is not None and pi is not None:
            S[mi, pi] = val

    yes_frac = np.nanmean(S == 1, axis=0)
    S = S[:, (yes_frac >= 0.05) & (yes_frac <= 0.95)]
    n_kept = S.shape[1]

    active = (~np.isnan(S)).sum(axis=1) >= max(3, 0.1 * n_kept)
    S = S[active]; nodes = nodes[active].reset_index(drop=True)

    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    present = (~np.isnan(S)).astype(np.float32)
    S_fill  = np.nan_to_num(S, nan=0.0)
    Cij = (S_fill @ S_fill.T) / np.maximum(present @ present.T, 1) - np.outer(m, m)
    np.fill_diagonal(Cij, 1.0 - m**2)

    parties = [p for p in PARTY_ORDER if p in nodes["party"].values]
    for p in nodes["party"].unique():
        if p not in parties: parties.append(p)
    n_p = len(parties)

    Jab = np.zeros((n_p, n_p)); Ma = np.zeros(n_p)
    masks = {}
    for a, pa in enumerate(parties):
        masks[a] = nodes["party"].values == pa
        Ma[a] = m[masks[a]].mean() if masks[a].sum() > 0 else 0.0
    for a in range(n_p):
        for b in range(a, n_p):
            block = Cij[np.ix_(masks[a], masks[b])].copy()
            if a == b: np.fill_diagonal(block, np.nan)
            val = np.nanmean(block)
            Jab[a, b] = Jab[b, a] = 0.0 if np.isnan(val) else val

    JM = Jab @ Ma
    valid = (np.abs(Ma) < 0.95) & (np.abs(JM) > 1e-4)
    if valid.sum() >= 2:
        betas = np.arctanh(Ma[valid]) / JM[valid]
        betas = betas[np.isfinite(betas) & (betas > 0)]
        T_eff = float(1.0 / np.median(betas)) if len(betas) else None
    else:
        T_eff = None

    if T_eff is None:
        def res(b): return np.sum((Ma - np.tanh(b * JM))**2)
        r = minimize_scalar(res, bounds=(0.01, 100.0), method="bounded")
        T_eff = float(1.0 / max(r.x, 0.01))

    T_c = float(np.linalg.eigvalsh(Jab).max())
    return T_eff, T_c


# ── Collect τ_eff per period ─────────────────────────────────────────────────

period_data = []
for pk, lbl in PERIODS:
    print(f"Fitting {lbl}…", end=" ", flush=True)
    T_eff, T_c = fit_period(pk)
    if T_eff and T_c and T_c > 0:
        tau = T_eff / T_c
        period_data.append(dict(label=lbl, tau=tau, T_eff=T_eff, T_c=T_c))
        ordered = "❄ ordered" if tau < 1 else "disordered"
        print(f"τ = {tau:.3f}  ({ordered})")
    else:
        print("skipped (missing data)")


# ── Universal free-energy surface ────────────────────────────────────────────

EPS = 1e-9

def F_norm(m, tau):
    """Normalised mean-field free energy f / T_c."""
    m = np.clip(m, -(1 - EPS), 1 - EPS)
    p, q = (1 + m) / 2, (1 - m) / 2
    entropy = p * np.log(p + EPS) + q * np.log(q + EPS)   # ≤ 0
    return -0.5 * m**2 + tau * entropy

m_grid   = np.linspace(-0.999, 0.999, 250)
tau_grid = np.linspace(0.05, 2.5, 250)
M, TAU   = np.meshgrid(m_grid, tau_grid)
F        = F_norm(M, TAU)

# Subtract row-minimum so wells sit at z = 0
F_rel = F - F.min(axis=1, keepdims=True)


# ── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor(BG)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)

# Main surface
surf = ax.plot_surface(
    M, TAU, F_rel,
    cmap="plasma", alpha=0.80,
    linewidth=0, antialiased=True,
    rcount=150, ccount=150,
)

# Period slices
for i, pd_item in enumerate(period_data):
    tau_i = pd_item["tau"]
    col   = SLICE_COLORS[i % len(SLICE_COLORS)]
    row   = np.argmin(np.abs(tau_grid - tau_i))
    ax.plot(
        m_grid, np.full_like(m_grid, tau_i), F_rel[row],
        color=col, lw=2.2, zorder=10,
    )
    # Label at right edge
    ax.text(
        1.02, tau_i, float(F_rel[row, -1]) + 0.002,
        pd_item["label"], color=col, fontsize=8.5, fontweight="bold",
        va="center",
    )

# T_c critical plane  (τ = 1)
row_c = np.argmin(np.abs(tau_grid - 1.0))
ax.plot(m_grid, np.ones_like(m_grid), F_rel[row_c],
        color="white", lw=2, ls="--", alpha=0.55, zorder=9)
ax.text(0.0, 1.03, float(F_rel[row_c, 125]) + 0.004,
        "T = T_c", color=TEXT, fontsize=8, ha="center", va="bottom")

# Equilibrium path  m*(τ) for τ ∈ (0, 1)
# Self-consistency: m = tanh(m/τ) → solved numerically along grid
m_eq = []
for tau_i in tau_grid:
    if tau_i >= 1.0:
        m_eq.append(0.0)
        continue
    # Find non-zero root
    from scipy.optimize import brentq
    try:
        m_star = brentq(lambda x: x - np.tanh(x / tau_i), 0.01, 0.9999)
    except Exception:
        m_star = 0.0
    m_eq.append(m_star)
m_eq = np.array(m_eq)

# Draw both ±m* paths
for sign in [+1, -1]:
    f_eq = np.array([F_rel[j, np.argmin(np.abs(m_grid - sign * m_eq[j]))]
                     for j in range(len(tau_grid))])
    ax.plot(sign * m_eq, tau_grid, f_eq,
            color="white", lw=1.4, alpha=0.80, zorder=11)

# Cosmetics
cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.02)
cbar.set_label("F(m) − F_min  [units of T_c]", color=SUBTEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(colors=SUBTEXT, labelsize=6)

ax.set_xlabel("m  (magnetisation)", color=SUBTEXT, fontsize=9, labelpad=10)
ax.set_ylabel("τ = T / T_c",        color=SUBTEXT, fontsize=9, labelpad=10)
ax.set_zlabel("ΔF / T_c",           color=SUBTEXT, fontsize=9, labelpad=8)
ax.tick_params(colors=SUBTEXT, labelsize=7)

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor(GRID)
ax.grid(color=GRID, lw=0.4, alpha=0.5)

ax.view_init(elev=22, azim=-52)
ax.set_xlim(-1, 1)
ax.set_ylim(tau_grid[0], tau_grid[-1])
ax.set_zlim(0, F_rel.max() * 1.05)

ax.set_title(
    "Mean-field free energy  F(τ, m)  ·  Bundestag Ising model\n"
    "Double-well (τ < 1, ordered) → single well (τ > 1, disordered)  "
    "·  white lines = equilibrium m*(τ)  ·  dashed = T_c",
    color=TEXT, fontsize=10, fontweight="bold", pad=14,
)
out    = IMG_DIR / f"mp_ising_free_energy.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
print(f"\nSaved → {out}")
