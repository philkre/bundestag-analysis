"""
mp_ising_mds_energy.py

3-D Ising energy landscape over the 2-D MDS political map.

1. Build 2-D MDS embedding from pairwise Cohen's κ (edges_allpairs.csv).
2. Compute each MP's local Ising energy:
       E_i = −∑_j κ_ij · m_i · m_j
   where m_i = mean vote of MP i (sign: +1 pro-government, −1 opposition).
3. Interpolate the per-MP energies onto a smooth 2-D grid → 3-D terrain.
4. Overlay MP scatter coloured by party.

Low-energy valleys  →  stable voting blocs (internally consistent, coupled to
                        similarly-voting neighbours)
High-energy ridges  →  frustrated MPs (coupled to those they disagree with)

Usage:
    python analysis/mp_ising_mds_energy.py                      # 2021-25, dark
    python analysis/mp_ising_mds_energy.py bundestag_2017_2021
    python analysis/mp_ising_mds_energy.py bundestag_2021_2025 light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path
from scipy.interpolate import RBFInterpolator

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent

PERIOD     = sys.argv[1] if len(sys.argv) > 1 else "bundestag_2021_2025"
LIGHT_MODE = len(sys.argv) > 2 and sys.argv[2] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)
BG      = "#ffffff" if LIGHT_MODE else "#0d1117"
TEXT    = "#1a1a1a" if LIGHT_MODE else "white"
SUBTEXT = "#555555" if LIGHT_MODE else "#888888"
GRID    = "#cccccc" if LIGHT_MODE else "#1e2530"

PARTY_ORDER = ["AfD", "CDU/CSU", "FDP", "BSW", "SPD",
               "BÜNDNIS 90/DIE GRÜNEN", "Die Linke"]
ALIASES = {"DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
           "DIE LINKE": "Die Linke", "Die Linke.": "Die Linke"}
def canon(p): return ALIASES.get(str(p), str(p))
VOTE_MAP = {"yes": 1.0, "no": -1.0}

# ── Load data ─────────────────────────────────────────────────────────────────

d = BASE_DIR / "output" / PERIOD
nodes = pd.read_csv(d / "nodes.csv")
nodes["party"] = nodes["party"].map(canon)
edges = pd.read_csv(d / "edges_allpairs.csv")
print(f"{PERIOD}: {len(nodes)} MPs, {len(edges)} pairs")

# Map person_id → index
pid_to_idx = {pid: i for i, pid in enumerate(nodes["person_id"])}
n = len(nodes)

# Build κ matrix (symmetric, NaN → 0 for missing pairs)
kappa = np.zeros((n, n))
for row in edges.itertuples():
    i, j = pid_to_idx.get(row.source), pid_to_idx.get(row.target)
    if i is not None and j is not None:
        kappa[i, j] = kappa[j, i] = row.weight
print(f"  κ range: [{kappa.min():.3f}, {kappa.max():.3f}]")

# ── Mean vote per MP from raw.json ────────────────────────────────────────────

with open(d / "raw.json") as f:
    raw = json.load(f)

poll_ids = [p["id"] for p in raw["polls"]]
poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
mp_idx   = {mid: i for i, mid in enumerate(nodes["person_id"])}

S = np.full((n, len(poll_ids)), np.nan, dtype=np.float32)
for v in raw["votes"]:
    val = VOTE_MAP.get(v["vote"])
    if val is None: continue
    mi = mp_idx.get(v["mandate"]["id"])
    pi = poll_idx.get(v["poll"]["id"])
    if mi is not None and pi is not None:
        S[mi, pi] = val

# Filter near-unanimous polls
yes_frac = np.nanmean(S == 1, axis=0)
S = S[:, (yes_frac >= 0.05) & (yes_frac <= 0.95)]

m_mp = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)   # mean vote per MP

# ── Local Ising energy per MP ─────────────────────────────────────────────────
# E_i = −∑_j κ_ij · m_i · m_j    (local field contribution)

E_local = -(kappa @ m_mp) * m_mp   # (n,)

# MPs with very few pairings → unreliable; mask them out
pair_count = (kappa != 0).sum(axis=1)
active = pair_count >= 5
print(f"  MPs with ≥5 pairs: {active.sum()} / {n}")

# ── 2-D MDS from κ distance ───────────────────────────────────────────────────
# Distance = 1 − κ  (κ=1 → perfectly similar → distance 0)

dist_mat = np.clip(1 - kappa, 0, 2)

# Use only active MPs for MDS
idx_act  = np.where(active)[0]
dist_sub = dist_mat[np.ix_(idx_act, idx_act)]

print(f"  Running classical MDS ({len(idx_act)} MPs)…", flush=True)

def classical_mds(D, k=2):
    """PCoA / classical MDS from a distance matrix D."""
    n  = D.shape[0]
    D2 = D ** 2
    H  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * H @ D2 @ H
    # Enforce symmetry
    B  = (B + B.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    # Take top-k positive eigenvalues
    pos  = eigvals[:k].clip(0)
    coords = eigvecs[:, :k] * np.sqrt(pos)
    return coords

xy = classical_mds(dist_sub, k=2)

# Flip so left = more-opposition (AfD / right) if possible
parties_sub = nodes["party"].values[idx_act]
afd_mask    = parties_sub == "AfD"
spd_mask    = parties_sub == "SPD"
if afd_mask.any() and spd_mask.any():
    if xy[afd_mask, 0].mean() > xy[spd_mask, 0].mean():
        xy[:, 0] *= -1

x_mp = xy[:, 0]
y_mp = xy[:, 1]
E_mp = E_local[idx_act]
p_mp = parties_sub

# ── Interpolate energy onto grid ──────────────────────────────────────────────

x_min, x_max = x_mp.min(), x_mp.max()
y_min, y_max = y_mp.min(), y_mp.max()
pad_x = (x_max - x_min) * 0.05
pad_y = (y_max - y_min) * 0.05

gx = np.linspace(x_min - pad_x, x_max + pad_x, 120)
gy = np.linspace(y_min - pad_y, y_max + pad_y, 120)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

# RBF interpolation (thin-plate spline feel via neighbours smoothing)
rbf = RBFInterpolator(
    np.column_stack([x_mp, y_mp]),
    E_mp,
    kernel="thin_plate_spline",
    smoothing=len(x_mp) * 0.3,
    degree=1,
)
GE = rbf(grid_pts).reshape(GX.shape)

# ── Party colours ─────────────────────────────────────────────────────────────

with open(BASE_DIR / "config" / "party_colours.json") as f:
    _raw_c = json.load(f)
pc = {canon(k): v for k, v in _raw_c.items()}
pc.setdefault("fraktionslos", "#888888")
pc.setdefault("BSW", "#a020f0")
if not LIGHT_MODE:
    pc["CDU/CSU"] = "#dddddd"   # near-white against dark bg
    pc["FDP"] = "#f5d800"

# ── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor(BG)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)

# Energy terrain surface
surf = ax.plot_surface(
    GX, GY, GE,
    cmap="RdYlBu_r", alpha=0.60,
    linewidth=0, antialiased=True,
    rcount=100, ccount=100,
)

# MP scatter coloured by party
for party in PARTY_ORDER + ["fraktionslos"]:
    mask = p_mp == party
    if not mask.any(): continue
    col  = pc.get(party, "#888888")
    ax.scatter(
        x_mp[mask], y_mp[mask], E_mp[mask],
        color=col, s=8, alpha=0.80, linewidths=0,
        label=party, zorder=6,
    )

# Colorbar for surface
cbar = fig.colorbar(surf, ax=ax, shrink=0.40, aspect=10, pad=0.02)
cbar.set_label("Local energy  E_i = −∑ κ_ij m_i m_j", color=SUBTEXT, fontsize=7.5)
cbar.ax.yaxis.set_tick_params(colors=SUBTEXT, labelsize=6)

# Legend for parties
leg = ax.legend(
    loc="upper left", fontsize=6.5, markerscale=1.6,
    frameon=False, ncol=2,
    bbox_to_anchor=(0.0, 0.92),
)
for t in leg.get_texts(): t.set_color(SUBTEXT)

period_lbl = PERIOD.replace("bundestag_", "").replace("_", " → ")
ax.set_xlabel("MDS dim 1  (left–right axis)",  color=SUBTEXT, fontsize=9, labelpad=8)
ax.set_ylabel("MDS dim 2  (coalition axis)",    color=SUBTEXT, fontsize=9, labelpad=8)
ax.set_zlabel("Local Ising energy",             color=SUBTEXT, fontsize=9, labelpad=8)
ax.tick_params(colors=SUBTEXT, labelsize=7)

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor(GRID)
ax.grid(color=GRID, lw=0.4, alpha=0.5)

ax.view_init(elev=30, azim=-50)

ax.set_title(
    f"Ising energy terrain over 2-D political map  ·  {period_lbl}\n"
    "Terrain = smoothed local energy  ·  Dots = MPs coloured by party\n"
    "Valleys = stable blocs  ·  Ridges = frustrated / cross-aisle MPs",
    color=TEXT, fontsize=10, fontweight="bold", pad=14,
)
out    = IMG_DIR / f"mp_ising_mds_energy.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
print(f"\nSaved → {out}")
