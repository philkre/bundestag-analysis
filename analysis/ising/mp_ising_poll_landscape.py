"""
mp_ising_poll_landscape.py

3-D energy landscape in poll-configuration space.

Each roll-call vote is a party configuration vector σ ∈ {−1,+1}^n_parties.
We embed those configurations in 2-D via PCA, then compute Ising energy
    E(σ) = −σᵀ J σ
using the mean-field coupling matrix J_ab fitted to the data.
The result is a scatter plot where x,y = political geometry of each vote
and z = how "energetically natural" that coalition pattern was.

Low energy  → configuration aligns with habitual couplings (natural coalition)
High energy → frustrated / cross-party vote

Usage:
    python analysis/mp_ising_poll_landscape.py                           # 2021-25
    python analysis/mp_ising_poll_landscape.py bundestag_2017_2021
    python analysis/mp_ising_poll_landscape.py bundestag_2021_2025 light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path
from scipy.optimize import minimize_scalar

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

# ── Load party × poll vote matrix ────────────────────────────────────────────

d = BASE_DIR / "output" / PERIOD
with open(d / "raw.json") as f:
    raw = json.load(f)
nodes = pd.read_csv(d / "nodes.csv")
nodes["party"] = nodes["party"].map(canon)
pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))

poll_ids = [p["id"] for p in raw["polls"]]
poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
poll_titles = {p["id"]: p.get("label", "") for p in raw["polls"]}

# Accumulate per-party mean vote per poll
acc = {}   # party -> poll_i -> [votes]
for v in raw["votes"]:
    val = VOTE_MAP.get(v["vote"])
    if val is None: continue
    party = pid_to_party.get(v["mandate"]["id"])
    if not party: continue
    pi = poll_idx.get(v["poll"]["id"])
    if pi is None: continue
    acc.setdefault(party, {}).setdefault(pi, []).append(val)

parties = [p for p in PARTY_ORDER if p in acc]
for p in acc:
    if p not in parties and p != "fraktionslos":
        parties.append(p)
n_p = len(parties)

# PV: (n_parties, n_polls) ±1 / NaN
PV = np.full((n_p, len(poll_ids)), np.nan)
for a, p in enumerate(parties):
    for pi, votes in acc.get(p, {}).items():
        m = np.mean(votes)
        if m != 0:
            PV[a, pi] = np.sign(m)

# Filter near-unanimous polls
yes_frac = np.nanmean(PV == 1, axis=0)
keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
PV = PV[:, keep]
poll_ids_kept = [poll_ids[i] for i in np.where(keep)[0]]
n_polls = PV.shape[1]
print(f"{PERIOD}: {n_p} parties, {n_polls} contested polls")

# Drop parties with sparse coverage (< 40 % of polls) — e.g. BSW, late entrants
coverage = (~np.isnan(PV)).mean(axis=1)
core     = coverage >= 0.40
parties  = [parties[a] for a in range(n_p) if core[a]]
PV       = PV[core, :]
n_p      = len(parties)
print(f"  core parties ({coverage.min():.0%} threshold): {parties}")

# Complete polls among core parties
complete = ~np.isnan(PV).any(axis=0)
PV_c  = PV[:, complete]               # (n_p, n_complete)
ids_c = [poll_ids_kept[i] for i, c in enumerate(complete) if c]
n_c   = PV_c.shape[1]
print(f"  complete configs: {n_c}")

# ── Fit mean-field J_ab ───────────────────────────────────────────────────────

M_obs = PV_c.mean(axis=1)
C_obs = (PV_c[:, None, :] * PV_c[None, :, :]).mean(axis=2)

Jab   = C_obs - np.outer(M_obs, M_obs)
np.fill_diagonal(Jab, 0.0)   # no self-coupling

# Symmetrise
Jab = (Jab + Jab.T) / 2

print(f"  J_ab range: [{Jab.min():.3f}, {Jab.max():.3f}]")

# ── Ising energy per poll ─────────────────────────────────────────────────────

S = PV_c.T                      # (n_c, n_p)
E = -np.einsum("ki,ij,kj->k", S, Jab, S)   # E(σ) = −σᵀ J σ

print(f"  energy range: [{E.min():.3f}, {E.max():.3f}]")

# ── 2-D PCA of poll configurations ───────────────────────────────────────────

S_centered = S - S.mean(axis=0, keepdims=True)
U, sv, Vt  = np.linalg.svd(S_centered, full_matrices=False)
# Each poll's projection onto PC1, PC2
pca_coords = U[:, :2] * sv[:2]   # (n_c, 2)

var_explained = sv**2 / (sv**2).sum()
print(f"  PCA variance: PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}")

x, y = pca_coords[:, 0], pca_coords[:, 1]

# ── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor(BG)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)

# Normalise energy for colour mapping
E_norm = (E - E.min()) / (E.max() - E.min() + 1e-9)
colors = plt.cm.plasma(E_norm)

sc = ax.scatter(
    x, y, E,
    c=E, cmap="plasma",
    s=18, alpha=0.75, linewidths=0,
    zorder=5,
)

# Vertical stems to z=floor for depth perception
E_floor = E.min() - (E.max() - E.min()) * 0.12
for xi, yi, ei in zip(x, y, E):
    ax.plot([xi, xi], [yi, yi], [E_floor, ei],
            color="#444466" if not LIGHT_MODE else "#cccccc",
            lw=0.3, alpha=0.35, zorder=2)

# Annotate the 5 lowest-energy polls (most natural configurations)
low_idx = np.argsort(E)[:5]
for idx in low_idx:
    title = poll_titles.get(ids_c[idx], "")[:30]
    ax.text(x[idx], y[idx], E[idx],
            f" {title}", color="#82e0aa",
            fontsize=5.5, va="bottom", zorder=15)

# Annotate the 5 highest-energy polls (most frustrated)
high_idx = np.argsort(E)[-5:]
for idx in high_idx:
    title = poll_titles.get(ids_c[idx], "")[:30]
    ax.text(x[idx], y[idx], E[idx],
            f" {title}", color="#e05c4a",
            fontsize=5.5, va="top", zorder=15)

cbar = fig.colorbar(sc, ax=ax, shrink=0.45, aspect=10, pad=0.02)
cbar.set_label("Ising energy  E(σ) = −σᵀJσ", color=SUBTEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(colors=SUBTEXT, labelsize=6)

period_lbl = PERIOD.replace("bundestag_", "").replace("_", " → ")
ax.set_xlabel(f"PC1  ({var_explained[0]:.1%})", color=SUBTEXT, fontsize=9, labelpad=8)
ax.set_ylabel(f"PC2  ({var_explained[1]:.1%})", color=SUBTEXT, fontsize=9, labelpad=8)
ax.set_zlabel("Energy E(σ)",                    color=SUBTEXT, fontsize=9, labelpad=8)
ax.tick_params(colors=SUBTEXT, labelsize=7)

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor(GRID)
ax.grid(color=GRID, lw=0.4, alpha=0.5)

ax.view_init(elev=28, azim=-60)

ax.set_title(
    f"Poll energy landscape  ·  {period_lbl}\n"
    "x,y = PCA of party vote configurations  ·  "
    "z = Ising energy (low = natural coalition, high = cross-aisle)\n"
    "green labels = 5 lowest-energy votes  ·  red = 5 most frustrated",
    color=TEXT, fontsize=10, fontweight="bold", pad=14,
)
out    = IMG_DIR / f"mp_ising_poll_landscape.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
print(f"\nSaved → {out}")
