"""
mp_ising_poll_landscape_all.py

Poll energy landscape pooled across all Bundestag periods.

For each period:
  1. Build party vote matrix per poll (core parties only, ≥40% coverage)
  2. Fit mean-field J_ab
  3. Compute Ising energy  E(σ) = −σᵀ J σ  per poll

Pool all polls into one matrix (union of parties, absent = 0).
PCA to 2-D, then:
  - z-axis = Ising energy
  - colour = period
  - marker size ∝ |E| (bigger = more extreme)

Shows whether different Bundestag eras occupy different regions of
configuration space and whether their energy distributions differ.

Usage:
    python analysis/mp_ising_poll_landscape_all.py        # dark
    python analysis/mp_ising_poll_landscape_all.py light
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

PERIOD_COLORS = {
    "2005–09": "#e05c4a",
    "2009–13": "#f5a623",
    "2013–17": "#f7dc6f",
    "2017–21": "#82e0aa",
    "2021–25": "#5dade2",
    "2025–29": "#a569bd",
}

MIN_COVERAGE = 0.40   # drop parties present in <40% of polls

# ── Load + process one period ─────────────────────────────────────────────────

def load_period(period_key, label):
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        print(f"  {label}: no raw.json, skipping")
        return None

    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))

    poll_ids    = [p["id"] for p in raw["polls"]]
    poll_titles = {p["id"]: p.get("label", "") for p in raw["polls"]}
    poll_idx    = {pid: i for i, pid in enumerate(poll_ids)}

    # Accumulate per-party majority vote per poll
    acc = {}
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

    PV = np.full((n_p, len(poll_ids)), np.nan)
    for a, p in enumerate(parties):
        for pi, votes in acc.get(p, {}).items():
            mv = np.mean(votes)
            if mv != 0:
                PV[a, pi] = np.sign(mv)

    # Drop near-unanimous polls
    yes_frac = np.nanmean(PV == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    PV = PV[:, keep]
    poll_ids_k = [poll_ids[i] for i in np.where(keep)[0]]

    # Drop sparse parties
    cov  = (~np.isnan(PV)).mean(axis=1)
    core = cov >= MIN_COVERAGE
    parties = [parties[a] for a in range(n_p) if core[a]]
    PV      = PV[core, :]
    n_p     = len(parties)

    # Complete polls only (for energy computation)
    complete = ~np.isnan(PV).any(axis=0)
    PV_c  = PV[:, complete]
    ids_c = [poll_ids_k[i] for i, c in enumerate(complete) if c]
    n_c   = PV_c.shape[1]

    if n_c < 5:
        print(f"  {label}: only {n_c} complete polls, skipping")
        return None

    # Fit mean-field J_ab
    M_obs = PV_c.mean(axis=1)
    C_obs = (PV_c[:, None, :] * PV_c[None, :, :]).mean(axis=2)
    Jab   = (C_obs - np.outer(M_obs, M_obs))
    np.fill_diagonal(Jab, 0.0)
    Jab   = (Jab + Jab.T) / 2

    # Ising energy per poll  E = −σᵀ J σ
    S = PV_c.T                                          # (n_c, n_p)
    E = -np.einsum("ki,ij,kj->k", S, Jab, S)

    titles = [poll_titles.get(pid, "") for pid in ids_c]

    print(f"  {label}: {n_p} parties, {n_c} complete polls  "
          f"E ∈ [{E.min():.2f}, {E.max():.2f}]")

    return dict(
        label=label,
        parties=parties,
        PV_c=PV_c,           # (n_p, n_c)
        E=E,                 # (n_c,)
        ids=ids_c,
        titles=titles,
        Jab=Jab,
    )


# ── Collect all periods ───────────────────────────────────────────────────────

records = []
for pk, lbl in PERIODS:
    print(f"Loading {lbl}…")
    rec = load_period(pk, lbl)
    if rec is not None:
        records.append(rec)

# ── Build union party list + pooled feature matrix ───────────────────────────

all_parties = []
for rec in records:
    for p in rec["parties"]:
        if p not in all_parties:
            all_parties.append(p)
# Sort by PARTY_ORDER, then remaining
party_union = [p for p in PARTY_ORDER if p in all_parties]
for p in all_parties:
    if p not in party_union:
        party_union.append(p)
n_union = len(party_union)
p_idx = {p: i for i, p in enumerate(party_union)}

print(f"\nUnion parties ({n_union}): {party_union}")

# Pooled feature matrix: one row per poll, one col per party (union)
# Missing party in a period → 0 (party absent, not voted)
rows_X  = []
rows_E  = []
rows_lbl= []

for rec in records:
    n_c = rec["PV_c"].shape[1]
    X_per = np.zeros((n_c, n_union))
    for a, p in enumerate(rec["parties"]):
        col = p_idx[p]
        X_per[:, col] = rec["PV_c"][a, :]    # row = poll, col = party
    rows_X.append(X_per)
    rows_E.append(rec["E"])
    rows_lbl.extend([rec["label"]] * n_c)

X_all  = np.vstack(rows_X)        # (total_polls, n_union)
E_all  = np.concatenate(rows_E)   # (total_polls,)
lbl_all = np.array(rows_lbl)

n_total = len(E_all)
print(f"\nPooled: {n_total} polls × {n_union} parties")
print(f"Energy range: [{E_all.min():.2f}, {E_all.max():.2f}]")

# ── PCA ───────────────────────────────────────────────────────────────────────

X_c = X_all - X_all.mean(axis=0, keepdims=True)
U, sv, Vt = np.linalg.svd(X_c, full_matrices=False)
pca_xy = U[:, :2] * sv[:2]

var_exp = sv**2 / (sv**2).sum()
print(f"PCA variance: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")

x_all = pca_xy[:, 0]
y_all = pca_xy[:, 1]

# ── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(BG)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)

# Floor for stems
E_floor = E_all.min() - (E_all.max() - E_all.min()) * 0.10

for rec in records:
    lbl  = rec["label"]
    mask = lbl_all == lbl
    col  = PERIOD_COLORS.get(lbl, "#aaaaaa")
    xm, ym, em = x_all[mask], y_all[mask], E_all[mask]

    # Scatter dots
    ax.scatter(xm, ym, em,
               color=col, s=22, alpha=0.75, linewidths=0,
               label=lbl, zorder=6, depthshade=True)

    # Vertical stems (thin)
    for xi, yi, ei in zip(xm, ym, em):
        ax.plot([xi, xi], [yi, yi], [E_floor, ei],
                color=col, lw=0.25, alpha=0.20, zorder=2)

# Horizontal energy=0 plane (reference)
xr = np.array([x_all.min(), x_all.max()])
yr = np.array([y_all.min(), y_all.max()])
XR, YR = np.meshgrid(xr, yr)
ax.plot_surface(XR, YR, np.zeros_like(XR),
                color="#888888" if not LIGHT_MODE else "#cccccc",
                alpha=0.08, linewidth=0)

# Annotate 3 lowest-energy polls overall
low3 = np.argsort(E_all)[:3]
for idx in low3:
    title = ""
    # Find which period
    for rec in records:
        mask = lbl_all == rec["label"]
        idx_in_rec = np.where(mask)[0]
        if idx in idx_in_rec:
            pos = list(idx_in_rec).index(idx)
            title = rec["titles"][pos][:28]
            break
    ax.text(x_all[idx], y_all[idx], E_all[idx],
            f" {title}", color="#82e0aa", fontsize=5.5, va="top", zorder=15)

# Annotate 3 highest-energy polls overall
high3 = np.argsort(E_all)[-3:]
for idx in high3:
    title = ""
    for rec in records:
        mask = lbl_all == rec["label"]
        idx_in_rec = np.where(mask)[0]
        if idx in idx_in_rec:
            pos = list(idx_in_rec).index(idx)
            title = rec["titles"][pos][:28]
            break
    ax.text(x_all[idx], y_all[idx], E_all[idx],
            f" {title}", color="#e05c4a", fontsize=5.5, va="bottom", zorder=15)

# Legend
leg = ax.legend(
    loc="upper left", fontsize=8, markerscale=1.8,
    frameon=False, ncol=2,
    bbox_to_anchor=(0.0, 0.95),
    title="Period", title_fontsize=7,
)
for t in leg.get_texts(): t.set_color(SUBTEXT)
leg.get_title().set_color(SUBTEXT)

ax.set_xlabel(f"PC1  ({var_exp[0]:.1%})", color=SUBTEXT, fontsize=9, labelpad=8)
ax.set_ylabel(f"PC2  ({var_exp[1]:.1%})", color=SUBTEXT, fontsize=9, labelpad=8)
ax.set_zlabel("Energy  E(σ) = −σᵀJσ",    color=SUBTEXT, fontsize=9, labelpad=8)
ax.tick_params(colors=SUBTEXT, labelsize=7)

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor(GRID)
ax.grid(color=GRID, lw=0.4, alpha=0.5)

ax.view_init(elev=25, azim=-55)

ax.set_title(
    f"Poll energy landscape  ·  all Bundestag periods  ({n_total} votes)\n"
    f"x,y = PCA of party vote configs (union of {n_union} parties, absent = 0)\n"
    "z = Ising energy  ·  green = lowest energy (natural coalitions)  ·  "
    "red = most frustrated",
    color=TEXT, fontsize=10, fontweight="bold", pad=14,
)
out    = IMG_DIR / f"mp_ising_poll_landscape_all.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
print(f"\nSaved → {out}")
