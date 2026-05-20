"""
mp_ising_polarisation.py

2-D scatter: which Bundestag roll-call votes are the most polarising?

Axes
----
  x  = PC1 of party vote configuration  (left ↔ right political alignment)
  y  = PC2  (secondary structure — usually coalition vs. opposition)
  colour = Ising energy  E(σ) = −σᵀJσ
           low (blue)  = vote follows habitual coupling pattern (natural)
           high (red)  = frustrated / cross-cutting coalition

"Most polarising" = highest energy: parties vote in unexpected combinations,
cutting across the normal bloc structure.

Annotates the top-N highest-energy polls (most cross-cutting) and the
top-N lowest-energy polls (most natural/disciplined) with their titles.

Usage:
    python analysis/mp_ising_polarisation.py        # dark
    python analysis/mp_ising_polarisation.py light
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
GRID    = "#e0e0e0" if LIGHT_MODE else "#1e2530"

PERIOD_MARKERS = {
    "2005–09": "o", "2009–13": "s", "2013–17": "D",
    "2017–21": "^", "2021–25": "P", "2025–29": "X",
}
PERIOD_COLORS  = {
    "2005–09": "#e05c4a", "2009–13": "#f5a623", "2013–17": "#f7dc6f",
    "2017–21": "#82e0aa", "2021–25": "#5dade2",  "2025–29": "#a569bd",
}

TOP_N    = 8    # annotate this many highest / lowest energy polls
MIN_COV  = 0.40

# ── Load + process (same logic as poll_landscape_all) ─────────────────────────

def load_period(period_key, label):
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        return None
    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))

    poll_ids    = [p["id"] for p in raw["polls"]]
    poll_titles = {p["id"]: p.get("label", "") for p in raw["polls"]}
    poll_idx    = {pid: i for i, pid in enumerate(poll_ids)}

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
        if p not in parties and p != "fraktionslos": parties.append(p)
    n_p = len(parties)

    PV = np.full((n_p, len(poll_ids)), np.nan)
    for a, p in enumerate(parties):
        for pi, votes in acc.get(p, {}).items():
            mv = np.mean(votes)
            if mv != 0: PV[a, pi] = np.sign(mv)

    yes_frac = np.nanmean(PV == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    PV = PV[:, keep]
    poll_ids_k = [poll_ids[i] for i in np.where(keep)[0]]

    cov  = (~np.isnan(PV)).mean(axis=1)
    core = cov >= MIN_COV
    parties = [parties[a] for a in range(n_p) if core[a]]
    PV = PV[core, :]; n_p = len(parties)

    complete = ~np.isnan(PV).any(axis=0)
    PV_c  = PV[:, complete]
    ids_c = [poll_ids_k[i] for i, c in enumerate(complete) if c]
    if PV_c.shape[1] < 5: return None

    M_obs = PV_c.mean(axis=1)
    C_obs = (PV_c[:, None, :] * PV_c[None, :, :]).mean(axis=2)
    Jab   = (C_obs - np.outer(M_obs, M_obs))
    np.fill_diagonal(Jab, 0.0)
    Jab   = (Jab + Jab.T) / 2

    S = PV_c.T
    E = -np.einsum("ki,ij,kj->k", S, Jab, S)
    titles = [poll_titles.get(pid, "") for pid in ids_c]

    print(f"  {label}: {n_p} parties, {PV_c.shape[1]} polls  "
          f"E∈[{E.min():.2f}, {E.max():.2f}]")
    return dict(label=label, parties=parties, PV_c=PV_c, E=E, titles=titles)


records = []
for pk, lbl in PERIODS:
    print(f"Loading {lbl}…")
    rec = load_period(pk, lbl)
    if rec: records.append(rec)

# ── Union feature matrix + PCA ────────────────────────────────────────────────

all_parties = []
for rec in records:
    for p in rec["parties"]:
        if p not in all_parties: all_parties.append(p)
party_union = [p for p in PARTY_ORDER if p in all_parties]
for p in all_parties:
    if p not in party_union: party_union.append(p)
n_union = len(party_union); p_idx = {p: i for i, p in enumerate(party_union)}

rows_X, rows_E, rows_lbl, rows_title = [], [], [], []
for rec in records:
    n_c  = rec["PV_c"].shape[1]
    X_p  = np.zeros((n_c, n_union))
    for a, p in enumerate(rec["parties"]):
        X_p[:, p_idx[p]] = rec["PV_c"][a, :]
    rows_X.append(X_p); rows_E.append(rec["E"])
    rows_lbl.extend([rec["label"]] * n_c)
    rows_title.extend(rec["titles"])

X_all   = np.vstack(rows_X)
E_all   = np.concatenate(rows_E)
lbl_all = np.array(rows_lbl)
ttl_all = np.array(rows_title)

X_c = X_all - X_all.mean(axis=0, keepdims=True)
U, sv, Vt = np.linalg.svd(X_c, full_matrices=False)
pca_xy = U[:, :2] * sv[:2]
var_exp = sv**2 / (sv**2).sum()

x_all = pca_xy[:, 0]
y_all = pca_xy[:, 1]
n_total = len(E_all)

print(f"\nPooled: {n_total} polls  "
      f"PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")
print(f"Energy range: [{E_all.min():.2f}, {E_all.max():.2f}]")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Background grid
for v in np.linspace(x_all.min() - 0.5, x_all.max() + 0.5, 8):
    ax.axvline(v, color=GRID, lw=0.4, zorder=0)
for v in np.linspace(y_all.min() - 0.5, y_all.max() + 0.5, 8):
    ax.axhline(v, color=GRID, lw=0.4, zorder=0)
ax.axvline(0, color=SUBTEXT, lw=0.7, alpha=0.4, zorder=1)
ax.axhline(0, color=SUBTEXT, lw=0.7, alpha=0.4, zorder=1)

# Colour map: energy low→high = blue→red
E_norm = (E_all - E_all.min()) / (E_all.max() - E_all.min() + 1e-9)
cmap   = plt.cm.RdYlBu_r

# Draw each period with its own marker shape
for lbl in [r["label"] for r in records]:
    mask   = lbl_all == lbl
    marker = PERIOD_MARKERS.get(lbl, "o")
    sc = ax.scatter(
        x_all[mask], y_all[mask],
        c=E_norm[mask], cmap=cmap, vmin=0, vmax=1,
        s=55, marker=marker, alpha=0.80, linewidths=0.3,
        edgecolors=SUBTEXT, zorder=4, label=lbl,
    )

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(E_all.min(), E_all.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Ising energy  E(σ) = −σᵀJσ\n"
               "← natural coalition             cross-cutting →",
               color=SUBTEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(colors=SUBTEXT, labelsize=7)

# ── Annotate high-energy (most polarising) ────────────────────────────────────
high_idx = np.argsort(E_all)[-TOP_N:]
for rank, idx in enumerate(high_idx):
    title = ttl_all[idx][:42] if ttl_all[idx] else f"poll {idx}"
    lbl_i = lbl_all[idx]
    ax.scatter(x_all[idx], y_all[idx],
               s=120, marker=PERIOD_MARKERS.get(lbl_i, "o"),
               facecolors="none", edgecolors="#e05c4a", linewidths=1.8,
               zorder=7)
    # Stagger labels above / below alternately
    dy = 0.04 if rank % 2 == 0 else -0.04
    t = ax.text(
        x_all[idx], y_all[idx] + dy,
        f"{title}\n({lbl_i}  E={E_all[idx]:.1f})",
        color="#e05c4a", fontsize=6.5, ha="center",
        va="bottom" if dy > 0 else "top",
        zorder=8,
    )
    t.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])

# ── Annotate low-energy (most natural/disciplined) ────────────────────────────
low_idx = np.argsort(E_all)[:TOP_N]
for rank, idx in enumerate(low_idx):
    title = ttl_all[idx][:42] if ttl_all[idx] else f"poll {idx}"
    lbl_i = lbl_all[idx]
    ax.scatter(x_all[idx], y_all[idx],
               s=120, marker=PERIOD_MARKERS.get(lbl_i, "o"),
               facecolors="none", edgecolors="#5dade2", linewidths=1.8,
               zorder=7)
    dy = 0.04 if rank % 2 == 0 else -0.04
    t = ax.text(
        x_all[idx], y_all[idx] + dy,
        f"{title}\n({lbl_i}  E={E_all[idx]:.1f})",
        color="#5dade2", fontsize=6.5, ha="center",
        va="bottom" if dy > 0 else "top",
        zorder=8,
    )
    t.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])

# ── Party loading arrows (biplot) ────────────────────────────────────────────
# Show which parties pull votes in which direction
with open(BASE_DIR / "config" / "party_colours.json") as f:
    _rc = json.load(f)
pc_colors = {canon(k): v for k, v in _rc.items()}
pc_colors.update({"fraktionslos": "#888888", "BSW": "#a020f0"})
if not LIGHT_MODE:
    pc_colors["CDU/CSU"] = "#dddddd"; pc_colors["FDP"] = "#f5d800"

scale = 0.6 * min(abs(x_all).max(), abs(y_all).max())
for i, p in enumerate(party_union):
    dx = float(Vt[0, i]) * scale
    dy = float(Vt[1, i]) * scale
    if abs(dx) < 0.05 and abs(dy) < 0.05: continue
    col = pc_colors.get(p, "#aaaaaa")
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.4, mutation_scale=10),
                zorder=5)
    short = p.replace("BÜNDNIS 90/DIE GRÜNEN", "Grüne").replace("Die Linke", "Linke")
    t = ax.text(dx * 1.12, dy * 1.12, short,
                color=col, fontsize=7.5, fontweight="bold",
                ha="center", va="center", zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2, foreground=BG)])

# ── Legend + labels ───────────────────────────────────────────────────────────
leg = ax.legend(
    title="Period", title_fontsize=8, fontsize=8,
    frameon=False, loc="lower right",
    markerscale=1.4,
)
for t in leg.get_texts():  t.set_color(SUBTEXT)
leg.get_title().set_color(SUBTEXT)

ax.set_xlabel(f"PC1  ({var_exp[0]:.1%} variance)  ·  vote configuration axis 1",
              color=SUBTEXT, fontsize=9)
ax.set_ylabel(f"PC2  ({var_exp[1]:.1%} variance)  ·  vote configuration axis 2",
              color=SUBTEXT, fontsize=9)
ax.tick_params(colors=SUBTEXT, labelsize=7)
for sp in ax.spines.values(): sp.set_visible(False)

ax.set_title(
    f"Which Bundestag vote was most polarising?  ({n_total} roll-calls, all periods)\n"
    "Colour = Ising energy  ·  red = cross-cutting (polarising)  ·  "
    "blue = natural coalition (disciplined)\n"
    "Ringed red = most polarising  ·  Ringed blue = most disciplined  ·  "
    "Arrows = party directions in config space",
    color=TEXT, fontsize=11, fontweight="bold", loc="left", pad=14,
)
out    = IMG_DIR / f"mp_ising_polarisation.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
print(f"\nSaved → {out}")
