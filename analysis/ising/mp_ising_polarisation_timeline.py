"""
mp_ising_polarisation_timeline.py

Timeline of Ising energy per roll-call vote — MP level.

  x = poll date  (field_poll_date from API)
  y = Ising energy  E(σ) = −σᵀKσ  /  n_voting_pairs
        K_ij = Cohen's κ from edges_allpairs.csv  (full-period coupling)
        σ_i  = +1 yes / −1 no / 0 absent or abstain (per MP per poll)

  High energy = cross-cutting vote — MPs vote against their usual pairing
  Low energy  = vote follows habitual coupling pattern

Energy is normalised by the number of voting MP pairs per poll, then
z-scored within each legislature for cross-period comparability.

Usage:
    python analysis/mp_ising_polarisation_timeline.py        # dark
    python analysis/mp_ising_polarisation_timeline.py light
"""

import json, sys, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

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

VOTE_MAP = {"yes": 1.0, "no": -1.0}   # abstain / no_show → 0 (absent)

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)
BG      = "#ffffff" if LIGHT_MODE else "#0d1117"
TEXT    = "#1a1a1a" if LIGHT_MODE else "white"
SUBTEXT = "#555555" if LIGHT_MODE else "#888888"
GRID    = "#e8e8e8" if LIGHT_MODE else "#1e2530"

HIGHLIGHT_COLOR = "#e05c4a" if not LIGHT_MODE else "#c0392b"
DOT_COLOR       = "#4a7fa5"

# ── Load one period ───────────────────────────────────────────────────────────

def load_period(period_key, label):
    d = BASE_DIR / "output" / period_key
    for f in ["raw.json", "edges_allpairs.csv", "nodes.csv"]:
        if not (d / f).exists():
            print(f"  {label}: missing {f}, skipping")
            return None

    # ── κ coupling matrix ─────────────────────────────────────────────────────
    nodes = pd.read_csv(d / "nodes.csv")
    n     = len(nodes)
    pid_to_idx = {pid: i for i, pid in enumerate(nodes["person_id"])}

    edges = pd.read_csv(d / "edges_allpairs.csv")
    ii    = edges["source"].map(pid_to_idx)
    jj    = edges["target"].map(pid_to_idx)
    valid = ii.notna() & jj.notna()
    ii    = ii[valid].astype(int).values
    jj    = jj[valid].astype(int).values
    ww    = edges["weight"].values[valid]

    K = np.zeros((n, n), dtype=np.float32)
    K[ii, jj] = ww
    K[jj, ii] = ww   # symmetrise

    # ── Vote matrix S: (n_mp, n_poll)  ±1 / 0 ────────────────────────────────
    with open(d / "raw.json") as f:
        raw = json.load(f)

    poll_ids    = [p["id"] for p in raw["polls"]]
    poll_dates  = {p["id"]: p.get("field_poll_date", "") for p in raw["polls"]}
    poll_titles = {p["id"]: p.get("label", "")           for p in raw["polls"]}
    poll_idx    = {pid: i for i, pid in enumerate(poll_ids)}

    S = np.zeros((n, len(poll_ids)), dtype=np.float32)
    for v in raw["votes"]:
        val = VOTE_MAP.get(v["vote"])          # None → absent/abstain → leave 0
        if val is None: continue
        mi = pid_to_idx.get(v["mandate"]["id"])
        pi = poll_idx.get(v["poll"]["id"])
        if mi is not None and pi is not None:
            S[mi, pi] = val

    # ── Filter near-unanimous polls ───────────────────────────────────────────
    n_voted = (S != 0).sum(axis=0)
    n_yes   = (S ==  1).sum(axis=0)
    yes_frac = np.where(n_voted > 0, n_yes / n_voted, 0.5)
    keep     = (yes_frac >= 0.05) & (yes_frac <= 0.95) & (n_voted >= 20)

    S_k          = S[:, keep]
    poll_ids_k   = [poll_ids[i] for i in np.where(keep)[0]]
    n_polls      = S_k.shape[1]

    # ── Energy per poll  E = −σᵀKσ  normalised by voting pairs ───────────────
    KS     = K @ S_k                            # (n, n_polls)
    E_raw  = -(S_k * KS).sum(axis=0)           # scalar per poll
    n_v    = (S_k != 0).sum(axis=0).astype(float)
    n_pairs = np.maximum(n_v * (n_v - 1) / 2, 1)
    E      = E_raw / n_pairs                    # average energy per MP pair

    dates  = [poll_dates.get(pid, "")  for pid in poll_ids_k]
    titles = [poll_titles.get(pid, "") for pid in poll_ids_k]

    print(f"  {label}: {n} MPs, {n_polls} polls  "
          f"E/pair ∈ [{E.min():.4f}, {E.max():.4f}]")
    return dict(label=label, E=E, dates=dates, titles=titles)


# ── Collect all periods ───────────────────────────────────────────────────────

all_rows = []
for pk, lbl in PERIODS:
    print(f"Loading {lbl}…")
    rec = load_period(pk, lbl)
    if rec is None: continue
    mu, sd = rec["E"].mean(), rec["E"].std()
    E_z = (rec["E"] - mu) / (sd + 1e-9)
    for i in range(len(rec["E"])):
        if not rec["dates"][i]: continue
        all_rows.append(dict(
            date   = pd.to_datetime(rec["dates"][i]),
            E_raw  = float(rec["E"][i]),
            E_z    = float(E_z[i]),
            title  = rec["titles"][i],
            period = lbl,
        ))

df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
print(f"\nTotal: {len(df)} dated polls")
print(f"Z-score range: [{df['E_z'].min():.2f}, {df['E_z'].max():.2f}]")


# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Period dividers + labels
period_bounds = df.groupby("period")["date"].agg(["min", "max"])
ordered_periods = [lbl for _, lbl in PERIODS if lbl in period_bounds.index]

# Collect boundary dates for x-tick placement
boundary_dates  = []
boundary_labels = []
for i, per in enumerate(ordered_periods):
    row = period_bounds.loc[per]
    # Left edge of this period only
    boundary_dates.append(row["min"])
    boundary_labels.append(str(row["min"].year))
    # Divider line at interior boundaries
    if i > 0:
        ax.axvline(row["min"], color=GRID, lw=0.8, alpha=0.7, zorder=1)

ax.set_xticks(boundary_dates)
ax.set_xticklabels(boundary_labels, fontsize=8, color=SUBTEXT)
ax.tick_params(axis="x", length=4, color=GRID)

# Reference lines
ax.axhline(0, color=SUBTEXT, lw=0.8, alpha=0.4, zorder=1)
for v in np.arange(-4, 6, 1):
    ax.axhline(v, color=GRID, lw=0.4, zorder=1)

# All dots
ax.scatter(df["date"], df["E_z"],
           color=DOT_COLOR, s=24, alpha=0.55, linewidths=0, zorder=4)

# ── One standout per period ───────────────────────────────────────────────────
used_y = []
for per in ordered_periods:
    sub = df[df["period"] == per]
    if sub.empty: continue
    idx = sub["E_z"].idxmax()
    row = df.loc[idx]
    title = textwrap.fill(row["title"], width=22)

    ax.scatter(row["date"], row["E_z"],
               s=80, color=HIGHLIGHT_COLOR,
               edgecolors="white", linewidths=1.2, zorder=7)

    y_base = float(row["E_z"]) + 0.25
    for yu in used_y:
        if abs(y_base - yu) < 0.55:
            y_base = yu + 0.60
    used_y.append(y_base)

    t = ax.text(
        row["date"], y_base, title,
        color=HIGHLIGHT_COLOR, fontsize=7, ha="center", va="bottom",
        zorder=8, linespacing=1.4,
    )
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=BG)])
    ax.plot([row["date"], row["date"]], [row["E_z"] + 0.08, y_base - 0.08],
            color=HIGHLIGHT_COLOR, lw=0.8, alpha=0.6, zorder=6)

# Labels
ax.set_xlabel("Date of roll-call vote", color=SUBTEXT, fontsize=10)
ax.set_ylabel("Ising energy per MP pair  (z-scored within legislature)\n"
              "high = MPs voted against their usual pairings  ·  low = habitual blocs",
              color=SUBTEXT, fontsize=9)
ax.tick_params(colors=SUBTEXT, labelsize=8)
for sp in ax.spines.values(): sp.set_visible(False)

ax.set_title(
    f"Which Bundestag vote was most cross-cutting?  ({len(df)} roll-calls, 2005–present)\n"
    "Energy = −σᵀKσ / n_pairs  ·  K = pairwise Cohen's κ  ·  "
    "σ = individual MP votes  ·  z-scored per legislature",
    color=TEXT, fontsize=11, fontweight="bold", loc="left", pad=12,
)
out    = IMG_DIR / f"mp_ising_polarisation_timeline.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
print(f"\nSaved → {out}")
