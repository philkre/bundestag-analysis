"""
mp_ising.py

Fits a mean-field Ising model to Bundestag roll-call voting data per period.

Pipeline
────────
1. Build MP × poll vote matrix  (+1/−1/NaN) from raw.json
2. Filter near-unanimous polls  (< 5 % minority)
3. Compute empirical moments    m_i = ⟨s_i⟩,  ⟨s_i s_j⟩ over joint observations
4. Coupling matrix              J_ij = C_ij = ⟨s_i s_j⟩ − m_i·m_j
5. Party-level J                J_ab = mean C_ij over MP pairs (a,b)
6. Fit T_eff                    Curie-Weiss self-consistency: M_a = tanh(β ΣJ_ab M_b)
7. Critical temperature         T_c  = λ_max(J_ab)
8. Susceptibility               χ    = T_eff / |T_eff − T_c|
9. Visualise cooling curve + susceptibility

Usage:
    python mp_ising.py           # dark mode
    python mp_ising.py light     # light mode
"""

import json, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar

# ── Paths ──────────────────────────────────────────────────────────────────────
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

ALIASES = {
    "DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "DIE LINKE":  "Die Linke",
    "Die Linke.": "Die Linke",
}
def canon(p): return ALIASES.get(str(p), str(p))

VOTE_MAP = {"yes": 1.0, "no": -1.0}   # abstain / no_show → NaN

# ── Theme ─────────────────────────────────────────────────────────────────────
LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)
if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", subtext="#555555",
             grid="#dddddd", zero="#aaaaaa", annot="#333333")
    C = dict(teff="#2471a3", tc="#c0392b", chi="#888888", band="#2471a3")
else:
    T = dict(bg="#0d1117", text="white", subtext="#888888",
             grid="#1e2530", zero="#444444", annot="#aaaaaa")
    C = dict(teff="#4a9fd4", tc="#e05c4a", chi="#aaaaaa", band="#4a9fd4")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1-4: build J_ab for one period
# ══════════════════════════════════════════════════════════════════════════════

def build_party_J(period_key):
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)

    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))

    # ── Vote matrix ───────────────────────────────────────────────────────────
    poll_ids  = [p["id"] for p in raw["polls"]]
    poll_idx  = {pid: i for i, pid in enumerate(poll_ids)}
    # Only keep MPs that appear in both nodes and votes
    mp_ids    = nodes["person_id"].tolist()
    mp_idx    = {mid: i for i, mid in enumerate(mp_ids)}
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

    # ── Filter near-unanimous polls (< 5 % minority vote) ────────────────────
    yes_frac = np.nanmean(S == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S = S[:, keep]
    n_poll_kept = keep.sum()
    print(f"    polls kept: {n_poll_kept} / {n_poll}")

    # Drop MPs who participated in fewer than 10 % of kept polls (mostly absent)
    participation = (~np.isnan(S)).sum(axis=1)
    active = participation >= max(3, 0.1 * n_poll_kept)
    S     = S[active]
    nodes = nodes[active].reset_index(drop=True)
    mp_ids = nodes["person_id"].tolist()
    n_mp   = len(mp_ids)
    print(f"    active MPs: {n_mp}")

    # ── Empirical moments ─────────────────────────────────────────────────────
    # m_i = ⟨s_i⟩ (mean vote per MP, ignoring NaN)
    m = np.nanmean(S, axis=1)   # (n_mp,)
    m = np.nan_to_num(m, nan=0.0)

    # ⟨s_i s_j⟩ over jointly observed polls
    # Use zero-fill trick: NaN → 0 so cross-products of missing = 0,
    # then divide by joint observation count.
    present = (~np.isnan(S)).astype(np.float32)   # (n_mp, n_poll)
    S_fill  = np.nan_to_num(S, nan=0.0)

    joint_n       = present  @ present.T              # (n_mp, n_mp)
    sum_products  = S_fill   @ S_fill.T               # (n_mp, n_mp)
    mean_products = sum_products / np.maximum(joint_n, 1)

    # C_ij = ⟨s_i s_j⟩ − m_i·m_j
    outer_m = np.outer(m, m)
    Cij = mean_products - outer_m                     # (n_mp, n_mp)
    np.fill_diagonal(Cij, 1.0 - m**2)                # variance on diagonal

    # ── Party-level J_ab ──────────────────────────────────────────────────────
    parties_present = [p for p in PARTY_ORDER
                       if p in nodes["party"].values]
    # also catch unlisted parties
    for p in nodes["party"].unique():
        if p not in parties_present:
            parties_present.append(p)

    n_p = len(parties_present)
    Jab = np.zeros((n_p, n_p))
    Ma  = np.zeros(n_p)

    party_masks = {}
    for a, pa in enumerate(parties_present):
        mask_a = nodes["party"].values == pa
        party_masks[a] = mask_a
        Ma[a] = m[mask_a].mean() if mask_a.sum() > 0 else 0.0

    for a in range(n_p):
        for b in range(a, n_p):
            mask_a = party_masks[a]
            mask_b = party_masks[b]
            if mask_a.sum() == 0 or mask_b.sum() == 0:
                continue
            block  = Cij[np.ix_(mask_a, mask_b)]
            if a == b:
                # exclude diagonal (self-coupling)
                np.fill_diagonal(block, np.nan)
            val = np.nanmean(block)
            Jab[a, b] = Jab[b, a] = 0.0 if np.isnan(val) else val

    return parties_present, Jab, Ma


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5-6: fit T_eff, compute T_c
# ══════════════════════════════════════════════════════════════════════════════

def fit_T_eff(Jab, Ma):
    """
    Curie-Weiss self-consistency: M_a = tanh(β · Σ_b J_ab · M_b)

    Direct estimator: β_a = arctanh(M_a) / (J @ M)_a
    Only uses parties where M_a is not saturated (avoids arctanh divergence).
    Falls back to residual minimisation if too few valid parties.
    """
    JM = Jab @ Ma   # effective field per party

    # Direct estimate — valid where M_a ∈ (−0.95, 0.95) and JM ≠ 0
    valid = (np.abs(Ma) < 0.95) & (np.abs(JM) > 1e-4)
    if valid.sum() >= 2:
        beta_candidates = np.arctanh(Ma[valid]) / JM[valid]
        beta_candidates = beta_candidates[np.isfinite(beta_candidates) & (beta_candidates > 0)]
        if len(beta_candidates) >= 1:
            return 1.0 / float(np.median(beta_candidates))

    # Fallback: residual minimisation
    def residual(beta):
        return np.sum((Ma - np.tanh(beta * JM))**2)
    result = minimize_scalar(residual, bounds=(0.01, 100.0), method="bounded")
    return 1.0 / max(result.x, 0.01)

def critical_T(Jab):
    """T_c = largest eigenvalue of J_ab (mean-field result)."""
    eigvals = np.linalg.eigvalsh(Jab)
    return float(eigvals.max())


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

records = []
for pk, lbl in PERIODS:
    print(f"{lbl}")
    try:
        parties, Jab, Ma = build_party_J(pk)
        T_eff = fit_T_eff(Jab, Ma)
        T_c   = critical_T(Jab)
        gap   = T_eff - T_c
        chi   = T_eff / abs(gap) if abs(gap) > 1e-6 else np.inf
        records.append(dict(period=lbl, T_eff=T_eff, T_c=T_c, chi=chi,
                            ordered=(gap < 0)))
        print(f"    T_eff={T_eff:.3f}  T_c={T_c:.3f}  "
              f"{'ORDERED ❄' if gap < 0 else 'disordered'}")
    except Exception as e:
        print(f"    ERROR: {e}")

df = pd.DataFrame(records)
xi = np.arange(len(df))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 8: Visualise
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                gridspec_kw={"height_ratios": [3, 2]})
fig.patch.set_facecolor(T["bg"])
for ax in (ax1, ax2):
    ax.set_facecolor(T["bg"])
    ax.spines[:].set_visible(False)

# ── Panel 1: Cooling curve ────────────────────────────────────────────────────
lw = 2.2; ms = 8

ax1.plot(xi, df["T_eff"], color=C["teff"], lw=lw, marker="o", ms=ms,
         label="T_eff  (fitted)", zorder=4)
ax1.plot(xi, df["T_c"],   color=C["tc"],   lw=lw, marker="s", ms=ms,
         linestyle="--", label="T_c  (critical)", zorder=4)

# Shade ordered region (T_eff < T_c)
ax1.fill_between(xi, df["T_eff"], df["T_c"],
                 where=(df["T_eff"] < df["T_c"]),
                 color=C["tc"], alpha=0.12, label="Ordered phase", zorder=2)
ax1.fill_between(xi, df["T_eff"], df["T_c"],
                 where=(df["T_eff"] >= df["T_c"]),
                 color=C["teff"], alpha=0.08, zorder=2)

# Phase label per period
for i, row in df.iterrows():
    label = "❄ ordered" if row["ordered"] else "disordered"
    col   = C["tc"] if row["ordered"] else C["teff"]
    ax1.text(i, max(row["T_eff"], row["T_c"]) + 0.02,
             label, ha="center", va="bottom",
             color=col, fontsize=6.5, clip_on=False)

# Final value annotations
for col, color in [("T_eff", C["teff"]), ("T_c", C["tc"])]:
    v = df[col].iloc[-1]
    ax1.text(xi[-1] + 0.15, v, f"{v:.2f}",
             color=color, fontsize=8, va="center", clip_on=False)

for y in np.arange(0, df[["T_eff","T_c"]].max().max() + 0.3, 0.1):
    ax1.axhline(y, color=T["grid"], lw=0.5, zorder=0)

ax1.set_xticks(xi)
ax1.set_xticklabels(df["period"], color=T["text"], fontsize=11, fontweight="bold")
ax1.tick_params(axis="x", length=0, pad=8)
ax1.tick_params(axis="y", colors=T["subtext"], labelsize=8, length=0)
ax1.set_xlim(-0.5, len(df) - 0.3)
ax1.set_ylim(0, df[["T_eff","T_c"]].max().max() + 0.25)

leg = ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
for t in leg.get_texts(): t.set_color(T["subtext"])

ax1.text(0, 1.13, "Ising model fit to Bundestag voting, 2005–2029",
         transform=ax1.transAxes, color=T["text"], fontsize=15,
         fontweight="bold", va="top", clip_on=False)
ax1.text(0, 1.065,
         "Effective temperature T_eff vs critical temperature T_c  ·  "
         "T_eff < T_c = ordered/frozen phase  ·  "
         "Coupling J_ij = connected correlations C_ij from raw roll-call votes",
         transform=ax1.transAxes, color=T["subtext"], fontsize=8.5,
         va="top", clip_on=False)

ax1.text(-0.01, 0.5, "Temperature", transform=ax1.transAxes,
         color=T["subtext"], fontsize=8, rotation=90,
         ha="right", va="center")

# ── Panel 2: Susceptibility ───────────────────────────────────────────────────
chi_plot = df["chi"].clip(upper=20)   # cap for display

ax2.bar(xi, chi_plot, color=C["chi"], alpha=0.6, width=0.5, zorder=3)
ax2.plot(xi, chi_plot, color=C["chi"], lw=1.5, marker="o", ms=6, zorder=4)

for i, (v, raw_v) in enumerate(zip(chi_plot, df["chi"])):
    label = f"{raw_v:.1f}" if raw_v < 20 else "→∞"
    ax2.text(i, v + 0.3, label, ha="center", va="bottom",
             color=T["subtext"], fontsize=7.5)

for y in [5, 10, 15, 20]:
    ax2.axhline(y, color=T["grid"], lw=0.5, zorder=0)

ax2.set_xticks(xi)
ax2.set_xticklabels(df["period"], color=T["text"], fontsize=11, fontweight="bold")
ax2.tick_params(axis="x", length=0, pad=8)
ax2.tick_params(axis="y", colors=T["subtext"], labelsize=8, length=0)
ax2.set_xlim(-0.5, len(df) - 0.3)
ax2.set_ylim(0, 22)

ax2.text(-0.01, 0.5, "Susceptibility χ", transform=ax2.transAxes,
         color=T["subtext"], fontsize=8, rotation=90,
         ha="right", va="center")
ax2.text(0, 1.06,
         "χ = T_eff / |T_eff − T_c|  ·  "
         "Higher = small perturbations cascade  ·  Diverges at phase transition",
         transform=ax2.transAxes, color=T["subtext"], fontsize=8,
         va="bottom", clip_on=False)

fig.subplots_adjust(left=0.06, right=0.93, top=0.83, bottom=0.08, hspace=0.55)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = IMG_DIR / f"mp_ising.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2,
            facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
