"""
mp_ising_exact.py

Exact inverse Ising model at party level, inspired by Lee et al. 2017.

For each period:
  1. Discretise party votes per poll: σ_a^v = sign(mean vote of party a)
  2. Fit h_a, J_ab by gradient ascent on exact log-likelihood
     (partition function computed exactly over all 2^n states)
  3. Compare J_ab (true couplings) vs C_ab (raw correlations) — reveals
     hidden negative interactions masked by indirect paths
  4. Fit T_eff from self-consistency using corrected J_ab
  5. Map energy landscape: enumerate all states, find local minima (valleys)
  6. Bootstrap T_eff over poll resamples for uncertainty

Outputs
-------
  output/img/mp_ising_exact.png          dark   (cooling + J heatmaps + valleys)
  output/img/mp_ising_exact_light.png    light

Usage
-----
  python mp_ising_exact.py           # dark
  python mp_ising_exact.py light     # light
"""

import json, re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
from pathlib import Path
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
DISPLAY = {"BÜNDNIS 90/DIE GRÜNEN": "Grüne", "CDU/CSU": "CDU/CSU",
           "Die Linke": "Linke", "BSW": "BSW", "AfD": "AfD",
           "FDP": "FDP", "SPD": "SPD", "fraktionslos": "fraktl."}

ALIASES = {"DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
           "DIE LINKE":  "Die Linke", "Die Linke.": "Die Linke"}
def canon(p): return ALIASES.get(str(p), str(p))

VOTE_MAP = {"yes": 1.0, "no": -1.0}

LIGHT_MODE = len(sys.argv) > 1 and sys.argv[1] == "light"
_theme  = "light" if LIGHT_MODE else "dark"
IMG_DIR  = BASE_DIR / "output" / "img" / _theme
IMG_DIR.mkdir(parents=True, exist_ok=True)
if LIGHT_MODE:
    T = dict(bg="#ffffff", text="#1a1a1a", subtext="#555555",
             grid="#dddddd", na="#eeeeee", gc="#ffffff")
    LIGHT_OV = {"CDU/CSU": "#3a3a3a", "FDP": "#f0c000"}
else:
    T = dict(bg="#0d1117", text="white", subtext="#888888",
             grid="#1e2530", na="#1a2030", gc="#0d1117")
    LIGHT_OV = {}

with open(BASE_DIR / "config" / "party_colours.json") as f:
    _raw = json.load(f)
party_color = {canon(k): v for k, v in _raw.items()}
party_color.setdefault("fraktionslos", "#888888")
party_color.setdefault("BSW", "#a020f0")
party_color.update(LIGHT_OV)

import matplotlib.colors as mcolors
cmap_J = mcolors.LinearSegmentedColormap.from_list(
    "J", ["#c0392b", T["na"], "#2471a3"], N=256)

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_party_votes(period_key):
    """Return (parties, PV) where PV is (n_parties, n_polls) array of ±1/NaN."""
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    pid_to_party = dict(zip(nodes["person_id"], nodes["party"]))

    poll_ids = [p["id"] for p in raw["polls"]]
    poll_idx = {pid: i for i, pid in enumerate(poll_ids)}

    # Accumulate party votes per poll
    acc = {}   # party -> poll_idx -> [+1/-1 values]
    for v in raw["votes"]:
        val = VOTE_MAP.get(v["vote"])
        if val is None: continue
        party = pid_to_party.get(v["mandate"]["id"])
        if not party: continue
        pi = poll_idx.get(v["poll"]["id"])
        if pi is None: continue
        acc.setdefault(party, {}).setdefault(pi, []).append(val)

    # Order parties
    present = [p for p in PARTY_ORDER if p in acc]
    for p in acc:
        if p not in present and p != "fraktionslos":
            present.append(p)

    n_p, n_v = len(present), len(poll_ids)
    PV = np.full((n_p, n_v), np.nan)
    for a, p in enumerate(present):
        for pi, votes in acc.get(p, {}).items():
            m = np.mean(votes)
            if m != 0:
                PV[a, pi] = np.sign(m)

    # Filter near-unanimous polls
    yes_frac = np.nanmean(PV == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    PV = PV[:, keep]

    return present, PV


# ══════════════════════════════════════════════════════════════════════════════
# Exact inverse Ising
# ══════════════════════════════════════════════════════════════════════════════

def all_states(n):
    return np.array(list(product([1.0, -1.0], repeat=n)))   # (2^n, n)

def model_moments(states, h, J):
    """Compute ⟨σ_a⟩ and ⟨σ_a σ_b⟩ from Boltzmann distribution."""
    E = -(states @ h) - 0.5 * np.einsum("ki,ij,kj->k", states, J, states)
    E -= E.min()                        # numerical stability
    w = np.exp(-E)
    Z = w.sum()
    p = w / Z
    m_mod = (p[:, None] * states).sum(0)
    C_mod = (p[:, None, None] * states[:, :, None] * states[:, None, :]).sum(0)
    return m_mod, C_mod, p

def inverse_ising(M_obs, C_obs, n_steps=4000, lr=0.05, lr_decay=0.9995):
    """
    Gradient ascent on exact log-likelihood.
    Returns h, J, final model moments.
    """
    n = len(M_obs)
    states = all_states(n)
    h = np.zeros(n)
    J = np.zeros((n, n))

    for step in range(n_steps):
        m_mod, C_mod, _ = model_moments(states, h, J)
        dh = M_obs - m_mod
        dJ = C_obs - C_mod
        np.fill_diagonal(dJ, 0.0)
        dJ = (dJ + dJ.T) / 2.0
        h += lr * dh
        J += lr * dJ
        lr *= lr_decay
        if step % 500 == 0:
            rmse = np.sqrt(np.mean(dh**2) + np.mean(dJ[np.triu_indices(n,1)]**2))
            if rmse < 5e-4:
                break

    m_mod, C_mod, p = model_moments(states, h, J)
    return h, J, m_mod, C_mod, p


# ══════════════════════════════════════════════════════════════════════════════
# T_eff from inverse-Ising J (direct arctanh estimator)
# ══════════════════════════════════════════════════════════════════════════════

def fit_T_eff(J, M):
    JM = J @ M
    valid = (np.abs(M) < 0.92) & (np.abs(JM) > 1e-4)
    if valid.sum() >= 2:
        betas = np.arctanh(M[valid]) / JM[valid]
        betas = betas[np.isfinite(betas) & (betas > 0)]
        if len(betas) >= 1:
            return float(1.0 / np.median(betas))
    def res(b): return np.sum((M - np.tanh(b * JM))**2)
    r = minimize_scalar(res, bounds=(0.01, 100.0), method="bounded")
    return float(1.0 / max(r.x, 0.01))

def critical_T(J):
    return float(np.linalg.eigvalsh(J).max())


# ══════════════════════════════════════════════════════════════════════════════
# Energy landscape
# ══════════════════════════════════════════════════════════════════════════════

def energy_landscape(states, h, J, parties):
    E = -(states @ h) - 0.5 * np.einsum("ki,ij,kj->k", states, J, states)
    # Local minima: no single-spin flip lowers energy
    n = len(h)
    local_min = []
    for idx in range(len(states)):
        is_min = True
        for i in range(n):
            flipped = states[idx].copy(); flipped[i] *= -1
            E_flip = -(flipped @ h) - 0.5 * flipped @ J @ flipped
            if E_flip < E[idx]:
                is_min = False; break
        if is_min:
            config = {parties[i]: int(states[idx, i]) for i in range(n)}
            local_min.append(dict(energy=E[idx], config=config))
    return E, sorted(local_min, key=lambda x: x["energy"])

def n_valleys(local_min):
    return len(local_min)


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap T_eff
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_T_eff(parties, PV, n_boot=200, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    n_polls = PV.shape[1]
    T_boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_polls, size=n_polls)
        PV_b = PV[:, idx]
        complete = ~np.isnan(PV_b).any(axis=0)
        if complete.sum() < 5: continue
        S = PV_b[:, complete]
        M_b = S.mean(1)
        C_b = (S[:, None, :] * S[None, :, :]).mean(2)
        try:
            _, J_b, *_ = inverse_ising(M_b, C_b, n_steps=2000, lr=0.05)
            T_boots.append(fit_T_eff(J_b, M_b))
        except Exception:
            pass
    return np.array(T_boots)


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

results = []
print("Fitting inverse Ising per period…")

for pk, lbl in PERIODS:
    print(f"\n{lbl}")
    try:
        parties, PV = load_party_votes(pk)
        complete = ~np.isnan(PV).any(axis=0)
        S = PV[:, complete]
        n_p, n_v = S.shape
        print(f"  parties={n_p}  complete_polls={n_v}")

        M_obs = S.mean(1)
        C_obs = (S[:, None, :] * S[None, :, :]).mean(2)

        h, J, m_mod, C_mod, probs = inverse_ising(M_obs, C_obs)
        rmse_m = np.sqrt(np.mean((M_obs - m_mod)**2))
        rmse_C = np.sqrt(np.mean((C_obs - C_mod)[np.triu_indices(n_p,1)]**2))
        print(f"  fit RMSE: M={rmse_m:.4f}  C={rmse_C:.4f}")

        T_eff = fit_T_eff(J, M_obs)
        T_c   = critical_T(J)
        print(f"  T_eff={T_eff:.3f}  T_c={T_c:.3f}  ratio={T_eff/T_c:.2f}")

        states = all_states(n_p)
        E, local_min = energy_landscape(states, h, J, parties)
        n_v_land = n_valleys(local_min)
        print(f"  valleys={n_v_land}")

        print("  bootstrap…")
        boots = bootstrap_T_eff(parties, PV, n_boot=300)
        T_lo, T_hi = np.percentile(boots, [5, 95]) if len(boots) > 10 else (T_eff, T_eff)

        results.append(dict(
            period=lbl, parties=parties, n_parties=n_p,
            M_obs=M_obs, C_obs=C_obs, J=J, h=h,
            T_eff=T_eff, T_c=T_c, T_lo=T_lo, T_hi=T_hi,
            n_valleys=n_v_land, local_min=local_min,
            E=E, probs=probs, states=states,
        ))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure: cooling curve + J heatmaps + valley count
# ══════════════════════════════════════════════════════════════════════════════

n_per = len(results)
xi    = np.arange(n_per)
lbls  = [r["period"] for r in results]

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(T["bg"])

gs_top = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                           top=0.93, bottom=0.62, wspace=0.32)
gs_bot = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97,
                           top=0.54, bottom=0.04, hspace=0.55, wspace=0.38)

# ── Panel A: cooling curve ────────────────────────────────────────────────────
ax_cool = fig.add_subplot(gs_top[0])
ax_cool.set_facecolor(T["bg"])

T_eff_arr = np.array([r["T_eff"] for r in results])
T_c_arr   = np.array([r["T_c"]   for r in results])
T_lo_arr  = np.array([r["T_lo"]  for r in results])
T_hi_arr  = np.array([r["T_hi"]  for r in results])

C_teff = "#4a9fd4" if not LIGHT_MODE else "#2471a3"
C_tc   = "#e05c4a" if not LIGHT_MODE else "#c0392b"

ax_cool.fill_between(xi, T_lo_arr, T_hi_arr, color=C_teff, alpha=0.18, zorder=2)
ax_cool.plot(xi, T_eff_arr, color=C_teff, lw=2.2, marker="o", ms=7,
             label="T_eff (fitted, exact J)", zorder=4)
ax_cool.plot(xi, T_c_arr,   color=C_tc,   lw=2.2, marker="s", ms=7,
             ls="--", label="T_c = λ_max(J)", zorder=4)
ax_cool.fill_between(xi, T_eff_arr, T_c_arr,
                     where=T_eff_arr < T_c_arr,
                     color=C_tc, alpha=0.10, zorder=1)

for i, r in enumerate(results):
    ax_cool.text(i, max(r["T_eff"], r["T_c"]) + 0.06,
                 f"{r['T_eff']/r['T_c']:.2f}",
                 ha="center", va="bottom", fontsize=7, color=T["subtext"])

for v in np.arange(0, T_c_arr.max() + 0.5, 0.2):
    ax_cool.axhline(v, color=T["grid"], lw=0.4, zorder=0)
ax_cool.set_xticks(xi); ax_cool.set_xticklabels(lbls, color=T["text"], fontsize=9, fontweight="bold")
ax_cool.tick_params(length=0, pad=6); ax_cool.tick_params(axis="y", labelsize=7, colors=T["subtext"])
ax_cool.set_xlim(-0.5, n_per - 0.3)
ax_cool.set_ylim(0, T_c_arr.max() + 0.5)
ax_cool.spines[:].set_visible(False)
leg = ax_cool.legend(frameon=False, fontsize=8, loc="upper right")
for t in leg.get_texts(): t.set_color(T["subtext"])
ax_cool.text(0, 1.08, "A  Cooling curve (exact inverse Ising J)",
             transform=ax_cool.transAxes, color=T["text"], fontsize=10,
             fontweight="bold", va="top")
ax_cool.text(0.5, -0.08, "Ratio T_eff/T_c shown above each period",
             transform=ax_cool.transAxes, color=T["subtext"], fontsize=7,
             ha="center", va="top")

# ── Panel B: valley count ─────────────────────────────────────────────────────
ax_val = fig.add_subplot(gs_top[1])
ax_val.set_facecolor(T["bg"])

n_val_arr = [r["n_valleys"] for r in results]
bar_col   = C_teff
ax_val.bar(xi, n_val_arr, color=bar_col, alpha=0.7, width=0.5, zorder=3)
for i, v in enumerate(n_val_arr):
    ax_val.text(i, v + 0.05, str(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=T["text"])

# Annotate the dominant valley configs
for i, r in enumerate(results):
    if r["local_min"]:
        bottom = r["local_min"][0]
        cfg_str = " ".join(
            ("+" if bottom["config"].get(p, 0) > 0 else "–")
            for p in r["parties"]
        )
        ax_val.text(i, -0.3, cfg_str, ha="center", va="top",
                    fontsize=5, color=T["subtext"], clip_on=False)

for v in range(0, max(n_val_arr) + 2):
    ax_val.axhline(v, color=T["grid"], lw=0.4, zorder=0)
ax_val.set_xticks(xi); ax_val.set_xticklabels(lbls, color=T["text"], fontsize=9, fontweight="bold")
ax_val.tick_params(length=0, pad=6); ax_val.tick_params(axis="y", labelsize=7, colors=T["subtext"])
ax_val.set_xlim(-0.5, n_per - 0.5); ax_val.set_ylim(0, max(n_val_arr) + 1)
ax_val.spines[:].set_visible(False)
ax_val.text(0, 1.08, "B  Energy landscape valleys per period",
            transform=ax_val.transAxes, color=T["text"], fontsize=10, fontweight="bold", va="top")
ax_val.text(0.5, -0.08, "Local energy minima = stable voting configurations",
            transform=ax_val.transAxes, color=T["subtext"], fontsize=7, ha="center", va="top")

# ── Panels C-H: J heatmaps ───────────────────────────────────────────────────
for idx, r in enumerate(results):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs_bot[row, col])
    ax.set_facecolor(T["bg"])

    parties = r["parties"]
    n_p = r["n_parties"]
    J   = r["J"]
    C   = r["C_obs"]
    tick_lbls = [DISPLAY.get(p, p) for p in parties]

    # Show J (true couplings) — key insight vs C
    vmax = max(abs(J[np.triu_indices(n_p, 1)]).max(), 0.5)
    im = ax.imshow(J, aspect="auto", cmap=cmap_J,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Annotate: J value; circle if opposite sign from C
    for i in range(n_p):
        for j in range(n_p):
            v_J = J[i, j]
            v_C = C[i, j] if i != j else np.nan
            txt = f"{v_J:.2f}"
            norm_v = (v_J + vmax) / (2 * vmax)
            bg_col = cmap_J(norm_v)
            lum = 0.299*bg_col[0] + 0.587*bg_col[1] + 0.114*bg_col[2]
            tc = "#111111" if lum > 0.5 else "#eeeeee"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5, color=tc)
            # Star if J and C have opposite signs (hidden antagonism)
            if i != j and not np.isnan(v_C):
                if np.sign(v_J) != np.sign(v_C) and abs(v_J) > 0.05:
                    ax.text(j + 0.35, i - 0.35, "✱", fontsize=4.5,
                            color="#ffcc00", ha="center", va="center")

    for k in range(n_p + 1):
        ax.axhline(k - 0.5, color=T["gc"], lw=0.6)
        ax.axvline(k - 0.5, color=T["gc"], lw=0.6)

    ax.set_xticks(range(n_p))
    ax.set_xticklabels(tick_lbls, fontsize=5, rotation=45, ha="right")
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(tick_lbls, fontsize=5)
    ax.tick_params(length=0, pad=1)
    for tick, p in zip(ax.get_xticklabels(), parties):
        tick.set_color(party_color.get(p, T["text"]))
    for tick, p in zip(ax.get_yticklabels(), parties):
        tick.set_color(party_color.get(p, T["text"]))
    ax.spines[:].set_visible(False)
    ax.set_title(r["period"], color=T["text"], fontsize=9, fontweight="bold", pad=4)

# ── Shared legend / annotation ────────────────────────────────────────────────
fig.text(0.06, 0.975,
         "Inverse Ising model fit to Bundestag party voting, 2005–2029",
         color=T["text"], fontsize=14, fontweight="bold", va="top")
fig.text(0.06, 0.955,
         "C–H: True couplings J_ij (not raw correlations C_ij)  ·  "
         "✱ = hidden antagonism: J < 0 but C > 0  ·  "
         "Removing indirect paths reveals genuine interaction sign",
         color=T["subtext"], fontsize=8, va="top")
out_path = IMG_DIR / f"mp_ising_exact.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2,
            facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
