#!/usr/bin/env python3
"""
Party-level MDS-based ideological positioning in the Bundestag over time.

1. Aggregate MP-pair kappas → party × party mean kappa matrix per period
2. Classical MDS (double-centring + eigendecomposition) → 1D raw positions
3. Orient: majority-vote across left/right reference pairs
4. Normalise: linear transform so Die Linke = −1, CDU/CSU = +1
   → scale is comparable across periods, anchored to the classic left-right axis
5. Plot party drift chart — dark theme, party colours
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]

# Canonical party names — merge variants that refer to the same party
CANONICAL = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
    "Bündnis 90/Die Grünen": "Grüne",
    "DIE GRÜNEN":            "Grüne",
    "DIE LINKE":             "Die Linke",
    "DIE LINKE.":            "Die Linke",
    "Die Linke.":            "Die Linke",
    "fraktionslos":          "fraktionslos",
}

# Reference pairs for axis orientation (all should satisfy left < right)
LEFT_REFS  = ["Die Linke", "Grüne", "SPD"]
RIGHT_REFS = ["CDU/CSU", "AfD", "FDP"]


# ── Data loading ──────────────────────────────────────────────────────────────

def canonicalise(party: str) -> str:
    return CANONICAL.get(party, party)


def load(period_dir):
    nodes_path = ROOT / "output" / period_dir / "nodes.csv"
    edges_path = ROOT / "output" / period_dir / "edges.csv"
    with open(nodes_path) as f:
        nodes = {r["person_id"].strip(): canonicalise(r["party"].strip())
                 for r in csv.DictReader(f)}
    kappas = {}
    with open(edges_path) as f:
        for r in csv.DictReader(f):
            kappas[(r["source"].strip(), r["target"].strip())] = float(r["weight"])
    return nodes, kappas


# ── Classical MDS ─────────────────────────────────────────────────────────────

def classical_mds_1d(D: np.ndarray) -> np.ndarray:
    """Double-centring → top eigenvector → 1D projection."""
    D2 = D.astype(np.float64) ** 2
    row_mean   = D2.mean(axis=1, keepdims=True)
    col_mean   = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + total_mean)
    eigvals, eigvecs = np.linalg.eigh(B)           # ascending order
    top = np.argmax(eigvals)
    return eigvecs[:, top] * np.sqrt(max(eigvals[top], 0.0))


# ── Position computation ──────────────────────────────────────────────────────

def compute_party_positions(nodes: dict, kappas: dict) -> dict:
    """
    Aggregate cross-party MP-pair kappas → party × party mean kappa matrix
    → 1D classical MDS. Returns {canonical_party: float_position}.
    """
    party_of = {pid: p for pid, p in nodes.items() if p != "fraktionslos"}
    parties  = sorted(set(party_of.values()))
    p_idx    = {p: i for i, p in enumerate(parties)}
    n        = len(parties)

    sums   = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)

    for (src, tgt), kappa in kappas.items():
        pa = party_of.get(src)
        pb = party_of.get(tgt)
        if pa is None or pb is None or pa == pb:
            continue
        i, j = p_idx[pa], p_idx[pb]
        sums[i, j]   += kappa;  sums[j, i]   += kappa
        counts[i, j] += 1;      counts[j, i] += 1

    mean_k = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    np.fill_diagonal(mean_k, 1.0)
    D = np.clip(1.0 - mean_k, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)

    pos = classical_mds_1d(D)
    return {p: float(pos[p_idx[p]]) for p in parties}


# ── Orientation ───────────────────────────────────────────────────────────────

def orient(positions: dict) -> dict:
    """Majority-vote: left references should be < right references."""
    keep = flip = 0
    for lp in LEFT_REFS:
        for rp in RIGHT_REFS:
            if lp in positions and rp in positions:
                if positions[lp] < positions[rp]:
                    keep += 1
                else:
                    flip += 1
    if flip > keep:
        return {p: -v for p, v in positions.items()}
    return positions


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_std(positions: dict) -> dict:
    """
    Centre at 0 and scale to unit SD across party positions.
    Robust to coalition structure changes — always produces a comparable spread.
    """
    vals = np.array(list(positions.values()))
    mean, std = float(vals.mean()), float(vals.std())
    if std < 1e-12:
        return positions
    return {p: (v - mean) / std for p, v in positions.items()}


# ── Colour helper ─────────────────────────────────────────────────────────────

def resolve_color(party: str, colors: dict) -> str:
    if party in colors:
        return colors[party]
    for key, val in colors.items():
        if party.startswith(key) or key.startswith(party):
            return val
    return "#888888"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(ROOT / "config" / "party_colours.json") as f:
        colors = json.load(f)

    all_positions: list[dict] = []
    period_labels: list[str]  = []

    for period_dir, label in PERIODS:
        print(f"Computing {label}…", end=" ", flush=True)
        nodes, kappas = load(period_dir)
        pos = compute_party_positions(nodes, kappas)
        pos = orient(pos)
        pos = normalise_std(pos)
        print({p: round(v, 3) for p, v in sorted(pos.items())})
        all_positions.append(pos)
        period_labels.append(label)

    # Gather parties present in ≥ 2 periods, skip fraktionslos
    party_count: dict[str, int] = {}
    for pos in all_positions:
        for p in pos:
            party_count[p] = party_count.get(p, 0) + 1
    parties = [p for p, cnt in party_count.items()
               if cnt >= 2 and p != "fraktionslos"]

    x = np.arange(len(period_labels))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white", labelsize=11)
    ax.spines[:].set_color("#333")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    for party in sorted(parties):
        color = resolve_color(party, colors)

        xs, vs = [], []
        for i, pos in enumerate(all_positions):
            if party in pos:
                xs.append(i)
                vs.append(pos[party])
        if len(xs) < 2:
            continue

        ax.plot(xs, vs, "o-", color=color, lw=2.5, ms=7, label=party, zorder=3)

    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, color="white", fontsize=11)
    ax.set_ylabel("Voting position (SD-normalised per period, left < 0 < right)",
                  color="white", fontsize=10)
    ax.set_title(
        "Party positions in the Bundestag over time\n"
        "Party-level 1D MDS on mean MP-pair Cohen's κ",
        color="white", fontsize=12, pad=14,
    )
    ax.legend(facecolor="#1a1a2e", labelcolor="white",
              framealpha=0.8, fontsize=10, loc="upper left")
    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    out = ROOT / "output" / "img" / "mds_drift.png"
    (ROOT / "output" / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
