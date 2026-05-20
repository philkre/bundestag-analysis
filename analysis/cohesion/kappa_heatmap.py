#!/usr/bin/env python3
"""
Party × party mean kappa heatmap grid — 2 rows × 3 cols, one panel per period.

Colour scale: RdBu  (red = hostile / negative kappa, blue = aligned / positive).
Party order fixed to the known German left-right spectrum.
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Canonical names — merge spelling variants
CANONICAL = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
    "Bündnis 90/Die Grünen": "Grüne",
    "DIE GRÜNEN":            "Grüne",
    "DIE LINKE":             "Die Linke",
    "DIE LINKE.":            "Die Linke",
    "Die Linke.":            "Die Linke",
    "fraktionslos":          None,
}

# Fixed left-to-right display order
LR_ORDER = ["Die Linke", "Grüne", "SPD", "BSW", "FDP", "CDU/CSU", "AfD"]


def canonicalise(party: str):
    return CANONICAL.get(party, party)


def resolve_color(party: str, colors: dict) -> str:
    if party in colors:
        return colors[party]
    for key, val in colors.items():
        if party.startswith(key) or key.startswith(party):
            return val
    return "#888888"


# ── Data loading ──────────────────────────────────────────────────────────────

def load(period_dir):
    nodes_path = ROOT / "output" / period_dir / "nodes.csv"
    edges_path = ROOT / "output" / period_dir / "edges.csv"
    with open(nodes_path) as f:
        nodes = {}
        for r in csv.DictReader(f):
            p = canonicalise(r["party"].strip())
            if p:
                nodes[r["person_id"].strip()] = p
    kappas = {}
    with open(edges_path) as f:
        for r in csv.DictReader(f):
            kappas[(r["source"].strip(), r["target"].strip())] = float(r["weight"])
    return nodes, kappas


# ── Party × party mean kappa matrix ──────────────────────────────────────────

def party_kappa_matrix(nodes: dict, kappas: dict) -> tuple[list[str], np.ndarray]:
    """
    Returns (party_list, matrix) where matrix[i,j] = mean kappa between
    parties party_list[i] and party_list[j].  Diagonal = NaN (not shown).
    Parties are sorted by LR_ORDER; unlisted parties appended alphabetically.
    """
    present = sorted(set(nodes.values()))
    ordered = [p for p in LR_ORDER if p in present]
    ordered += sorted(p for p in present if p not in LR_ORDER)
    n = len(ordered)
    idx = {p: i for i, p in enumerate(ordered)}

    sums   = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)

    for (src, tgt), kappa in kappas.items():
        pa = nodes.get(src)
        pb = nodes.get(tgt)
        if pa is None or pb is None or pa == pb:
            continue
        i, j = idx.get(pa), idx.get(pb)
        if i is None or j is None:
            continue
        sums[i, j]   += kappa;  sums[j, i]   += kappa
        counts[i, j] += 1;      counts[j, i] += 1

    matrix = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    np.fill_diagonal(matrix, np.nan)
    return ordered, matrix


# ── Plot ──────────────────────────────────────────────────────────────────────

def main():
    with open(ROOT / "config" / "party_colours.json") as f:
        colors = json.load(f)

    all_parties_list, all_matrices, period_labels = [], [], []
    for period_dir, label in PERIODS:
        nodes, kappas = load(period_dir)
        parties, matrix = party_kappa_matrix(nodes, kappas)
        all_parties_list.append(parties)
        all_matrices.append(matrix)
        period_labels.append(label)
        print(f"{label}: {parties}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 11),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Party voting similarity (mean Cohen's κ) per Bundestag period",
        color="white", fontsize=13, y=0.98,
    )

    # Shared colour scale: symmetric around 0
    vmax = 0.6
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu")     # red = hostile, blue = aligned

    axes_flat = list(axes.flat)
    for ax, parties, matrix, label in zip(
        axes_flat, all_parties_list, all_matrices, period_labels
    ):
        ax.set_facecolor("#0d1117")

        # Draw heatmap
        n = len(parties)
        im = ax.imshow(matrix, cmap=cmap, norm=norm,
                       aspect="equal", interpolation="nearest")

        # Axis labels with party colours
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(parties, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(parties, fontsize=8)

        for tick, party in zip(ax.get_xticklabels(), parties):
            tick.set_color(resolve_color(party, colors))
        for tick, party in zip(ax.get_yticklabels(), parties):
            tick.set_color(resolve_color(party, colors))

        # Annotate cells with kappa value
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(val) > 0.3 else "#aaa")

        ax.set_title(label, color="white", fontsize=11, pad=6)
        ax.tick_params(colors="#888", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared colour bar — placed to the right of the grid
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("Mean Cohen's κ\n← hostile  |  aligned →",
                   color="white", fontsize=9, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.outline.set_edgecolor("#333")
    out = ROOT / "output" / "img" / "kappa_heatmap.png"
    (ROOT / "output" / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
