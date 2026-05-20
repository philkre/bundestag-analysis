#!/usr/bin/env python3
"""
Coalition cohesion deep-dive.

Top:    2×3 mini network diagrams — one per period.
        Nodes  = coalition parties (party colour, size ∝ MP count).
        Edges  = within-coalition mean kappa
                 (thickness + colour: gold=tight, red=loose).

Bottom: time series of every recurring coalition pair's kappa,
        anchoring CDU↔SPD as the main narrative thread.
"""

import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
BG   = "#0d1117"

PERIODS = [
    ("bundestag_2005_2009", "2005–09", ["CDU/CSU", "SPD"]),
    ("bundestag_2009_2013", "2009–13", ["CDU/CSU", "FDP"]),
    ("bundestag_2013_2017", "2013–17", ["CDU/CSU", "SPD"]),
    ("bundestag_2017_2021", "2017–21", ["CDU/CSU", "SPD"]),
    ("bundestag_2021_2025", "2021–25", ["SPD", "Grüne", "FDP"]),
    ("bundestag_2025_2029", "2025–29", ["CDU/CSU", "SPD"]),
]

CANONICAL = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne", "Bündnis 90/Die Grünen": "Grüne",
    "DIE GRÜNEN": "Grüne", "DIE LINKE": "Die Linke",
    "DIE LINKE.": "Die Linke", "Die Linke.": "Die Linke",
}
def canon(p): return CANONICAL.get(p, p)

# Edge colour: gold at κ=1, desaturated orange/red as κ falls
EDGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "coal", ["#c0392b", "#e67e22", "#f1c40f", "#FFD700"], N=256
)
EDGE_NORM = mcolors.Normalize(vmin=0.45, vmax=1.0)


def resolve_color(party, colors):
    if party in colors: return colors[party]
    for k, v in colors.items():
        if party.startswith(k) or k.startswith(party): return v
    return "#888"


def load(period_dir):
    with open(ROOT / "output" / period_dir / "nodes.csv") as f:
        nodes = {r["person_id"].strip(): canon(r["party"].strip())
                 for r in csv.DictReader(f)}
    with open(ROOT / "output" / period_dir / "edges.csv") as f:
        edges = [(r["source"].strip(), r["target"].strip(), float(r["weight"]))
                 for r in csv.DictReader(f)]
    return nodes, edges


def coalition_pair_kappas(nodes, edges, coalition):
    """Return {frozenset(pa,pb): mean_kappa} for all intra-coalition pairs."""
    sums, counts = defaultdict(float), defaultdict(int)
    for src, tgt, w in edges:
        pa, pb = nodes.get(src), nodes.get(tgt)
        if not pa or not pb or pa == pb: continue
        if pa in coalition and pb in coalition:
            key = frozenset([pa, pb])
            sums[key] += w; counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}


def node_positions(parties):
    """Place 2 or 3 nodes nicely centred at origin."""
    n = len(parties)
    if n == 2:
        return {parties[0]: (-0.45, 0.0), parties[1]: (0.45, 0.0)}
    # equilateral triangle, point up
    angles = [np.pi/2 + i * 2*np.pi/3 for i in range(3)]
    r = 0.40
    return {p: (r * np.cos(a), r * np.sin(a))
            for p, a in zip(parties, angles)}


def draw_coalition_panel(ax, nodes, edges, coalition, label, colors):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    sizes   = Counter(v for v in nodes.values() if v in coalition)
    max_mps = max(sizes.values()) if sizes else 1
    kappas  = coalition_pair_kappas(nodes, edges, coalition)
    pos     = node_positions(sorted(coalition))

    # ── Edges ─────────────────────────────────────────────────────────────────
    for pair, kappa in kappas.items():
        pa, pb  = sorted(pair)
        x0, y0  = pos[pa]
        x1, y1  = pos[pb]
        ecol    = EDGE_CMAP(EDGE_NORM(kappa))
        lw      = 2 + 10 * (kappa - 0.45) / 0.55       # 2–12 pt

        ax.plot([x0, x1], [y0, y1], color=ecol, lw=lw,
                solid_capstyle="round", zorder=1, alpha=0.9)

        # Kappa label at edge midpoint, slightly offset perpendicular
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx, dy = x1-x0, y1-y0
        length = np.hypot(dx, dy) or 1
        ox, oy = -dy/length * 0.10, dx/length * 0.10  # perpendicular offset
        ax.text(mx + ox, my + oy, f"κ = {kappa:.3f}",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=4)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for party, (x, y) in pos.items():
        n_mps  = sizes.get(party, 50)
        radius = 0.10 + 0.14 * np.sqrt(n_mps / max_mps)
        pcolor = resolve_color(party, colors)

        circle = plt.Circle((x, y), radius, color=pcolor, zorder=2)
        ax.add_patch(circle)

        # Party name inside (or just below for small circles)
        ax.text(x, y, party, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=3)
        ax.text(x, y - radius - 0.06, f"{n_mps} MPs",
                ha="center", va="top", fontsize=5.5, color="#888", zorder=3)

    ax.set_title(label, color="white", fontsize=11, pad=6)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.80, 0.85)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(ROOT / "config" / "party_colours.json") as f:
        colors = json.load(f)

    # Collect data for all periods
    all_kappas = {}
    period_labels = []
    for i, (period_dir, label, coalition_raw) in enumerate(PERIODS):
        nodes, edges = load(period_dir)
        coalition = set(coalition_raw)
        all_kappas[i] = {
            "label":     label,
            "coalition": coalition,
            "pairs":     coalition_pair_kappas(nodes, edges, coalition),
            "sizes":     Counter(v for v in nodes.values() if v in coalition),
            "nodes":     nodes,
            "edges":     edges,
        }
        period_labels.append(label)

    # ── Figure layout: 2×3 panels + bottom time series ────────────────────────
    fig = plt.figure(figsize=(15, 12), facecolor=BG)
    gs  = fig.add_gridspec(3, 3,
                           height_ratios=[1, 1, 0.75],
                           hspace=0.45, wspace=0.15)

    # Top 6 panels
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax, (i, d) in zip(axes, all_kappas.items()):
        draw_coalition_panel(ax, d["nodes"], d["edges"],
                             d["coalition"], d["label"], colors)

    # ── Bottom: pair kappa time series ────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[2, :])
    ax_ts.set_facecolor(BG)
    ax_ts.tick_params(colors="white", labelsize=9)
    ax_ts.spines[:].set_color("#333")

    # Build per-pair series
    pair_series: dict[tuple, list] = defaultdict(list)
    for i in range(len(PERIODS)):
        for pair, kappa in all_kappas[i]["pairs"].items():
            pair_series[tuple(sorted(pair))].append((i, kappa))

    for pair, pts in sorted(pair_series.items()):
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        # Use first party's colour; CDU↔SPD gets extra weight
        c1, c2   = pair
        is_main  = pair == ("CDU/CSU", "SPD")
        col      = resolve_color(c1, colors)
        lw       = 3.0 if is_main else 1.8
        ls       = "-"
        alpha    = 1.0 if is_main else 0.65
        label    = f"{c1} ↔ {c2}"

        ax_ts.plot(xs, ys, "o-", color=col, lw=lw, ms=7,
                   ls=ls, alpha=alpha, label=label, zorder=3 if is_main else 2)
        for x, y in zip(xs, ys):
            ax_ts.text(x, y + 0.012, f"{y:.2f}",
                       ha="center", va="bottom", fontsize=7,
                       color=col, alpha=alpha)

    ax_ts.set_xticks(range(len(PERIODS)))
    ax_ts.set_xticklabels(period_labels, color="white", fontsize=9)
    ax_ts.set_ylabel("Within-coalition κ", color="white", fontsize=9)
    ax_ts.set_ylim(0.40, 1.08)
    ax_ts.axhline(1.0, color="#333", lw=0.7, ls="--")
    ax_ts.grid(axis="y", color="#333", ls="--", lw=0.6)
    ax_ts.legend(facecolor="#1a1a2e", labelcolor="white",
                 fontsize=8, loc="lower right", framealpha=0.8)
    ax_ts.set_title("Within-coalition pair kappa over time",
                    color="white", fontsize=10, pad=6)

    fig.suptitle("Coalition cohesion — how tightly do coalition partners vote together?",
                 color="white", fontsize=13, y=0.99)

    out = ROOT / "output" / "img" / "coalition_cohesion.png"
    (ROOT / "output" / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
