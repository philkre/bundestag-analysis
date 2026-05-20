#!/usr/bin/env python3
"""
Compute and plot a polarisation time series across all Bundestag periods.

Metrics (all computed from raw, unfiltered edges):
  - gov_opp_kappa   : mean kappa between coalition and opposition MPs  ← main polarisation signal
  - coalition_kappa : mean kappa within coalition partners
  - intra_kappa     : mean kappa within each party (party discipline)
  - cross_party_kappa: mean kappa across ALL cross-party pairs
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09", ["CDU/CSU", "SPD"]),
    ("bundestag_2009_2013", "2009–13", ["CDU/CSU", "FDP"]),
    ("bundestag_2013_2017", "2013–17", ["CDU/CSU", "SPD"]),
    ("bundestag_2017_2021", "2017–21", ["CDU/CSU", "SPD"]),
    ("bundestag_2021_2025", "2021–25", ["SPD", "BÜNDNIS 90/DIE GRÜNEN", "FDP"]),
    ("bundestag_2025_2029", "2025–29", ["CDU/CSU", "SPD"]),
]


def load(period_dir):
    nodes_path = ROOT / "output" / period_dir / "nodes.csv"
    edges_path = ROOT / "output" / period_dir / "edges.csv"
    with open(nodes_path) as f:
        nodes = {r["person_id"].strip(): r["party"].strip() for r in csv.DictReader(f)}
    with open(edges_path) as f:
        edges = [
            (r["source"].strip(), r["target"].strip(), float(r["weight"]))
            for r in csv.DictReader(f)
        ]
    return nodes, edges


def compute_metrics(nodes, edges, coalition_parties):
    coalition = set(coalition_parties)

    intra, cross_all, gov_opp, coal_internal = [], [], [], []

    for src, tgt, w in edges:
        pa = nodes.get(src)
        pb = nodes.get(tgt)
        if not pa or not pb:
            continue

        if pa == pb:
            intra.append(w)
        else:
            cross_all.append(w)
            src_in = pa in coalition
            tgt_in = pb in coalition
            if src_in and tgt_in:
                coal_internal.append(w)
            elif src_in != tgt_in:  # one in, one out
                gov_opp.append(w)

    def mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "intra_kappa":      mean(intra),
        "cross_party_kappa": mean(cross_all),
        "coalition_kappa":  mean(coal_internal),
        "gov_opp_kappa":    mean(gov_opp),
        "n_cross":          len(cross_all),
        "n_gov_opp":        len(gov_opp),
    }


def main():
    labels = []
    metrics_list = []

    for period_dir, label, coalition in PERIODS:
        print(f"Loading {label}…", end=" ", flush=True)
        nodes, edges = load(period_dir)
        m = compute_metrics(nodes, edges, coalition)
        print(f"cross={m['n_cross']:,}  gov↔opp={m['n_gov_opp']:,}  "
              f"gov↔opp kappa={m['gov_opp_kappa']:.3f}")
        labels.append(label)
        metrics_list.append(m)

    x = np.arange(len(labels))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── Panel 1: polarisation signal ──────────────────────────────────────────
    ax = axes[0]

    gov_opp   = [m["gov_opp_kappa"]    for m in metrics_list]
    cross_all = [m["cross_party_kappa"] for m in metrics_list]
    coal_int  = [m["coalition_kappa"]  for m in metrics_list]

    ax.plot(x, gov_opp,   "o-", color="#E3000F", lw=2.5, ms=7,
            label="Gov ↔ Opposition (polarisation)")
    ax.plot(x, cross_all, "o--", color="#888888", lw=1.5, ms=5,
            label="All cross-party")
    ax.plot(x, coal_int,  "o--", color="#FFEF00", lw=1.5, ms=5,
            label="Within coalition")

    ax.set_ylabel("Mean kappa (voting similarity)", color="white")
    ax.set_title("Polarisation in the Bundestag: Gov ↔ Opposition voting similarity", color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.8, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.7)

    # Shade alternating coalition periods
    for i, (_, _, coalition) in enumerate(PERIODS):
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="white")

    # ── Panel 2: party discipline (intra-party kappa) ─────────────────────────
    ax2 = axes[1]
    intra = [m["intra_kappa"] for m in metrics_list]
    ax2.plot(x, intra, "o-", color="#1AA037", lw=2.5, ms=7,
             label="Intra-party (discipline)")
    ax2.set_ylabel("Mean kappa", color="white")
    ax2.set_title("Party discipline over time", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.8, fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax2.grid(axis="y", color="#333", linestyle="--", linewidth=0.7)
    for i in range(len(labels)):
        ax2.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="white")

    plt.xticks(x, labels, color="white")
    plt.tight_layout()

    out = ROOT / "output" / "img" / "polarisation.png"
    (ROOT / "output" / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
