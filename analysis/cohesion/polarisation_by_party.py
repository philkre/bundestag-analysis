#!/usr/bin/env python3
"""
Decompose the gov ↔ opposition kappa by opposition party.

For each period, for each party currently in opposition, compute the mean
kappa between all coalition MPs and that opposition party's MPs.

Produces output/polarisation_by_party.png — one line per party (shown only
in periods when that party is in opposition), plus the aggregate gov↔opp
line for reference.
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
    ("bundestag_2005_2009", "2005–09", ["CDU/CSU", "SPD"]),
    ("bundestag_2009_2013", "2009–13", ["CDU/CSU", "FDP"]),
    ("bundestag_2013_2017", "2013–17", ["CDU/CSU", "SPD"]),
    ("bundestag_2017_2021", "2017–21", ["CDU/CSU", "SPD"]),
    ("bundestag_2021_2025", "2021–25", ["SPD", "BÜNDNIS 90/DIE GRÜNEN", "FDP"]),
    ("bundestag_2025_2029", "2025–29", ["CDU/CSU", "SPD"]),
]

CANONICAL = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
    "Bündnis 90/Die Grünen": "Grüne",
    "DIE GRÜNEN":            "Grüne",
    "DIE LINKE":             "Die Linke",
    "DIE LINKE.":            "Die Linke",
    "Die Linke.":            "Die Linke",
}

# Canonical names for coalition lists too
COALITION_CANONICAL = {
    "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
}


def canon(party: str) -> str:
    return CANONICAL.get(party, party)


def canon_coalition(parties: list) -> set:
    return {COALITION_CANONICAL.get(p, p) for p in parties}


def load(period_dir):
    nodes_path = ROOT / "output" / period_dir / "nodes.csv"
    edges_path = ROOT / "output" / period_dir / "edges.csv"
    with open(nodes_path) as f:
        nodes = {r["person_id"].strip(): canon(r["party"].strip())
                 for r in csv.DictReader(f)}
    with open(edges_path) as f:
        edges = [
            (r["source"].strip(), r["target"].strip(), float(r["weight"]))
            for r in csv.DictReader(f)
        ]
    return nodes, edges


def gov_opp_by_party(nodes, edges, coalition):
    """
    Returns {opp_party: mean_kappa} for all opposition parties,
    plus "ALL" for the aggregate gov↔opp kappa.
    """
    by_opp: dict[str, list] = {}
    all_gov_opp: list = []

    for src, tgt, w in edges:
        pa = nodes.get(src)
        pb = nodes.get(tgt)
        if not pa or not pb or pa == pb:
            continue
        src_in = pa in coalition
        tgt_in = pb in coalition
        if src_in == tgt_in:
            continue                          # both in or both out — skip
        opp_party = pb if src_in else pa      # whichever is in opposition
        by_opp.setdefault(opp_party, []).append(w)
        all_gov_opp.append(w)

    result = {p: float(np.mean(v)) for p, v in by_opp.items()}
    result["__all__"] = float(np.mean(all_gov_opp)) if all_gov_opp else float("nan")
    return result


def resolve_color(party: str, colors: dict) -> str:
    if party in colors:
        return colors[party]
    for key, val in colors.items():
        if party.startswith(key) or key.startswith(party):
            return val
    return "#888888"


def main():
    with open(ROOT / "config" / "party_colours.json") as f:
        colors = json.load(f)

    period_labels = []
    # {party: [kappa_or_nan per period]}
    party_series: dict[str, list] = {}
    agg_series: list = []

    # First pass: collect all parties that appear in opposition at least once
    all_opp_parties: set = set()
    for period_dir, label, coalition_raw in PERIODS:
        nodes, edges = load(period_dir)
        coalition = canon_coalition(coalition_raw)
        result = gov_opp_by_party(nodes, edges, coalition)
        for p in result:
            if p != "__all__":
                all_opp_parties.add(p)

    # Second pass: build full series (NaN when party is in government or absent)
    for period_dir, label, coalition_raw in PERIODS:
        print(f"Loading {label}…", end=" ", flush=True)
        nodes, edges = load(period_dir)
        coalition = canon_coalition(coalition_raw)
        result = gov_opp_by_party(nodes, edges, coalition)

        present_parties = set(nodes.values())
        for p in all_opp_parties:
            if p == "fraktionslos":
                continue
            party_series.setdefault(p, [])
            if p not in present_parties:
                party_series[p].append(float("nan"))   # not in Bundestag
            elif p in coalition:
                party_series[p].append(float("nan"))   # in government this term
            else:
                party_series[p].append(result.get(p, float("nan")))

        agg_series.append(result["__all__"])
        period_labels.append(label)
        print(f"opp parties: {sorted(p for p in result if p != '__all__')}")

    x = np.arange(len(period_labels))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white", labelsize=11)
    ax.spines[:].set_color("#333")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    # Aggregate gov↔opp line (thin, white, background reference)
    ax.plot(x, agg_series, color="white", lw=1.5, ls="--",
            alpha=0.4, label="Gov ↔ Opp (all)", zorder=2)

    # Per-party lines (only plotted in periods when in opposition)
    for party in sorted(party_series):
        vals = party_series[party]
        if all(np.isnan(v) for v in vals):
            continue
        color = resolve_color(party, colors)

        # Plot only connected segments where data exists
        xs_seg, vs_seg = [], []
        for i, v in enumerate(vals):
            if not np.isnan(v):
                xs_seg.append(i)
                vs_seg.append(v)
            else:
                # flush current segment
                if len(xs_seg) >= 1:
                    mk = "o" if len(xs_seg) > 1 else "o"
                    ax.plot(xs_seg, vs_seg, "o-" if len(xs_seg) > 1 else "o",
                            color=color, lw=2.2, ms=7, label=party, zorder=3)
                    xs_seg, vs_seg = [], []
        if xs_seg:
            ax.plot(xs_seg, vs_seg, "o-" if len(xs_seg) > 1 else "o",
                    color=color, lw=2.2, ms=7, label=party, zorder=3)

    ax.axhline(0, color="#444", linewidth=0.8, linestyle="--")
    coalition_labels = [
        "+".join(sorted(canon_coalition(c))) for _, _, c in PERIODS
    ]
    combined = [f"{p}\n{c}" for p, c in zip(period_labels, coalition_labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(combined, color="white", fontsize=9, linespacing=1.6)
    ax.set_ylabel("Mean Cohen's κ  (gov ↔ party)", color="white", fontsize=10)
    ax.set_title(
        "Government ↔ Opposition kappa by party over time\n"
        "Lines shown only when party is in opposition  —  dashed white = aggregate",
        color="white", fontsize=12, pad=14,
    )

    # Deduplicate legend (each party label only once)
    handles, labels_leg = ax.get_legend_handles_labels()
    seen, deduped = set(), []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            deduped.append((h, l))
    ax.legend(*zip(*deduped), facecolor="#1a1a2e", labelcolor="white",
              framealpha=0.8, fontsize=10, loc="lower right")

    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    out = ROOT / "output" / "img" / "polarisation_by_party.png"
    (ROOT / "output" / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
