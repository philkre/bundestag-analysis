from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and plot a voting similarity network."
    )
    parser.add_argument("--votes", default="votes.jsonl", help="Path to votes.jsonl")
    parser.add_argument(
        "--polls", default="polls.jsonl", help="Path to polls.jsonl (unused for now)"
    )
    parser.add_argument(
        "--parliament",
        default="Bundestag 2017 - 2021",
        help="Parliament label to filter on (must match polls.jsonl field_legislature label).",
    )
    parser.add_argument(
        "--start-date", default="2017-10-24", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default="2021-10-26", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--out-dir", default="output", help="Output directory for CSV/GEXF/plot"
    )
    parser.add_argument("--plot", default="network.png", help="Plot filename")
    parser.add_argument(
        "--metric-name",
        default="Cohen's kappa",
        help="Metric name used in plot titles and histogram labels.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=-1.0,
        help="Keep edges with weight strictly greater than this value.",
    )
    parser.add_argument(
        "--min-co-votes",
        type=int,
        default=50,
        help="Minimum number of shared roll-call votes for an MP pair.",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Write a histogram of edge weights to output directory.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of bins for histogram.",
    )
    parser.add_argument(
        "--labels", action="store_true", help="Draw node labels (slow for big graphs)"
    )
    parser.add_argument(
        "--layout",
        default="party",
        choices=["party", "force_atlas2"],
        help="Layout algorithm for node positions.",
    )
    parser.add_argument(
        "--only-cross-party",
        action="store_true",
        help="Only draw edges between MPs of different parties.",
    )
    parser.add_argument(
        "--party-colors",
        default="config/party_colours.json",
        help="Path to JSON mapping party label -> color (hex or any matplotlib color).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild graph and CSV/GEXF outputs from JSONL instead of using existing CSVs.",
    )
    parser.add_argument(
        "--exclude-votes",
        action="append",
        default=["no_show"],
        help="Vote values to exclude (e.g. --exclude-votes no_show). Can be repeated.",
    )
    return parser.parse_args()
