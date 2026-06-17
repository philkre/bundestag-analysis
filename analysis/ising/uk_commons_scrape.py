"""
uk_commons_scrape.py

Scrape UK House of Commons division data from the Parliament API and
produce raw.json + nodes.csv compatible with the Bundestag PLM pipeline.

Periods covered (API data available from 2016 onward):
  2017-19  — Theresa May parliament
  2019-24  — Johnson / Truss / Sunak parliament
  2024-    — Starmer parliament

Usage
-----
  python analysis/ising/uk_commons_scrape.py
  python analysis/ising/uk_commons_scrape.py --resume   # skip already-done periods
"""

from __future__ import annotations
import json, sys, time, urllib.request
from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
API_BASE = "https://commonsvotes-api.parliament.uk/data"

PERIODS = [
    ("uk_commons_2017_2019", "2017-06-09", "2019-11-05", "2017–19"),
    ("uk_commons_2019_2024", "2019-12-13", "2024-05-22", "2019–24"),
    ("uk_commons_2024_2029", "2024-07-05", "2099-01-01", "2024–"),
]

RESUME = "--resume" in sys.argv


def fetch_json(url: str, retries: int = 4) -> object:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"\n  retry {attempt+1} after {wait}s ({e})")
            time.sleep(wait)


def get_division_ids(start: str, end: str) -> list[dict]:
    """Paginate through division list, return list of division stubs."""
    divisions = []
    skip = 0
    while True:
        url = (
            f"{API_BASE}/divisions.json/search"
            f"?queryParameters.startDate={start}"
            f"&queryParameters.endDate={end}"
            f"&queryParameters.take=25&queryParameters.skip={skip}"
        )
        batch = fetch_json(url)
        divisions.extend(batch)
        print(f"  listing: {len(divisions)} so far…", end="\r", flush=True)
        if len(batch) < 25:
            break
        skip += 25
        time.sleep(0.08)
    print()
    return divisions


for folder, start_date, end_date, label in PERIODS:
    out_dir = BASE_DIR / "output" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path   = out_dir / "raw.json"
    nodes_path = out_dir / "nodes.csv"

    if RESUME and raw_path.exists() and nodes_path.exists():
        print(f"{label}: already done, skipping")
        continue

    print(f"\n{'='*60}")
    print(f"{label}  ({start_date} → {end_date})")
    print(f"{'='*60}")

    # ── 1. Get division list ─────────────────────────────────────
    print("Fetching division list…")
    div_stubs = get_division_ids(start_date, end_date)
    print(f"  {len(div_stubs)} divisions found")

    # ── 2. Fetch each division's detail ──────────────────────────
    polls: list[dict] = []
    votes: list[dict] = []
    mp_info: dict[int, dict] = {}   # member_id -> {name, party}

    for i, stub in enumerate(div_stubs):
        div_id = stub["DivisionId"]
        print(f"  [{i+1:4d}/{len(div_stubs)}] div {div_id}…", end="\r", flush=True)

        try:
            detail = fetch_json(f"{API_BASE}/division/{div_id}.json")
        except Exception as e:
            print(f"\n  Warning: skipping {div_id}: {e}")
            continue

        # Teller MemberIds — exclude from vote counts (fields can be null)
        teller_ids = {
            t["MemberId"]
            for t in (detail.get("AyeTellers") or []) + (detail.get("NoTellers") or [])
        }

        polls.append({"id": str(div_id), "title": detail.get("Title", "")})

        for mp in detail.get("Ayes", []):
            mid = mp["MemberId"]
            if mid in teller_ids:
                continue
            mp_info[mid] = {"name": mp["Name"], "party": mp["Party"]}
            votes.append({
                "mandate": {"id": mid},
                "poll":    {"id": str(div_id)},
                "vote":    "yes",
            })

        for mp in detail.get("Noes", []):
            mid = mp["MemberId"]
            if mid in teller_ids:
                continue
            mp_info[mid] = {"name": mp["Name"], "party": mp["Party"]}
            votes.append({
                "mandate": {"id": mid},
                "poll":    {"id": str(div_id)},
                "vote":    "no",
            })

        time.sleep(0.04)   # ~25 req/s, well within limits

    print(f"\n  {len(polls)} polls | {len(votes)} votes | {len(mp_info)} MPs")

    # ── 3. Save raw.json ─────────────────────────────────────────
    with open(raw_path, "w") as f:
        json.dump({"polls": polls, "votes": votes}, f)
    print(f"  Saved raw.json")

    # ── 4. Save nodes.csv ────────────────────────────────────────
    nodes_df = pd.DataFrame([
        {"person_id": mid, "name": info["name"], "party": info["party"]}
        for mid, info in mp_info.items()
    ])
    nodes_df.to_csv(nodes_path, index=False)
    print(f"  Saved nodes.csv  ({len(nodes_df)} MPs)")

print("\nDone.")
