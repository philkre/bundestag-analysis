"""
uk_commons_plm_cache.py

Build PLM coupling cache for UK House of Commons periods.
Saves output/uk_ising_field_cache.csv with columns:
  period, name, party, person_id, coupling

Usage
-----
  python analysis/ising/uk_commons_plm_cache.py
"""

from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent.parent
CACHE    = BASE_DIR / "output" / "uk_ising_field_cache.csv"

PERIODS = [
    ("uk_commons_2017_2019", "2017–19"),
    ("uk_commons_2019_2024", "2019–24"),
    ("uk_commons_2024_2029", "2024–"),
]

VOTE_MAP = {"yes": 1.0, "no": -1.0}
PLM_C    = 0.05
N_JOBS   = -1


def load_vote_matrix(period_key: str):
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)

    nodes = pd.read_csv(d / "nodes.csv")

    poll_ids = [p["id"] for p in raw["polls"]]
    poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
    mp_ids   = nodes["person_id"].tolist()
    mp_idx   = {mid: i for i, mid in enumerate(mp_ids)}

    S = np.full((len(mp_ids), len(poll_ids)), np.nan, dtype=np.float32)
    for v in raw["votes"]:
        val = VOTE_MAP.get(v["vote"])
        if val is None:
            continue
        mi = mp_idx.get(v["mandate"]["id"])
        pi = poll_idx.get(v["poll"]["id"])
        if mi is not None and pi is not None:
            S[mi, pi] = val

    # Filter near-unanimous polls
    yes_frac = np.nanmean(S == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S = S[:, keep]

    # Filter low-participation MPs
    n_poll_k = keep.sum()
    active = (~np.isnan(S)).sum(axis=1) >= max(3, 0.1 * n_poll_k)
    nodes = nodes[active].reset_index(drop=True)
    S     = S[active]

    return S, nodes


def _fit_one_spin(i: int, S: np.ndarray, m: np.ndarray):
    N = S.shape[0]
    obs = ~np.isnan(S[i])
    y   = S[i, obs]
    X   = S[:, obs].T.copy()
    for j in range(N):
        nm = np.isnan(X[:, j])
        X[nm, j] = m[j]
    X[:, i] = 0.0
    if len(y) < 5 or len(np.unique(y)) < 2:
        return 0.0, np.zeros(N)
    clf = LogisticRegression(C=PLM_C, penalty="l2", solver="lbfgs",
                             max_iter=500, fit_intercept=True)
    clf.fit(X, (y + 1) / 2)
    h_i = float(clf.intercept_[0])
    J_i = clf.coef_[0].copy()
    J_i[i] = 0.0
    return h_i, J_i


rows = []

for period_key, lbl in PERIODS:
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        print(f"{lbl}: no data, skipping (run uk_commons_scrape.py first)")
        continue

    print(f"\n{lbl}: loading…", flush=True)
    S, nodes = load_vote_matrix(period_key)
    N, M = S.shape
    print(f"  {N} MPs × {M} polls", flush=True)

    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit_one_spin)(i, S, m) for i in range(N))

    J_raw = np.array([r[1] for r in results])
    J = (J_raw + J_raw.T) / 2.0
    np.fill_diagonal(J, 0.0)

    coupling = np.abs(J).sum(axis=1) / (N - 1) * 1000

    for idx, (_, row) in enumerate(nodes.iterrows()):
        rows.append({
            "period":    lbl,
            "name":      row["name"],
            "party":     row["party"],
            "person_id": row["person_id"],
            "coupling":  float(coupling[idx]),
        })
    print(f"  coupling range: {coupling.min():.3f} – {coupling.max():.3f}")

df = pd.DataFrame(rows)
df.to_csv(CACHE, index=False)
print(f"\nSaved {len(df)} rows → {CACHE}")
