"""
mp_ising_subsample_spread.py

Control for poll budget: subsample every period to the same K polls,
refit PLM coupling, measure the spread of coupling across MPs.
Bootstrap B times for confidence intervals.

If the narrowing trend survives equal poll budgets, it's real.
If it vanishes, it was a poll-count artifact.

Usage
-----
  python analysis/ising/mp_ising_subsample_spread.py
"""

from __future__ import annotations
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent.parent

PERIODS = [
    ("bundestag_2005_2009", "2005–09"),
    ("bundestag_2009_2013", "2009–13"),
    ("bundestag_2013_2017", "2013–17"),
    ("bundestag_2017_2021", "2017–21"),
    ("bundestag_2021_2025", "2021–25"),
    ("bundestag_2025_2029", "2025–29"),
]

ALIASES = {"DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN", "DIE LINKE": "Die Linke", "Die Linke.": "Die Linke"}
def canon(p): return ALIASES.get(str(p), str(p))
VOTE_MAP = {"yes": 1.0, "no": -1.0}

PLM_C  = 0.05
N_JOBS = -1
K_POLLS = 46     # common poll budget (= min period)
B = 12           # bootstrap reps


def load_vote_matrix(period_key: str):
    d = BASE_DIR / "output" / period_key
    with open(d / "raw.json") as f:
        raw = json.load(f)
    nodes = pd.read_csv(d / "nodes.csv")
    nodes["party"] = nodes["party"].map(canon)
    poll_ids = [p["id"] for p in raw["polls"]]
    poll_idx = {pid: i for i, pid in enumerate(poll_ids)}
    mp_ids   = nodes["person_id"].tolist()
    mp_idx   = {mid: i for i, mid in enumerate(mp_ids)}
    S = np.full((len(mp_ids), len(poll_ids)), np.nan, dtype=np.float32)
    for v in raw["votes"]:
        val = VOTE_MAP.get(v["vote"])
        if val is None: continue
        mi = mp_idx.get(v["mandate"]["id"]); pi = poll_idx.get(v["poll"]["id"])
        if mi is not None and pi is not None:
            S[mi, pi] = val
    yes_frac = np.nanmean(S == 1, axis=0)
    keep = (yes_frac >= 0.05) & (yes_frac <= 0.95)
    S = S[:, keep]
    return S


def _fit(i, S, m):
    N = S.shape[0]
    obs = ~np.isnan(S[i]); y = S[i, obs]
    X = S[:, obs].T.copy()
    for j in range(N):
        nm = np.isnan(X[:, j]); X[nm, j] = m[j]
    X[:, i] = 0.0
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.zeros(N)
    clf = LogisticRegression(C=PLM_C, penalty="l2", solver="lbfgs",
                             max_iter=500, fit_intercept=True)
    clf.fit(X, (y + 1) / 2)
    J_i = clf.coef_[0].copy(); J_i[i] = 0.0
    return J_i


def coupling_spread(S):
    """Fit PLM on S, return (p10-p90 width, std) of coupling."""
    N = S.shape[0]
    # drop MPs with too few obs in this subsample
    active = (~np.isnan(S)).sum(axis=1) >= 3
    S = S[active]
    N = S.shape[0]
    m = np.nan_to_num(np.nanmean(S, axis=1), nan=0.0)
    rows = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_fit)(i, S, m) for i in range(N))
    J_raw = np.array(rows)
    J = (J_raw + J_raw.T) / 2.0; np.fill_diagonal(J, 0.0)
    coup = np.abs(J).sum(axis=1) / (N - 1) * 1000
    p10, p90 = np.percentile(coup, [10, 90])
    return p90 - p10, coup.std()


print(f"Subsampling every period to K={K_POLLS} polls, B={B} reps\n")
print(f"{'period':8} {'full_polls':>10} {'p10-p90 (mean±sd)':>22} {'std (mean±sd)':>18}")

rng = np.random.default_rng(42)
results = {}
for period_key, lbl in PERIODS:
    d = BASE_DIR / "output" / period_key
    if not (d / "raw.json").exists():
        continue
    S_full = load_vote_matrix(period_key)
    n_poll = S_full.shape[1]
    widths, stds = [], []
    for b in range(B):
        cols = rng.choice(n_poll, size=min(K_POLLS, n_poll), replace=False)
        w, sd = coupling_spread(S_full[:, cols])
        widths.append(w); stds.append(sd)
    widths, stds = np.array(widths), np.array(stds)
    results[lbl] = (n_poll, widths, stds)
    print(f"{lbl:8} {n_poll:>10} "
          f"{widths.mean():>10.2f} ± {widths.std():>5.2f}      "
          f"{stds.mean():>8.2f} ± {stds.std():>5.2f}")

print("\nReference (full data, no subsample) p10-p90 was:")
print("  2005-09:5.20  2009-13:4.30  2013-17:4.54  2017-21:3.57  2021-25:3.48  2025-29:1.70")
