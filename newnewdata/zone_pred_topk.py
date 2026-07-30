"""Mover-conditional top-k destination coverage at the FACE horizon
(h = 6): for vehicles that leave their zone within 6 rounds, how often is
the realized destination among the model's k highest-probability zones?
Same split/decay/methodology as zone_pred_movers.py.

Out: newnewdata/zone_pred_topk.csv  (window, k, coverage, n_movers)
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np

CELL, DECAY, H = 300.0, 0.97, 6
KMAX = 8


def run(cache):
    tr = np.load(cache, allow_pickle=True)
    XY = tr["veh_xy"]; K, N = XY.shape[0], XY.shape[1]
    Z = np.floor(XY / CELL).astype(int)
    zid, zseq = {}, np.zeros((K, N), dtype=int)
    for k in range(K):
        for i in range(N):
            zseq[k, i] = zid.setdefault((Z[k, i, 0], Z[k, i, 1]), len(zid))
    nz = len(zid); split = int(0.6 * K)
    C = np.zeros((nz, nz))
    for k in range(1, split):
        C *= DECAY
        for i in range(N):
            C[zseq[k-1, i], zseq[k, i]] += 1.0
    rs = C.sum(1, keepdims=True)
    P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
    Ph = np.linalg.matrix_power(P, H)
    R = Ph.copy(); np.fill_diagonal(R, 0.0)            # destination dist
    order = np.argsort(R, axis=1)[:, ::-1]             # same tie-break as
    pos = np.empty_like(order)                         # zone_pred_movers.py
    for c in range(nz):
        pos[c, order[c]] = np.arange(nz)
    hits = np.zeros(KMAX); tot = 0
    for k in range(split, K - H):
        cur, fut = zseq[k], zseq[k + H]
        for i in np.where(cur != fut)[0]:
            c, f = cur[i], fut[i]
            if R[c].sum() == 0:
                continue
            r = pos[c, f]
            if r < KMAX:
                hits[r] += 1
            tot += 1
    return np.cumsum(hits) / tot, tot


rows = []
for name, cache in [("Rush-hour peak", "newnewdata/v2x_seoul_trace_evening45.npz"),
                    ("Late-night off-peak", "newnewdata/v2x_seoul_trace_nightw23.npz")]:
    cov, n = run(cache)
    print(f"== {name} (h={H}, n={n}) ==")
    for k in range(KMAX):
        print(f"  top-{k+1}: {100*cov[k]:.1f}%")
        rows.append([name, k + 1, f"{cov[k]:.4f}", n])
with open("newnewdata/zone_pred_topk.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window", "k", "coverage", "n_movers"]); w.writerows(rows)
print("wrote newnewdata/zone_pred_topk.csv")
