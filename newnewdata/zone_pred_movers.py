"""Mover-conditional validation: for vehicles that LEAVE their zone within
h rounds, does the decayed zone-transition model predict the destination?
Baselines: uniform over the 8 adjacent cells. Also reports top-3 coverage.
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np

CELL, DECAY = 300.0, 0.97
HS = [1, 2, 3, 6, 12, 18, 24, 30]


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
    inv = {v: k for k, v in zid.items()}
    out = []
    for h in HS:
        Ph = np.linalg.matrix_power(P, h)
        top1 = top3 = unif = tot = 0
        for k in range(split, K - h):
            cur, fut = zseq[k], zseq[k + h]
            mv = cur != fut
            for i in np.where(mv)[0]:
                c, f = cur[i], fut[i]
                row = Ph[c].copy(); row[c] = 0.0        # destination dist
                if row.sum() == 0: continue
                order = np.argsort(row)[::-1]
                top1 += int(order[0] == f)
                top3 += int(f in order[:3])
                # uniform over 8-neighbour cells baseline
                cx, cy = inv[c]
                unif += 1.0 / 8.0
                tot += 1
        out.append((h, top1/tot, top3/tot, unif/tot, tot))
    return out


rows = []
for name, cache in [("Rush-hour peak", "newnewdata/v2x_seoul_trace_evening45.npz"),
                    ("Late-night off-peak", "newnewdata/v2x_seoul_trace_nightw23.npz")]:
    print(f"== {name} (mover-conditional destination prediction) ==")
    for h, t1, t3, u, n in run(cache):
        print(f"  h={h:2d}: top1 {100*t1:4.1f}%  top3 {100*t3:4.1f}%  uniform-8 {100*u:4.1f}%  (n={n})")
        rows.append([name, h, f"{t1:.4f}", f"{t3:.4f}", f"{u:.4f}", n])
with open("newnewdata/zone_pred_movers.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["window","horizon","markov_top1","markov_top3","uniform8","n_movers"]); w.writerows(rows)
print("wrote newnewdata/zone_pred_movers.csv")
