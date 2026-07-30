"""How do the zone-transition statistics improve as observations
accumulate? Train the transition model on the last L rounds before the
train/test split (same 60/40 split, decay, h = 6, mover-conditional as
zone_pred_movers.py), sweeping L. Also a no-decay variant to separate
"more memory" from "recency weighting".

Out: newnewdata/zone_pred_learncurve.csv
     (window, decay, L_rounds, top1, top3, n_movers)
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np

CELL, H = 300.0, 6
LS = (15, 30, 60, 120, 180, 240, 324)      # rounds of observed history


def prep(cache):
    tr = np.load(cache, allow_pickle=True)
    XY = tr["veh_xy"]; K, N = XY.shape[0], XY.shape[1]
    Z = np.floor(XY / CELL).astype(int)
    zid, zseq = {}, np.zeros((K, N), dtype=int)
    for k in range(K):
        for i in range(N):
            zseq[k, i] = zid.setdefault((Z[k, i, 0], Z[k, i, 1]), len(zid))
    return zseq, len(zid), K


def evaluate(zseq, nz, K, L, decay):
    split = int(0.6 * K)
    C = np.zeros((nz, nz))
    for k in range(max(1, split - L), split):
        C *= decay
        np.add.at(C, (zseq[k-1], zseq[k]), 1.0)
    rs = C.sum(1, keepdims=True)
    P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
    Ph = np.linalg.matrix_power(P, H)
    R = Ph.copy(); np.fill_diagonal(R, 0.0)
    order = np.argsort(R, axis=1)[:, ::-1]
    pos = np.empty_like(order)
    for c in range(nz):
        pos[c, order[c]] = np.arange(nz)
    ok = R.sum(1) > 0
    # movers whose origin has NO observed outgoing transition yet fall back
    # to the smoothed adjacency prior (as in FACE, Eq. 11) -> expected hit
    # of a blind guess over the 8 adjacent cells
    t1 = t3 = 0.0
    tot = seen = 0
    for k in range(split, K - H):
        cur, fut = zseq[k], zseq[k + H]
        for i in np.where(cur != fut)[0]:
            c, f = cur[i], fut[i]
            if ok[c]:
                r = pos[c, f]
                t1 += int(r < 1); t3 += int(r < 3); seen += 1
            else:
                t1 += 1.0 / 8.0; t3 += 3.0 / 8.0
            tot += 1
    return t1 / tot, t3 / tot, tot, seen / tot


rows = []
for name, cache in [("Rush-hour peak", "newnewdata/v2x_seoul_trace_evening45.npz"),
                    ("Late-night off-peak", "newnewdata/v2x_seoul_trace_nightw23.npz")]:
    zseq, nz, K = prep(cache)
    for decay in (0.97, 1.0):
        for L in LS:
            if L > int(0.6 * K):
                continue
            t1, t3, n, sf = evaluate(zseq, nz, K, L, decay)
            print(f"{name:<20s} decay={decay:4.2f} L={L:3d} "
                  f"({L/6:5.1f} min)  top1 {100*t1:4.1f}%  top3 {100*t3:4.1f}%"
                  f"  seen {100*sf:4.1f}%  (n={n})", flush=True)
            rows.append([name, decay, L, f"{t1:.4f}", f"{t3:.4f}", n,
                         f"{sf:.4f}"])
with open("newnewdata/zone_pred_learncurve.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window", "decay", "L_rounds", "top1", "top3", "n_movers",
                "seen_frac"])
    w.writerows(rows)
print("wrote newnewdata/zone_pred_learncurve.csv")
