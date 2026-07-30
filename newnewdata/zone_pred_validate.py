"""Empirical validation of the zone-based mobility prediction used by the
future-contact utility (Sec. IV-B): learn the decayed zone-transition
matrix P on the first 60% of each real trace, then measure h-step-ahead
zone prediction accuracy on the held-out 40%, against a persistence
baseline (predict the current zone).

Out: Figures/fig_zone_pred_1x2.{png,pdf}
     newnewdata/zone_pred_validation.csv (panel-a data)
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

CELL = 300.0
DECAY = 0.97
HS = [1, 2, 3, 6, 12, 18, 24, 30]


def zones_of(xy):
    return np.floor(xy / CELL).astype(int)


def run_window(cache):
    tr = np.load(cache, allow_pickle=True)
    XY = tr["veh_xy"]                     # K x N x 2 (local metres)
    K, N = XY.shape[0], XY.shape[1]
    Z = zones_of(XY)                      # K x N x 2 (cell indices)
    zid = {}
    zseq = np.zeros((K, N), dtype=int)
    for k in range(K):
        for i in range(N):
            key = (Z[k, i, 0], Z[k, i, 1])
            zseq[k, i] = zid.setdefault(key, len(zid))
    nz = len(zid)
    split = int(0.6 * K)
    # decayed transition counts on the training prefix
    C = np.zeros((nz, nz))
    for k in range(1, split):
        C *= DECAY
        for i in range(N):
            C[zseq[k - 1, i], zseq[k, i]] += 1.0
    rs = C.sum(1, keepdims=True)
    P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
    # h-step top-1 predictions on the held-out part
    acc_m, acc_p = [], []
    for h in HS:
        Ph = np.linalg.matrix_power(P, h)
        pred = Ph.argmax(1)               # most likely zone h steps ahead
        ok_m = ok_p = tot = 0
        for k in range(split, K - h):
            cur = zseq[k]; fut = zseq[k + h]
            ok_m += int((pred[cur] == fut).sum())
            ok_p += int((cur == fut).sum())
            tot += N
        acc_m.append(ok_m / tot); acc_p.append(ok_p / tot)
    # per-zone Markov accuracy at h=6 for the map panel
    h = 6
    Ph = np.linalg.matrix_power(P, h); pred = Ph.argmax(1)
    zok = np.zeros(nz); zn = np.zeros(nz)
    for k in range(split, K - h):
        cur = zseq[k]; fut = zseq[k + h]
        for i in range(N):
            zn[cur[i]] += 1
            zok[cur[i]] += float(pred[cur[i]] == fut[i])
    zacc = np.divide(zok, zn, out=np.full(nz, np.nan), where=zn >= 30)
    inv = {v: k for k, v in zid.items()}
    centers = np.array([[(inv[z][0] + .5) * CELL, (inv[z][1] + .5) * CELL]
                        for z in range(nz)])
    return np.array(acc_m), np.array(acc_p), zacc, centers, zn, tr["ctr"]


res = {}
for name, cache in [("Rush-hour peak", "newnewdata/v2x_seoul_trace_evening45.npz"),
                    ("Late-night off-peak", "newnewdata/v2x_seoul_trace_nightw23.npz")]:
    res[name] = run_window(cache)
    m, p = res[name][0], res[name][1]
    print(f"{name}: h={HS}")
    print(f"  markov      {np.round(100*m,1)}")
    print(f"  persistence {np.round(100*p,1)}")

with open("newnewdata/zone_pred_validation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window", "horizon_rounds", "markov_top1_acc",
                "persistence_top1_acc"])
    for name in res:
        for h, m, p in zip(HS, res[name][0], res[name][1]):
            w.writerow([name, h, f"{m:.4f}", f"{p:.4f}"])
print("wrote newnewdata/zone_pred_validation.csv")
