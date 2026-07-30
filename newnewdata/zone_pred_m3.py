"""Direction-conditioned zone statistics across all horizons + learning
curve. M3 = direct h-step decayed-free counts keyed by (current zone,
entry direction: 9 buckets from the last distinct zone), falling back to
the unconditioned h-step counts (M2) for unseen states; M1 = first-order
matrix power (memoryless reference, = zone_pred_movers.csv). Same trace,
60/40 split, mover-conditional evaluation.

Out: newnewdata/zone_pred_m3.csv       (window,horizon,model,top1,top3,n)
     newnewdata/zone_pred_m3_learn.csv (window,L_rounds,top1,top3,n)
     -- learning curve at h=6; movers from a state never observed under
        either key fall back to the blind expectation (1/8, 3/8).
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np

CELL, DECAY = 300.0, 0.97
HS = [1, 2, 3, 6, 12, 18, 24, 30]
LS = (15, 30, 60, 120, 180, 240)
H6 = 6


def prep(cache):
    tr = np.load(cache, allow_pickle=True)
    XY = tr["veh_xy"]; K, N = XY.shape[0], XY.shape[1]
    Z = np.floor(XY / CELL).astype(int)
    zid, zseq = {}, np.zeros((K, N), dtype=int)
    for k in range(K):
        for i in range(N):
            zseq[k, i] = zid.setdefault((Z[k, i, 0], Z[k, i, 1]), len(zid))
    nz = len(zid)
    gxy = np.array([k for k, v in sorted(zid.items(), key=lambda t: t[1])])
    dirn = np.full((K, N), 8, dtype=int)          # 8 = unknown
    last = np.full(N, -1, dtype=int)
    for k in range(K):
        for i in range(N):
            c = zseq[k, i]
            if last[i] >= 0 and last[i] != c:
                dx, dy = np.clip(gxy[c] - gxy[last[i]], -1, 1)
                d = (dx + 1) * 3 + (dy + 1)
                dirn[k:, i] = 8 if d == 4 else d
            if k > 0 and zseq[k-1, i] != c:
                last[i] = zseq[k-1, i]
    return zseq, dirn, nz, K


def counts(zseq, dirn, nz, k0, k1, h):
    """h-step counts on train rounds [k0, k1): C2 (zone) and C3 (zone,dir)."""
    C2 = np.zeros((nz, nz)); C3 = {}
    N = zseq.shape[1]
    for k in range(k0, k1 - h):
        np.add.at(C2, (zseq[k], zseq[k+h]), 1.0)
        for i in range(N):
            key = (zseq[k, i], dirn[k, i])
            row = C3.get(key)
            if row is None:
                row = C3[key] = np.zeros(nz)
            row[zseq[k+h, i]] += 1.0
    return C2, C3


def ehit(row, c, f):
    """Expected top-1/top-3 hit under fair (random) tie-breaking: with g
    zones strictly above the true destination's score and t tied with it,
    P(rank <= k) = clip((k - g)/t, 0, 1)."""
    r = row.copy(); r[c] = 0.0
    if r.sum() <= 0:
        return None
    g = int((r > r[f]).sum()); t = int((r == r[f]).sum())
    return (min(max((1 - g) / t, 0.0), 1.0),
            min(max((3 - g) / t, 0.0), 1.0))


rows_h, rows_l = [], []
for name, cache in [("Rush-hour peak", "newnewdata/v2x_seoul_trace_evening45.npz"),
                    ("Late-night off-peak", "newnewdata/v2x_seoul_trace_nightw23.npz")]:
    zseq, dirn, nz, K = prep(cache)
    split = int(0.6 * K)
    # one-step decayed chain for the M1 reference
    C1 = np.zeros((nz, nz))
    for k in range(1, split):
        C1 *= DECAY
        np.add.at(C1, (zseq[k-1], zseq[k]), 1.0)
    rs = C1.sum(1, keepdims=True)
    P1 = np.divide(C1, rs, out=np.zeros_like(C1), where=rs > 0)

    for h in HS:                                   # --- horizon sweep ---
        Ph = np.linalg.matrix_power(P1, h)
        C2, C3 = counts(zseq, dirn, nz, 1, split, h)
        res = {m: [0.0, 0.0, 0] for m in ("M1", "M3")}
        for k in range(split, K - h):
            cur, fut = zseq[k], zseq[k + h]
            for i in np.where(cur != fut)[0]:
                c, f = cur[i], fut[i]
                r3 = C3.get((c, dirn[k, i]))
                m3row = r3 if r3 is not None and r3.sum() else \
                    (C2[c] if C2[c].sum() else Ph[c])
                for m, row in (("M1", Ph[c]), ("M3", m3row)):
                    e = ehit(row, c, f)
                    if e is None:
                        continue
                    res[m][0] += e[0]
                    res[m][1] += e[1]
                    res[m][2] += 1
        for m, (t1, t3, n) in res.items():
            print(f"{name:<20s} h={h:2d} {m}: top1 {100*t1/n:4.1f}%  "
                  f"top3 {100*t3/n:4.1f}%  (n={n})", flush=True)
            rows_h.append([name, h, m, f"{t1/n:.4f}", f"{t3/n:.4f}", n])

    for L in LS:                                   # --- learning curve ---
        C2, C3 = counts(zseq, dirn, nz, max(1, split - L), split, H6)
        t1 = t3 = 0.0; tot = 0
        for k in range(split, K - H6):
            cur, fut = zseq[k], zseq[k + H6]
            for i in np.where(cur != fut)[0]:
                c, f = cur[i], fut[i]
                r3 = C3.get((c, dirn[k, i]))
                row = r3 if r3 is not None and r3.sum() else \
                    (C2[c] if C2[c].sum() else None)
                e = ehit(row, c, f) if row is not None else None
                if e is None:
                    t1 += 1/8; t3 += 3/8
                else:
                    t1 += e[0]; t3 += e[1]
                tot += 1
        print(f"{name:<20s} L={L:3d} ({L/6:4.1f} min) M3: "
              f"top1 {100*t1/tot:4.1f}%  top3 {100*t3/tot:4.1f}%", flush=True)
        rows_l.append([name, L, f"{t1/tot:.4f}", f"{t3/tot:.4f}", tot])

with open("newnewdata/zone_pred_m3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window", "horizon", "model", "top1", "top3", "n"])
    w.writerows(rows_h)
with open("newnewdata/zone_pred_m3_learn.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window", "L_rounds", "top1", "top3", "n"])
    w.writerows(rows_l)
print("wrote newnewdata/zone_pred_m3.csv, zone_pred_m3_learn.csv")
