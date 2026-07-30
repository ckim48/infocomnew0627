"""Is the first-order decayed Markov chain (what FACE maintains) leaving
accuracy on the table? Same trace/split/eval as zone_pred_movers.py
(mover-conditional destination at h = 6), comparing:

  M1  matrix-power  : P-hat^6 from one-step decayed counts (current FACE)
  M2  direct-h      : direct 6-step decayed counts  cur -> dest
  M3  +entry-dir    : direct 6-step counts conditioned on (cur, entry
                      direction), where entry direction is the grid offset
                      from the last distinct zone (9 buckets incl. unknown);
                      unseen states fall back M3 -> M2 -> M1.

Out: newnewdata/zone_pred_upgrades.csv
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np

CELL, DECAY, H = 300.0, 0.97, 6


def run(cache):
    tr = np.load(cache, allow_pickle=True)
    XY = tr["veh_xy"]; K, N = XY.shape[0], XY.shape[1]
    Z = np.floor(XY / CELL).astype(int)
    zid, zseq = {}, np.zeros((K, N), dtype=int)
    for k in range(K):
        for i in range(N):
            zseq[k, i] = zid.setdefault((Z[k, i, 0], Z[k, i, 1]), len(zid))
    nz = len(zid); split = int(0.6 * K)
    gxy = np.array([k for k, v in sorted(zid.items(), key=lambda t: t[1])])

    # entry direction per (k, i): grid offset sign from last distinct zone
    dirn = np.full((K, N), 8, dtype=int)          # 8 = unknown
    last = np.full(N, -1, dtype=int)
    for k in range(K):
        for i in range(N):
            c = zseq[k, i]
            if last[i] >= 0 and last[i] != c:
                dx, dy = np.clip(gxy[c] - gxy[last[i]], -1, 1)
                dirn[k:, i] = (dx + 1) * 3 + (dy + 1)
                if (dx, dy) == (0, 0):
                    dirn[k:, i] = 8
            if k > 0 and zseq[k-1, i] != c:
                last[i] = zseq[k-1, i]

    C1 = np.zeros((nz, nz))                        # one-step
    C2 = np.zeros((nz, nz))                        # direct h-step
    C3 = {}                                        # (cur, dir) -> counts
    for k in range(1, split):
        C1 *= DECAY; C2 *= DECAY
        np.add.at(C1, (zseq[k-1], zseq[k]), 1.0)
        if k + H < split:
            np.add.at(C2, (zseq[k], zseq[k+H]), 1.0)
            for i in range(N):
                key = (zseq[k, i], dirn[k, i])
                row = C3.get(key)
                if row is None:
                    row = C3[key] = np.zeros(nz)
                row[zseq[k+H, i]] += 1.0
    for row in C3.values():
        pass                                       # (decay omitted for C3:
    # per-key decay tracking is costly; direct counts over 54 min behave
    # the same as decayed here -- see zone_pred_learncurve.csv)

    rs = C1.sum(1, keepdims=True)
    P1 = np.divide(C1, rs, out=np.zeros_like(C1), where=rs > 0)
    Ph = np.linalg.matrix_power(P1, H)

    def rank_of(row, c, f):
        r = row.copy(); r[c] = 0.0
        s = r.sum()
        if s <= 0:
            return None
        return int((r > r[f]).sum())

    res = {m: [0, 0, 0] for m in ("M1", "M2", "M3")}   # t1, t3, tot
    for k in range(split, K - H):
        cur, fut = zseq[k], zseq[k + H]
        for i in np.where(cur != fut)[0]:
            c, f = cur[i], fut[i]
            rows = {"M1": Ph[c], "M2": C2[c] if C2[c].sum() else Ph[c]}
            r3 = C3.get((c, dirn[k, i]))
            rows["M3"] = r3 if r3 is not None and r3.sum() else rows["M2"]
            for m, row in rows.items():
                rk = rank_of(row, c, f)
                if rk is None:
                    continue
                res[m][0] += int(rk < 1)
                res[m][1] += int(rk < 3)
                res[m][2] += 1
    return res


out = []
for name, cache in [("Rush-hour peak", "newnewdata/v2x_seoul_trace_evening45.npz"),
                    ("Late-night off-peak", "newnewdata/v2x_seoul_trace_nightw23.npz")]:
    res = run(cache)
    print(f"== {name} (mover-conditional, h={H}) ==")
    for m, (t1, t3, n) in res.items():
        print(f"  {m}: top1 {100*t1/n:4.1f}%  top3 {100*t3/n:4.1f}%  (n={n})")
        out.append([name, m, f"{t1/n:.4f}", f"{t3/n:.4f}", n])
with open("newnewdata/zone_pred_upgrades.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window", "model", "top1", "top3", "n"]); w.writerows(out)
print("wrote newnewdata/zone_pred_upgrades.csv")
