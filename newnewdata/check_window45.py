"""Verification for the Monday 90-min no-replay windows (TODO_0727 items
1/2/4/5): trace shape, fixed-tau reachability on seed-mean curves, last-20
accuracy ranking, paired FACE-Learning seed differences, and comm savings.

Run:  python3 newnewdata/check_window45.py [evening45|night45 ...]
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

ORDER = [("Caching-assisted", "Cached-DFL"), ("V2V-aware", "V2V"),
         ("Learning-aware", "Learning"), ("mmFedMC", "mmFedMC"),
         ("AutoFed", "AutoFed"), ("Proposed", "FACE")]
FIXED_TAUS = {"kitti": 0.55, "nuscenes": 0.69}
TAIL = 20

windows = sys.argv[1:] or ["evening45", "night45"]
for w in windows:
    tr_path = f"newnewdata/v2x_seoul_trace_{w}.npz"
    if os.path.exists(tr_path):
        tr = np.load(tr_path)
        print(f"== {w}: trace N={tr['veh_xy'].shape[1]}, "
              f"K={tr['veh_seg'].shape[0]}")
    else:
        print(f"== {w}: trace cache missing ({tr_path})")
    for ds in ("kitti", "nuscenes"):
        p = f"newnewdata/metrics_v2x_real_{ds}_{w}.npz"
        if not os.path.exists(p):
            print(f"   {ds}: MISSING {p}")
            continue
        d = np.load(p)
        tau = FIXED_TAUS[ds]
        print(f"   -- {ds} (fixed tau={100*tau:.0f}%) --")
        rows = {}
        for s, disp in ORDER:
            if f"{s}__acc_all" not in d.files:
                continue
            A = d[f"{s}__acc_all"]; M = d[f"{s}__txmb_all"]
            am = A.mean(0)
            reach = am >= tau
            r = int(np.argmax(reach)) + 1 if reach.any() else None
            gb = M[:, :r].sum(1).mean() / 1024.0 if r else None
            rows[s] = dict(
                tail=A[:, -TAIL:].mean(1), r=r, gb=gb,
                total_gb=M.sum(1).mean() / 1024.0,
                final=am[-1], peak=am.max(), T=A.shape[1])
            print(f"      {disp:11s} last20={100*rows[s]['tail'].mean():5.1f}"
                  f" +-{100*rows[s]['tail'].std():4.1f}"
                  f"  final={100*am[-1]:5.1f}  peak={100*am.max():5.1f}"
                  f"  tau@{r if r else '>'+str(A.shape[1])}"
                  f"  comm@tau={f'{gb:.1f}GB' if gb else '---'}"
                  f"  total={rows[s]['total_gb']:.1f}GB")
        if "Proposed" in rows:
            face = rows["Proposed"]["tail"]
            base = {s: v for s, v in rows.items() if s != "Proposed"}
            bb = max(base, key=lambda s: base[s]["tail"].mean())
            diff = face - rows[bb]["tail"]
            print(f"      paired FACE-{bb}: "
                  + " ".join(f"{100*x:+.2f}" for x in diff)
                  + f"  mean={100*diff.mean():+.2f}pp"
                  f"  wins={int((diff > 0).sum())}/{len(diff)}")
            if rows["Proposed"]["gb"]:
                for s, v in base.items():
                    if v["gb"]:
                        sav = 100 * (1 - rows["Proposed"]["gb"] / v["gb"])
                        print(f"      comm@tau vs {s}: FACE saves {sav:+.1f}%")
