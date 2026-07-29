"""Collection-only window runner: poll the Seoul V2X API for 90 min, save
the raw trace + derived cache, and STOP (no FL). Used to bank windows so
they never need re-collecting.

Usage: python3 newnewdata/collect_only.py <name>   (e.g. evening45b)
Out:   data/gangnam/seoul_v2x_trace_<name>.npz  (raw)
       newnewdata/v2x_seoul_trace_<name>.npz    (trace cache)
"""
import os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

NAME = sys.argv[1]
RAW = f"data/gangnam/seoul_v2x_trace_{NAME}.npz"
CACHE = f"newnewdata/v2x_seoul_trace_{NAME}.npz"
DUR = 5400


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] [{NAME}] {msg}", flush=True)


try:
    if not os.path.exists(RAW):
        from sim.seoul_v2x import collect_trace
        log(f"collecting {DUR} s (~{DUR // 10} snapshots)")
        collect_trace(duration_s=DUR, interval_s=10, out=RAW)
    else:
        log("raw exists, skipping collection")
    import numpy as np
    _d = np.load(RAW, allow_pickle=True)
    _keep = ~np.isnan(_d["pos"][:, :, 0]).all(axis=1)
    if (~_keep).sum():
        log(f"trimming {int((~_keep).sum())}/{len(_keep)} empty snapshots")
        np.savez_compressed(RAW, ids=_d["ids"], pos=_d["pos"][_keep],
                            times=_d["times"][_keep])
        if os.path.exists(CACHE):
            os.remove(CACHE)
    from sim.config import Config
    from sim.v2x_trace import build_v2x_trace
    cfg = Config(); cfg.results_dir = "newnewdata"
    tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
    log(f"trace: N={tr['veh_xy'].shape[1]}, K={tr['veh_seg'].shape[0]} "
        f"(banked, no FL)")
except Exception:
    log(f"FAILED\n{traceback.format_exc()}")
