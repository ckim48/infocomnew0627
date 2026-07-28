"""Drop all-NaN (API-outage) snapshots from a banked raw trace and rebuild
its derived cache. Idempotent; keeps a *_with_gap backup once.

Usage: python3 newnewdata/trim_trace.py <name>   (e.g. evening45b)
"""
import os, sys, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
NAME = sys.argv[1]
RAW = f"data/gangnam/seoul_v2x_trace_{NAME}.npz"
BAK = RAW.replace(".npz", "_with_gap.npz")
CACHE = f"newnewdata/v2x_seoul_trace_{NAME}.npz"

d = np.load(RAW, allow_pickle=True)
pos, times, ids = d["pos"], d["times"], d["ids"]
keep = ~np.isnan(pos[:, :, 0]).all(axis=1)
print(f"[{NAME}] {len(keep)} snapshots, dropping {int((~keep).sum())} empty")
if (~keep).sum():
    if not os.path.exists(BAK):
        shutil.copy2(RAW, BAK)
    np.savez_compressed(RAW, ids=ids, pos=pos[keep], times=times[keep])
    if os.path.exists(CACHE):
        os.remove(CACHE)
from sim.config import Config
from sim.v2x_trace import build_v2x_trace
cfg = Config(); cfg.results_dir = "newnewdata"
tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
print(f"[{NAME}] rebuilt: N={tr['veh_xy'].shape[1]}, K={tr['veh_seg'].shape[0]}")
