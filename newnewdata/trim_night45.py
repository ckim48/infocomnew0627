"""Remove the 23:47-00:02 API-outage snapshots (all-NaN rows) from the
night45 raw trace so the FL run sees only genuinely observed traffic.
Keeps a backup of the untrimmed raw and deletes the derived trace cache."""
import os, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/gangnam/seoul_v2x_trace_night45.npz")
BAK = RAW.replace(".npz", "_with_gap.npz")
CACHE = os.path.join(ROOT, "newnewdata/v2x_seoul_trace_night45.npz")

d = np.load(RAW, allow_pickle=True)
pos, times, ids = d["pos"], d["times"], d["ids"]
keep = ~np.isnan(pos[:, :, 0]).all(axis=1)
print(f"snapshots: {len(keep)} total, dropping {int((~keep).sum())} empty")
if (~keep).sum() == 0:
    print("nothing to trim, exiting")
    raise SystemExit
if not os.path.exists(BAK):
    shutil.copy2(RAW, BAK)
    print("backup ->", BAK)
np.savez_compressed(RAW, ids=ids, pos=pos[keep], times=times[keep])
print(f"trimmed raw saved: pos={pos[keep].shape}")
if os.path.exists(CACHE):
    os.remove(CACHE)
    print("removed stale cache", CACHE)
