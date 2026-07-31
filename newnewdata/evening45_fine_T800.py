"""Convergence-horizon run: fine 3-class nuScenes on the evening45 window,
T = 800 rounds (trace replayed), FACE + strongest DFL baseline + the two
feasibility bounds (NoComm lower, FullContact upper), 3 seeds.

Out: newnewdata/metrics_v2x_real_nuscenes_evening45_fine_T800.npz
"""
import os, sys, time

os.environ["NUSC_FINE"] = "1"
os.environ["NUSC_KEEPALL"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sim.config import Config
from sim.v2x_trace import build_v2x_trace
import sim.run_v2x_real as R

OUT = "newnewdata"
RAW = "data/gangnam/seoul_v2x_trace_evening45.npz"
CACHE = os.path.join(OUT, "v2x_seoul_trace_evening45.npz")

cfg = Config(); cfg.results_dir = OUT
tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
R.build_v2x_trace = lambda c: build_v2x_trace(
    c, cache=CACHE, v2x_file=RAW, verbose=False)

out_npz = os.path.join(OUT,
                       "metrics_v2x_real_nuscenes_evening45_fine_T800.npz")
if os.path.exists(out_npz):
    print("[T800] exists, skipping"); sys.exit(0)
cfg2 = Config(); cfg2.results_dir = OUT
t0 = time.time()
R.run(cfg=cfg2, seeds=[2026, 2027, 2028],
      dataset="nuscenes", rounds=800,
      num_vehicles=min(180, tr["veh_xy"].shape[1]), merge=True,
      schemes=["Proposed", "Caching-assisted", "NoComm", "FullContact"],
      out_name="metrics_v2x_real_nuscenes_evening45_fine_T800.npz")
print(f"[T800] done in {(time.time() - t0) / 60:.0f} min", flush=True)
