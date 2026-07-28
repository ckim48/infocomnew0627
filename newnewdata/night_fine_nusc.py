"""Fine-grained 3-class nuScenes on the banked Friday late-night replay
trace (N=120, T=250, the Table II protocol): 8 seeds x 5 table methods.
Plan B for the off-peak table after two consecutive nights of Seoul V2X
API outages killed the 90-min night windows.

Out: newnewdata/metrics_v2x_real_nuscenes_night_fine.npz
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
RAW = "data/gangnam/seoul_v2x_trace_fri_night.npz"
CACHE = os.path.join(OUT, "v2x_seoul_trace_night.npz")

cfg = Config(); cfg.results_dir = OUT
tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
R.build_v2x_trace = lambda c: build_v2x_trace(
    c, cache=CACHE, v2x_file=RAW, verbose=False)

out_npz = os.path.join(OUT, "metrics_v2x_real_nuscenes_night_fine.npz")
if os.path.exists(out_npz):
    print("[nightfine] exists, skipping"); sys.exit(0)
cfg2 = Config(); cfg2.results_dir = OUT
t0 = time.time()
R.run(cfg=cfg2, seeds=[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033],
      dataset="nuscenes", rounds=250,
      num_vehicles=min(180, tr["veh_xy"].shape[1]), merge=True,
      schemes=["Caching-assisted", "V2V-aware", "mmFedMC", "AutoFed",
               "Proposed"],
      out_name="metrics_v2x_real_nuscenes_night_fine.npz")
print(f"[nightfine] done in {(time.time() - t0) / 60:.0f} min", flush=True)
