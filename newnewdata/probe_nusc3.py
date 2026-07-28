"""1-seed probe: 3-class nuScenes (Car/Ped/Cyclist, min_class_count=100)
on the evening45 trace, to gauge whether the harder task widens scheme
separation before committing to a full recompute of the nuScenes column.

Out: newnewdata/metrics_probe_nusc3_evening45.npz
"""
import os, sys, time

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

cfg2 = Config(); cfg2.results_dir = OUT
T = min(400, tr["veh_seg"].shape[0])
t0 = time.time()
variant = "fine" if os.environ.get("NUSC_FINE") == "1" else "3"
R.run(cfg=cfg2, seeds=[2026], dataset="nuscenes", rounds=T,
      num_vehicles=min(180, tr["veh_xy"].shape[1]), merge=True,
      min_class_count=100,
      schemes=["Caching-assisted", "V2V-aware", "mmFedMC", "AutoFed",
               "Proposed"],
      out_name=f"metrics_probe_nusc{variant}_evening45.npz")
print(f"[probe-nusc3] done in {(time.time() - t0) / 60:.0f} min")
