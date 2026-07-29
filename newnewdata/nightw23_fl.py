"""FL on the clean Wednesday 23:00-00:30 late-night window (N=76, K=505,
coverage>=0.7 per the window45 sparse-cohort fallback): 8 seeds x 5 table
methods x T=400 no-replay, KITTI (default task) + nuScenes (fine 3-class).

Out: newnewdata/metrics_v2x_real_kitti_nightw23.npz
     newnewdata/metrics_v2x_real_nuscenes_nightw23_fine.npz
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
RAW = "data/gangnam/seoul_v2x_trace_nightw23.npz"
CACHE = os.path.join(OUT, "v2x_seoul_trace_nightw23.npz")

cfg = Config(); cfg.results_dir = OUT
tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
R.build_v2x_trace = lambda c: build_v2x_trace(
    c, cache=CACHE, v2x_file=RAW, verbose=False)
T = min(400, tr["veh_seg"].shape[0])
SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
SCHEMES = ["Caching-assisted", "V2V-aware", "mmFedMC", "AutoFed", "Proposed"]

for ds, out in [("kitti", "metrics_v2x_real_kitti_nightw23.npz"),
                ("nuscenes", "metrics_v2x_real_nuscenes_nightw23_fine.npz")]:
    if os.path.exists(os.path.join(OUT, out)):
        print(f"[nightw23] {ds}: exists, skipping", flush=True)
        continue
    cfg2 = Config(); cfg2.results_dir = OUT
    t0 = time.time()
    R.run(cfg=cfg2, seeds=SEEDS, dataset=ds, rounds=T,
          num_vehicles=min(180, tr["veh_xy"].shape[1]), merge=True,
          schemes=SCHEMES, out_name=out)
    print(f"[nightw23] {ds}: T={T}, done in {(time.time() - t0) / 60:.0f} min",
          flush=True)
print("[nightw23] ALL DONE", flush=True)
