"""Seed extension for the Monday evening45 window: 5 extra seeds
(2029-2033) x 5 table methods x T=400 on the SAME banked trace, so the
peak table can report 8 seeds total (2026-2028 already exist).

Out: newnewdata/metrics_v2x_real_{kitti,nuscenes}_evening45_s5.npz
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
T = min(400, tr["veh_seg"].shape[0])

for ds in ("kitti", "nuscenes"):
    out_npz = os.path.join(OUT, f"metrics_v2x_real_{ds}_evening45_s5.npz")
    if os.path.exists(out_npz):
        print(f"[seeds5] {ds}: exists, skipping", flush=True)
        continue
    cfg2 = Config(); cfg2.results_dir = OUT
    t0 = time.time()
    R.run(cfg=cfg2, seeds=[2029, 2030, 2031, 2032, 2033], dataset=ds,
          rounds=T, num_vehicles=min(180, tr["veh_xy"].shape[1]), merge=True,
          schemes=["Caching-assisted", "V2V-aware", "mmFedMC", "AutoFed",
                   "Proposed"],
          out_name=f"metrics_v2x_real_{ds}_evening45_s5.npz")
    print(f"[seeds5] {ds}: done in {(time.time() - t0) / 60:.0f} min",
          flush=True)
print("[seeds5] ALL DONE", flush=True)
