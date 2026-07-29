"""Component ablation (4 figure variants) on the SPARSE late-night window,
5 seeds, T=250 wrapped -- the regime the future-contact utility targets.

Out: newnewdata/metrics_face_ablation_v2x.npz (night; the peak-era file in
results/ is untouched)
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sim.config import Config
from sim.v2x_trace import build_v2x_trace
import sim.face_ablation as FA

RAW = "data/gangnam/seoul_v2x_trace_fri_night.npz"
CACHE = "newnewdata/v2x_seoul_trace_night.npz"
FA.build_v2x_trace = lambda cfg: build_v2x_trace(
    cfg, cache=CACHE, v2x_file=RAW, verbose=False)


def _cfg():
    c = Config(); c.results_dir = "newnewdata"
    return c


FA.Config = _cfg
FA.run(seeds=(2026, 2027, 2028, 2029, 2030), num_vehicles=123, rounds=250,
       variants=["FACE (full)", "w/o relay ferrying", "w/o demand",
                 "w/o future value"])
