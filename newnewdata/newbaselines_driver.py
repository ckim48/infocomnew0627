"""Phase-2 queue: the 6 NEW baselines (PRoPHET, Spray-and-Wait, Cached-DFL
LFU/random, No-comm lower bound, Full-contact upper bound) on the original
evening-peak window, KITTI, 3 seeds, 250 rounds -- directly comparable to the
Table-I runs (same seeds, same protocol). Waits politely until the phase-1
overnight driver finishes so the GPU is never shared."""
import os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "newnewdata"
LOG1 = os.path.join(OUT, "overnight.log")
SENTINEL = "=== overnight driver done ==="
NEW = ["PRoPHET", "SprayWait", "Caching-LFU", "Caching-rand",
       "NoComm", "FullContact"]


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


out_npz = os.path.join(OUT, "metrics_v2x_real_kitti_newbase.npz")
if os.path.exists(out_npz):
    log("newbase metrics exist, nothing to do")
    sys.exit(0)

log("waiting for the phase-1 overnight driver to finish ...")
while not (os.path.exists(LOG1) and SENTINEL in open(LOG1).read()):
    time.sleep(120)
log("phase-1 done; starting the new-baseline comparison")

from sim.config import Config
import sim.run_v2x_real as R

try:
    cfg = Config(); cfg.results_dir = OUT
    t0 = time.time()
    R.run(cfg=cfg, seeds=[2026, 2027, 2028], dataset="kitti", rounds=250,
          num_vehicles=180, schemes=NEW, merge=True,
          out_name="metrics_v2x_real_kitti_newbase.npz")
    log(f"newbase: done in {(time.time() - t0) / 60:.0f} min")
except Exception:
    log(f"newbase: FAILED\n{traceback.format_exc()}")
