"""Third mobility window: Monday MORNING-peak Seoul V2X trace + full 6-scheme
real-backend comparison. Fired by cron on 2026-07-27 08:30 KST; self-guarding,
so a re-fire (cron matches the date every year) is a no-op once done.

Collects 45 minutes (~265 snapshots at 10 s), so K >= T=250 and the FL run
uses every snapshot exactly once -- NO cyclic replay. Comparing this
full-horizon window against the replayed evening/night windows empirically
answers the "does replay distort mobility realism" concern."""
import os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "newnewdata"
RAW = "data/gangnam/seoul_v2x_trace_mon_morning.npz"
CACHE = os.path.join(OUT, "v2x_seoul_trace_morning.npz")
OUT_NPZ = os.path.join(OUT, "metrics_v2x_real_kitti_morning.npz")


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


if os.path.exists(OUT_NPZ):
    log("morning metrics exist, nothing to do")
    sys.exit(0)

try:
    if not os.path.exists(RAW):
        from sim.seoul_v2x import collect_trace
        log("collecting Monday morning-peak V2X window (2700 s, no-replay run)")
        collect_trace(duration_s=2700, interval_s=10, out=RAW)
    import numpy as np
    from sim.config import Config
    from sim.v2x_trace import build_v2x_trace
    import sim.run_v2x_real as R

    cfg = Config(); cfg.results_dir = OUT
    tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
    n = tr["veh_xy"].shape[1]
    log(f"morning trace: N={n}, K={tr['veh_seg'].shape[0]}")
    if n < 60:
        os.remove(CACHE)
        tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW, min_cov=0.7)
        n = tr["veh_xy"].shape[1]
        log(f"morning trace rebuilt with min_cov=0.7: N={n}")
    R.build_v2x_trace = lambda c: build_v2x_trace(
        c, cache=CACHE, v2x_file=RAW, verbose=False)
    cfg2 = Config(); cfg2.results_dir = OUT
    t0 = time.time()
    R.run(cfg=cfg2, seeds=[2026, 2027, 2028], dataset="kitti", rounds=250,
          num_vehicles=min(180, n), merge=True,
          out_name="metrics_v2x_real_kitti_morning.npz")
    log(f"morning run done in {(time.time() - t0) / 60:.0f} min")
except Exception:
    log(f"FAILED\n{traceback.format_exc()}")
