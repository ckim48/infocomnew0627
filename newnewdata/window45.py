"""45-minute NO-REPLAY window runner: collect ~265 snapshots (>= T=250, so
every FL round uses a distinct observed snapshot) and run the full 6-scheme
x 3-seed comparison on BOTH datasets.

Fired by cron on Mon 2026-07-27 (weekday, matching the original windows):
  18:42 KST  ->  python3 newnewdata/window45.py evening45   (evening peak)
  23:36 KST  ->  python3 newnewdata/window45.py night45     (late-night off-peak)

Self-guarding per output file, so re-fires are no-ops once done."""
import os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

NAME = sys.argv[1] if len(sys.argv) > 1 else "evening45"
assert NAME in ("evening45", "night45",
                "evening45_sat", "night45_sat"), NAME
OUT = "newnewdata"
RAW = f"data/gangnam/seoul_v2x_trace_{NAME}.npz"
CACHE = os.path.join(OUT, f"v2x_seoul_trace_{NAME}.npz")


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] [{NAME}] {msg}", flush=True)


try:
    if not os.path.exists(RAW):
        from sim.seoul_v2x import collect_trace
        log("collecting 2700 s (~265 snapshots, no-replay horizon)")
        collect_trace(duration_s=2700, interval_s=10, out=RAW)

    import numpy as np
    from sim.config import Config
    from sim.v2x_trace import build_v2x_trace
    import sim.run_v2x_real as R

    cfg = Config(); cfg.results_dir = OUT
    tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW)
    n = tr["veh_xy"].shape[1]
    log(f"trace: N={n}, K={tr['veh_seg'].shape[0]}")
    if n < 60:                       # strict 45-min presence: relax coverage
        os.remove(CACHE)
        tr = build_v2x_trace(cfg, cache=CACHE, v2x_file=RAW, min_cov=0.7)
        n = tr["veh_xy"].shape[1]
        log(f"trace rebuilt with min_cov=0.7: N={n}")
    R.build_v2x_trace = lambda c: build_v2x_trace(
        c, cache=CACHE, v2x_file=RAW, verbose=False)

    for ds in ("kitti", "nuscenes"):
        out_npz = os.path.join(OUT, f"metrics_v2x_real_{ds}_{NAME}.npz")
        if os.path.exists(out_npz):
            log(f"{ds}: metrics exist, skipping")
            continue
        cfg2 = Config(); cfg2.results_dir = OUT
        t0 = time.time()
        R.run(cfg=cfg2, seeds=[2026, 2027, 2028], dataset=ds, rounds=250,
              num_vehicles=min(180, n), merge=True,
              out_name=f"metrics_v2x_real_{ds}_{NAME}.npz")
        log(f"{ds}: done in {(time.time() - t0) / 60:.0f} min")
except Exception:
    log(f"FAILED\n{traceback.format_exc()}")
