"""Overnight queue addressing the INFOCOM review risks (started Fri 2026-07-24 23:3x KST).

1) Second real mobility window: collect a Fri *late-night* Seoul V2X trace
   (contrast to the weekday evening-peak window of Table I), then run the
   full 6-scheme real-backend comparison on it (3 seeds).
2) Real-backend FACE-only sensitivity sweeps on the original window:
   gamma-noise robustness, comm range, zone size, link rate, vehicle density.

All outputs land in newnewdata/ (results_dir); results/ and Figures/ are
untouched (merge=True skips the paper-figure plotting path).
Restart-safe: finished steps are skipped by their output file.
"""
import os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

OUT = "newnewdata"
NIGHT_RAW = "data/gangnam/seoul_v2x_trace_fri_night.npz"
NIGHT_CACHE = os.path.join(OUT, "v2x_seoul_trace_night.npz")


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def step_collect():
    if os.path.exists(NIGHT_RAW):
        log(f"collect: {NIGHT_RAW} exists, skipping")
        return
    from sim.seoul_v2x import collect_trace
    log("collect: polling Seoul V2X API for 900 s (Fri late-night window)")
    collect_trace(duration_s=900, interval_s=10, out=NIGHT_RAW)
    log("collect: done")


def step_night():
    out_npz = os.path.join(OUT, "metrics_v2x_real_kitti_night.npz")
    if os.path.exists(out_npz):
        log("night: metrics exist, skipping")
        return
    import sim.run_v2x_real as R
    from sim.config import Config
    from sim.v2x_trace import build_v2x_trace
    cfg = Config(); cfg.results_dir = OUT
    tr = build_v2x_trace(cfg, cache=NIGHT_CACHE, v2x_file=NIGHT_RAW)
    n = tr["veh_xy"].shape[1]
    log(f"night trace: N={n}, K={tr['veh_seg'].shape[0]}")
    if n < 60:                      # sparse late-night cohort: relax coverage
        os.remove(NIGHT_CACHE)
        tr = build_v2x_trace(cfg, cache=NIGHT_CACHE, v2x_file=NIGHT_RAW,
                             min_cov=0.7)
        n = tr["veh_xy"].shape[1]
        log(f"night trace rebuilt with min_cov=0.7: N={n}")
    R.build_v2x_trace = lambda c: build_v2x_trace(
        c, cache=NIGHT_CACHE, v2x_file=NIGHT_RAW, verbose=False)
    cfg2 = Config(); cfg2.results_dir = OUT
    t0 = time.time()
    R.run(cfg=cfg2, seeds=[2026, 2027, 2028], dataset="kitti", rounds=250,
          num_vehicles=min(180, n), merge=True,
          out_name="metrics_v2x_real_kitti_night.npz")
    log(f"night: done in {(time.time() - t0) / 60:.0f} min")


# (tag, Config overrides, gamma-noise sigma in units of std(Gamma), num_vehicles)
SWEEPS = [
    ("noise05",  {},                          0.5,  None),
    ("noise10",  {},                          1.0,  None),
    ("range100", {"comm_range": 100.0},       None, None),
    ("range200", {"comm_range": 200.0},       None, None),
    ("zone150",  {"face_zone_cell": 150.0},   None, None),
    ("zone600",  {"face_zone_cell": 600.0},   None, None),
    ("rate6",    {"tx_rate_mbps": 6.0},       None, None),
    ("rate24",   {"tx_rate_mbps": 24.0},      None, None),
    ("n90",      {},                          None, 90),
    ("n135",     {},                          None, 135),
]


def step_sweeps():
    import sim.run_v2x_real as R
    from sim.config import Config
    from sim.v2x_trace import build_v2x_trace
    from sim.hgat import future_contact_scores as TRUE_FC
    # back to the original (evening-peak) window, cached copy in newnewdata/
    R.build_v2x_trace = lambda c: build_v2x_trace(c, verbose=False)
    for tag, over, sigma, nveh in SWEEPS:
        out = f"metrics_sweep_{tag}.npz"
        if os.path.exists(os.path.join(OUT, out)):
            log(f"sweep {tag}: exists, skipping")
            continue
        try:
            cfg = Config(); cfg.results_dir = OUT
            for k, v in over.items():
                setattr(cfg, k, v)
            if sigma is not None:
                rng = np.random.default_rng(777)
                def noisy(c, road, mob, model, ei, device="cpu",
                          _s=sigma, _r=rng):
                    g = TRUE_FC(c, road, mob, model, ei, device=device)
                    return g + _r.normal(0.0, _s * (g.std() + 1e-9),
                                         size=g.shape)
                R.future_contact_scores = noisy
            else:
                R.future_contact_scores = TRUE_FC
            t0 = time.time()
            R.run(cfg=cfg, seeds=[2026], dataset="kitti", rounds=250,
                  num_vehicles=nveh or 180, schemes=["Proposed"],
                  merge=True, out_name=out)
            log(f"sweep {tag}: done in {(time.time() - t0) / 60:.1f} min "
                f"(wall-clock incl. GAT warmup + per-round eval)")
        except Exception:
            log(f"sweep {tag}: FAILED\n{traceback.format_exc()}")
    R.future_contact_scores = TRUE_FC


if __name__ == "__main__":
    log("=== overnight driver start ===")
    for step in (step_collect, step_night, step_sweeps):
        try:
            step()
        except Exception:
            log(f"{step.__name__}: FAILED\n{traceback.format_exc()}")
    log("=== overnight driver done ===")
