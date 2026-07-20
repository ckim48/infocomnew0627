"""
REAL-backend setup-sensitivity sweeps answering the "the 15%/clean-data
setup is hand-made for FACE" critique: carrier fraction, poor-data quality
ceiling, poor-vehicle data quantity, and LiDAR availability, each run on
the real KITTI backend (FACE vs the strongest baseline, 150 rounds,
seed 2026). Results: results/sens_setup/metrics_{tag}_{val}.npz.

Run:  python3 -m sim.face_setup_sens
"""

import os
import numpy as np

from .config import Config
from .run_v2x_real import run

SCHEMES = ["Proposed", "Learning-aware"]
ROUNDS = 150
SEED = 2026
OUT = "sens_setup"

# (tag, value, config-mutator); the (0.15, 0.3-ceiling, 8-frame, 0.85)
# default point is shared across sweeps as metrics_base.npz
SWEEPS = [
    ("carrier", 0.05, lambda c: setattr(c, "frac_good", 0.05)),
    ("carrier", 0.10, lambda c: setattr(c, "frac_good", 0.10)),
    ("carrier", 0.25, lambda c: setattr(c, "frac_good", 0.25)),
    ("carrier", 0.40, lambda c: setattr(c, "frac_good", 0.40)),
    ("poorq", 0.5, lambda c: setattr(c, "poor_q_range", (0.1, 0.5))),
    ("poorq", 0.7, lambda c: setattr(c, "poor_q_range", (0.1, 0.7))),
    ("skew", 30, lambda c: setattr(c, "poor_size_range", (26, 34))),
    ("skew", 60, lambda c: setattr(c, "poor_size_range", (56, 64))),
    ("plidar", 0.6, lambda c: setattr(
        c, "modality_prob_override", {"camera": 1.0, "lidar": 0.6})),
    ("plidar", 1.0, lambda c: setattr(
        c, "modality_prob_override", {"camera": 1.0, "lidar": 1.0})),
]


def main():
    os.makedirs(os.path.join("results", OUT), exist_ok=True)
    # shared default point
    base = os.path.join("results", OUT, "metrics_base.npz")
    if not os.path.exists(base):
        print("[sens] base config ...", flush=True)
        run(cfg=Config(), seeds=[SEED], dataset="kitti", rounds=ROUNDS,
            schemes=SCHEMES, out_name=os.path.join(OUT, "metrics_base.npz"),
            merge=True, record_class=False)
    for tag, val, mut in SWEEPS:
        out = os.path.join(OUT, f"metrics_{tag}_{val}.npz")
        if os.path.exists(os.path.join("results", out)):
            print(f"[sens] skip existing {out}", flush=True)
            continue
        print(f"[sens] {tag}={val} ...", flush=True)
        cfg = Config()
        mut(cfg)
        run(cfg=cfg, seeds=[SEED], dataset="kitti", rounds=ROUNDS,
            schemes=SCHEMES, out_name=out, merge=True, record_class=False)
    # symlink-style copies of the shared default into each sweep's naming
    b = np.load(base)
    for tag, dv in (("carrier", 0.15), ("poorq", 0.3),
                    ("skew", 8), ("plidar", 0.85)):
        p = os.path.join("results", OUT, f"metrics_{tag}_{dv}.npz")
        if not os.path.exists(p):
            np.savez(p, **dict(b))
    print("[sens] done")


if __name__ == "__main__":
    main()
