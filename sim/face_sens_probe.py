"""
Sensitivity sweeps of FACE on the abstract Seoul backend (3 seeds each):
vehicle density N (trace-column subsampling), cache capacity Lambda,
and future-contact horizon H. The copy-cap K_x sweep comes from
sim/face_kx_probe.py. Output: results/face_sens_probe.npz.

Run:  python3 -m sim.face_sens_probe
"""

import numpy as np
import torch

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .mfl import MultimodalFL
from .simulator import make_modality_availability, make_arch_assignment
from .v2x_trace import build_v2x_trace
from .face import FACE

SEEDS = (2026, 2027, 2028)
ROUNDS = 250
NS = (60, 100, 140, 180)
LAMS = (15.0, 30.0, 45.0, 60.0, 90.0)
HS = (1, 2, 4, 6, 8, 12)


def _one(cfg, trace, sd):
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(sd + 11), avail)
    mfl = MultimodalFL(cfg, np.random.default_rng(sd), avail)
    mfl.arch = arch
    alg = FACE(cfg, mfl, mob, seed=sd)
    for k in range(ROUNDS):
        mob.k = k % mob.Krounds
        mfl.local_train()
        alg.run_round(k)
    return mfl.mean_accuracy(), mfl.poor_accuracy()


def main():
    base = Config()
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    full = build_v2x_trace(Config())
    out = {}
    for n in NS:                                     # vehicle density
        for si, sd in enumerate(SEEDS):
            cfg = Config()
            cfg.num_vehicles = n
            trace = dict(full)
            if n < full["veh_xy"].shape[1]:
                idx = np.sort(np.random.default_rng(4242).choice(
                    full["veh_xy"].shape[1], n, replace=False))
                for kk in ("veh_seg", "veh_xy", "veh_speed"):
                    trace[kk] = full[kk][:, idx]
            a, p = _one(cfg, trace, sd)
            out.setdefault("N", []).append((n, sd, a, p))
            print(f"  N={n:<4d} seed={sd}  acc={a:.3f} poor={p:.3f}",
                  flush=True)
    for lam in LAMS:                                 # cache capacity
        for sd in SEEDS:
            cfg = Config()
            cfg.num_vehicles = 180
            cfg.cache_capacity_mb = lam
            a, p = _one(cfg, dict(full), sd)
            out.setdefault("LAM", []).append((lam, sd, a, p))
            print(f"  Lam={lam:<5.0f} seed={sd}  acc={a:.3f} poor={p:.3f}",
                  flush=True)
    for h in HS:                                     # future horizon
        for sd in SEEDS:
            cfg = Config()
            cfg.num_vehicles = 180
            cfg.face_H = h
            a, p = _one(cfg, dict(full), sd)
            out.setdefault("H", []).append((h, sd, a, p))
            print(f"  H={h:<3d} seed={sd}  acc={a:.3f} poor={p:.3f}",
                  flush=True)
    np.savez("results/face_sens_probe.npz",
             **{k: np.array(v) for k, v in out.items()})
    print("saved results/face_sens_probe.npz")


if __name__ == "__main__":
    main()
