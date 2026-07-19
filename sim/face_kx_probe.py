"""
Empirical counterpart of Proposition 2 (diminishing copy value): sweep the
copy cap K_x on the abstract Seoul backend and measure the delivered value.
If the placement utility is (approximately) submodular in the copy set, the
value gained by allowing more copies must saturate: adoptions/useful
deliveries increase in K_x with diminishing increments.

Run:  python3 -m sim.face_kx_probe   ->  results/face_kx_probe.npz
"""

import numpy as np
import torch

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .mfl import MultimodalFL
from .simulator import make_modality_availability, make_arch_assignment
from .v2x_trace import build_v2x_trace
from .face import FACE

KS = (1, 2, 4, 8, 16, 32)
SEEDS = (2026, 2027, 2028)
ROUNDS = 250


def main():
    cfg = Config()
    cfg.num_vehicles = 180
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    adopt = np.zeros((len(KS), len(SEEDS)))
    usat = np.zeros((len(KS), len(SEEDS)))
    acc = np.zeros((len(KS), len(SEEDS)))
    for ki, K in enumerate(KS):
        for si, sd in enumerate(SEEDS):
            cfg.face_K_tickets = K
            avail = make_modality_availability(
                cfg, np.random.default_rng(sd + 7))
            arch = make_arch_assignment(
                cfg, np.random.default_rng(sd + 11), avail)
            mfl = MultimodalFL(cfg, np.random.default_rng(sd), avail)
            mfl.arch = arch
            alg = FACE(cfg, mfl, mob, seed=sd)
            n_adopt, us = 0, []
            for k in range(ROUNDS):
                mob.k = k % mob.Krounds
                mfl.local_train()
                alg.run_round(k)
                n_adopt += alg._n_adopt
                us.append(getattr(alg, "last_useful_sat", 0.0))
            adopt[ki, si] = n_adopt
            usat[ki, si] = float(np.mean(us))
            acc[ki, si] = mfl.mean_accuracy()
            print(f"  K={K:<3d} seed={sd}  adoptions={n_adopt:5d}  "
                  f"usat={usat[ki, si]:.4f}  acc={acc[ki, si]:.3f}",
                  flush=True)
    np.savez("results/face_kx_probe.npz", ks=np.array(KS),
             adopt=adopt, usat=usat, acc=acc)
    print("saved results/face_kx_probe.npz")


if __name__ == "__main__":
    main()
