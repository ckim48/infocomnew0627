"""K_x (copy-ticket budget) sweep under the strict matching constraint.

Diagnosis: with half-duplex single-peer matching (eq:matching_constraint)
exchange opportunities per round dropped for every scheme, so K_x = 6 copies
per version is now too tight -- the 'w/o copy tickets' (K = inf) ablation
variant beats full FACE by ~4.5pp while spending +24 MB/rd. This sweep finds
the operating point where bounded replication recovers the accuracy of
unbounded replication at lower communication cost.

Usage: python3 -m experiments.sweeps_seoul.kx_sweep <K> [seed] [rounds]
(one process per K so the sweep parallelizes across cores)
"""

import sys
import numpy as np

from sim.config import Config
from sim.mobility import RoadNetwork, MobilitySim
from sim.mfl import MultimodalFL
from sim.simulator import make_modality_availability
from sim.v2x_trace import build_v2x_trace
from sim.face import FACE


def run_one(Kt, sd=2026, rounds=250):
    cfg = Config()
    cfg.num_vehicles = 180
    if Kt > 0:
        cfg.face_K_tickets = Kt
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
    rng = np.random.default_rng(sd)
    mfl = MultimodalFL(cfg, rng, avail)
    flags = {} if Kt > 0 else dict(use_tickets=False)
    alg = FACE(cfg, mfl, mob, seed=sd, flags=flags)
    accs, poors, mbs, beyond = [], [], [], 0
    for k in range(rounds):
        mob.k = k % mob.Krounds
        mfl.local_train()
        alg.run_round(k)
        accs.append(mfl.mean_accuracy())
        poors.append(mfl.poor_accuracy())
        mbs.append(alg.last_tx_mb)
        beyond += alg._n_beyond_adopt
    tag = f"K={Kt}" if Kt > 0 else "K=inf"
    print(f"[kx] {tag:6s} seed={sd} acc={np.mean(accs[-20:]):.3f} "
          f"poor={np.mean(poors[-20:]):.3f} MB/rd={np.mean(mbs):.0f} "
          f"beyond={beyond}", flush=True)


if __name__ == "__main__":
    Kt = int(sys.argv[1])                       # 0 -> K = inf (no tickets)
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    run_one(Kt, sd, rounds)
