"""
Print measured mobility/communication statistics from the InTAS trace that
populate the simulation-parameters table (Table~\\ref{tab:setup}).
"""

import numpy as np
from .config import Config
from .intas_trace import get_or_build_trace
from .mobility import RoadNetwork, MobilitySim
from .motivation import _adj_stack, _contact_durations


def main(comm_range=100.0, rate_mbps=16.0):
    cfg = Config()
    cfg.num_vehicles = 150
    cfg.K = 150
    cfg.comm_range = comm_range
    tr = get_or_build_trace(cfg, f"results/intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    road = RoadNetwork(tr)
    mob = MobilitySim(cfg, road, tr)
    A = _adj_stack(cfg, mob)
    dt = mob.dt
    deg = A.sum(2)
    durs = _contact_durations(A, dt)
    rate = rate_mbps / 8.0                      # MB/s
    sizes = cfg.encoder_size

    print("=== InTAS measured statistics (range %.0f m, %.0f Mbps) ===" % (comm_range, rate_mbps))
    print(f"  avg vehicle speed     : {mob.veh_speed.mean()*3.6:.1f} km/h "
          f"({mob.veh_speed.mean():.2f} m/s)")
    print(f"  avg V2V degree        : {deg.mean():.2f}  (max {int(deg.max())})")
    print(f"  contacts observed     : {len(durs)}")
    print(f"  contact duration      : mean {durs.mean():.1f} s, median {np.median(durs):.1f} s")
    print(f"  data per contact      : {durs.mean()*rate:.1f} MB (avg)")
    print(f"  encoder size S_m,r    : " +
          ", ".join(f"{k} {v} MB" for k, v in sizes.items()) +
          f"  (avg {np.mean(list(sizes.values())):.1f} MB)")


if __name__ == "__main__":
    main()
