"""
Contact-opportunity statistics of the real Seoul-Gangnam V2X trace. Pure
geometry on the recorded trajectories (no learning), so it directly quantifies
the V2V dissemination opportunity that motivates store-carry-forward:

  * instantaneous contact is sparse (few neighbors per round),
  * but distinct contacts accumulate over time, and
  * a single relay hop (carry-forward) reaches most of the fleet.

Caches to results/contact_stats.npz for tab_contacts() in seoul_pack.
"""
import os
import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy.sparse import csr_matrix

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .v2x_trace import build_v2x_trace


def compute(cfg=None, out="results/contact_stats.npz"):
    cfg = cfg or Config()
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    K, N = mob.Krounds, mob.N
    R = cfg.comm_range
    xy = mob.veh_xy
    dt = mob.dt

    def adj(t):
        d = np.linalg.norm(xy[t][:, None, :] - xy[t][None, :, :], axis=2)
        A = d <= R
        np.fill_diagonal(A, False)
        return A

    deg = np.array([adj(t).sum(1) for t in range(K)])       # [K, N]
    met = np.zeros((N, N), dtype=bool)                      # union contact graph
    cum = np.zeros(K)                                       # mean distinct peers
    seen = np.zeros((N, N), dtype=bool)
    for t in range(K):
        a = adj(t)
        met |= a
        seen |= a
        np.fill_diagonal(seen, False)
        cum[t] = seen.sum(1).mean()
    np.fill_diagonal(met, False)
    uniq = met.sum(1)                                       # direct peers / trace

    G = met.astype(int)
    r2 = (G + G @ G) > 0                                    # <=2-hop reachability
    np.fill_diagonal(r2, False)
    reach2 = r2.sum(1)

    _, lab = csgraph.connected_components(csr_matrix(met), directed=False)
    comp_max = int(np.bincount(lab).max())

    denom = max(N - 1, 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        N=N, K=K, dt=dt, comm_range=R, span_min=K * dt / 60.0,
        deg_mean=deg.mean(), deg_median=np.median(deg), deg_max=deg.max(),
        frac_rounds_contact=(deg > 0).mean(),
        uniq_mean=uniq.mean(), uniq_median=np.median(uniq), uniq_max=uniq.max(),
        uniq_frac=uniq.mean() / denom,
        reach2_mean=reach2.mean(), reach2_frac=reach2.mean() / denom,
        comp_max=comp_max, comp_frac=comp_max / N,
        cum=cum,
    )
    print(f"[contact-stats] N={N} K={K} span~{K*dt/60:.0f}min R={R}m")
    print(f"  degree mean={deg.mean():.1f}  contact-rounds={100*(deg>0).mean():.0f}%")
    print(f"  unique/15min={uniq.mean():.1f} ({100*uniq.mean()/denom:.0f}%)  "
          f"2-hop={reach2.mean():.1f} ({100*reach2.mean()/denom:.0f}%)  "
          f"largest-comp={comp_max}/{N} ({100*comp_max/N:.0f}%)")
    print(f"[contact-stats] saved {out}")
    return out


if __name__ == "__main__":
    compute()
