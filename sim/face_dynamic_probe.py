"""
Dynamic sensing-environment experiment (the "demand changes as vehicles
move through different environments" claim, Sec. I): the map is split into
four fixed regions with distinct sensing profiles -- clean / low-light
(camera) / rain-noise (camera+radar) / sparse-LiDAR -- and every round each
vehicle's LOCAL TRAINING data is corrupted according to the region it is
currently driving through (RealMFL.env_profile). Demand then shifts with
mobility, not only with training progress.

Records per scheme: fleet accuracy and the share of useful deliveries whose
receiver is currently environment-degraded. Output:
results/face_dynamic_probe.npz.

Run:  python3 -m sim.face_dynamic_probe
"""

import numpy as np
import torch

from .config import Config
from .run_v2x_real import _prepare_v2x
from .real_fl import RealMFL, _prep_data, _device
from .face import FACE
from .simulator import make_modality_availability, make_arch_assignment
from .v2x_trace import build_v2x_trace

SCHEMES = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]
ROUNDS = 250
SEED = 2026


def _env_profiles(trace):
    """Per-round env profile from fixed map quadrants: 0 clean (NE),
    1 low-light (NW), 2 rain/noise (SE), 3 sparse-LiDAR (SW)."""
    xy = trace["veh_xy"]                       # K x N x 2
    mx, my = np.median(xy[..., 0]), np.median(xy[..., 1])
    west, south = xy[..., 0] < mx, xy[..., 1] < my
    prof = np.zeros(xy.shape[:2], dtype=np.int64)
    prof[west & ~south] = 1
    prof[~west & south] = 2
    prof[west & south] = 3
    return prof


def _degraded(prof_k, avail):
    """Mask of vehicles whose CURRENT region degrades one of their own
    modalities (camera always available; LiDAR only if equipped)."""
    N = len(prof_k)
    out = np.zeros(N, dtype=bool)
    for i in range(N):
        p = int(prof_k[i])
        out[i] = (p in (1, 2)) or (p == 3 and "lidar" in avail[i])
    return out


def main():
    cfg = Config()
    cfg.num_vehicles = 180
    cfg.face_ttl = 15
    cfg.face_K_tickets = 6
    cfg.face_Qpub = 1
    cfg.face_lam = 0.001
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = _device()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    road, mob, gammas = _prepare_v2x(cfg, device)
    trace = build_v2x_trace(cfg)
    prof = _env_profiles(trace)
    data = _prep_data(cfg, cfg.seed, dataset="kitti", min_class_count=0)

    out = {}
    avail = make_modality_availability(cfg, np.random.default_rng(SEED + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(SEED + 11), avail)
    deg_frac = np.array([_degraded(prof[k % mob.Krounds], avail).mean()
                         for k in range(ROUNDS)])
    for scheme in SCHEMES:
        torch.manual_seed(SEED)
        mfl = RealMFL(cfg, np.random.default_rng(SEED), avail, data,
                      device=device)
        mfl.arch = arch
        alg = FACE(cfg, mfl, mob, scheme, seed=SEED)
        acc_h, dd_h = [], []
        for k in range(ROUNDS):
            kk = k % mob.Krounds
            mob.k = kk
            mfl.env_profile = prof[kk]         # environment follows mobility
            mfl.local_train()
            mfl.refresh_strengths()
            g = gammas[kk] if alg.flags.get("use_dis") \
                or alg.flags.get("cache_policy") == "psi" \
                else np.zeros(mob.N)
            alg.run_round(k, g, gamma_eval=gammas[kk])
            accs = mfl.evaluate("test")
            acc_h.append(float(accs.mean()))
            deg = _degraded(prof[kk], avail)
            rcv = list(getattr(alg, "last_useful_receivers", []))
            dd_h.append(float(np.mean([deg[j] for j in rcv]))
                        if rcv else np.nan)
            if (k + 1) % 50 == 0:
                print(f"  [{scheme}] round {k+1}/{ROUNDS} "
                      f"acc {acc_h[-1]:.3f}", flush=True)
        out[f"{scheme}__acc"] = np.array(acc_h)
        dd = np.array(dd_h)
        # forward-fill rounds with no deliveries for a clean curve
        idx = np.where(~np.isnan(dd))[0]
        if len(idx):
            dd = np.interp(np.arange(len(dd)), idx, dd[idx])
        out[f"{scheme}__deg_deliv"] = dd
        print(f"  [{scheme}] final acc {acc_h[-1]:.3f} "
              f"deg-deliv {np.nanmean(dd):.3f}", flush=True)
        del mfl, alg
        if device == "cuda":
            torch.cuda.empty_cache()
    out["deg_frac"] = deg_frac
    np.savez("results/face_dynamic_probe.npz", **out)
    print("saved results/face_dynamic_probe.npz")


if __name__ == "__main__":
    main()
