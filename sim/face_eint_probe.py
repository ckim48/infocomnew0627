"""
Empirical interaction-error probe (Sec. IV bound constant eps_int):
run FACE on the real KITTI backend with record_eint=True so every accepted
aggregation set A logs (|A|, sum of solo marginal gains, joint gain).
Output: results/face_eint_probe.npz  (nset, solo_sum, joint).

Run:  python3 -m sim.face_eint_probe
"""

import numpy as np
import torch

from .config import Config
from .run_v2x_real import _prepare_v2x
from .real_fl import RealMFL, _prep_data, _device
from .face import FACE
from .simulator import make_modality_availability, make_arch_assignment

ROUNDS = 250
SEED = 2026


def main():
    cfg = Config()
    cfg.num_vehicles = 180
    cfg.face_ttl = 15
    cfg.face_K_tickets = 6
    cfg.face_Qpub = 1
    cfg.face_lam = 0.001
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    cfg.record_eint = True
    device = _device()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    road, mob, gammas = _prepare_v2x(cfg, device)
    data = _prep_data(cfg, cfg.seed, dataset="kitti", min_class_count=0)

    avail = make_modality_availability(cfg, np.random.default_rng(SEED + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(SEED + 11), avail)
    torch.manual_seed(SEED)
    mfl = RealMFL(cfg, np.random.default_rng(SEED), avail, data,
                  device=device)
    mfl.arch = arch
    alg = FACE(cfg, mfl, mob, "Proposed", seed=SEED)
    for k in range(ROUNDS):
        kk = k % mob.Krounds
        mob.k = kk
        mfl.local_train()
        mfl.refresh_strengths()
        g = gammas[kk] if alg.flags.get("use_dis") \
            or alg.flags.get("cache_policy") == "psi" else np.zeros(mob.N)
        alg.run_round(k, g, gamma_eval=gammas[kk])
        if (k + 1) % 25 == 0:
            n = len(getattr(mfl, "eint", []))
            print(f"  round {k+1}/{ROUNDS}  eint events {n}", flush=True)
    ev = np.array(getattr(mfl, "eint", []), dtype=np.float64)
    if ev.size == 0:
        print("  [warn] no eint events recorded")
        ev = np.zeros((0, 3))
    np.savez("results/face_eint_probe.npz",
             nset=ev[:, 0], solo_sum=ev[:, 1], joint=ev[:, 2])
    m = ev[:, 0] >= 2
    if m.any():
        e = np.abs(ev[m, 1] - ev[m, 2])
        print(f"  saved results/face_eint_probe.npz  n={len(ev)} "
              f"(|A|>=2: {int(m.sum())})  eps_int mean {e.mean():.4f} "
              f"p95 {np.percentile(e, 95):.4f}")


if __name__ == "__main__":
    main()
