"""
Component ablation of FACE on the REAL KITTI backend (RealMFL), so the
ablation numbers live on the same scale as Table I (test accuracy of the
main 250-round protocol). Variants mirror the paper figure:

  FACE (full) / w/o relay ferrying (caching) / w/o demand /
  w/o future value

Output: results/metrics_face_ablation_real_kitti.npz with
{variant}__{acc,poor,txmb,redund}_all  (seeds x rounds).

Run:  python3 -m sim.face_ablation_real
"""

import os
import numpy as np
import torch

from .config import Config
from .run_v2x_real import _prepare_v2x
from .real_fl import RealMFL, _prep_data, _device
from .face import FACE
from .simulator import make_modality_availability, make_arch_assignment

VARIANTS = {
    "FACE (full)":        {},
    "w/o relay ferrying": dict(use_relay=False),
    "w/o demand":         dict(use_demand=False),
    "w/o future value":   dict(use_future=False),
}
METRICS = ["acc", "poor", "txmb", "redund"]
ROUNDS = 250
SEEDS = (2026, 2027, 2028)


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
    data = _prep_data(cfg, cfg.seed, dataset="kitti", min_class_count=0)

    stacks = {v: {m: [] for m in METRICS} for v in VARIANTS}
    for sd in SEEDS:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        arch = make_arch_assignment(cfg, np.random.default_rng(sd + 11),
                                    avail)
        for name, flags in VARIANTS.items():
            torch.manual_seed(sd)
            mfl = RealMFL(cfg, np.random.default_rng(sd), avail, data,
                          device=device)
            mfl.arch = arch
            alg = FACE(cfg, mfl, mob, "Proposed", seed=sd, flags=flags)
            pm = mfl.poor_mask()
            hist = {m: [] for m in METRICS}
            for k in range(ROUNDS):
                kk = k % mob.Krounds
                mob.k = kk
                mfl.local_train()
                mfl.refresh_strengths()
                g = gammas[kk] if alg.flags.get("use_dis") \
                    or alg.flags.get("cache_policy") == "psi" \
                    else np.zeros(mob.N)
                alg.run_round(k, g, gamma_eval=gammas[kk])
                accs = mfl.evaluate("test")
                hist["acc"].append(float(accs.mean()))
                hist["poor"].append(float(accs[pm].mean()) if pm.any()
                                    else 0.0)
                hist["txmb"].append(alg.last_tx_mb)
                nd = getattr(alg, "last_deliv", 0)
                nu_ = getattr(alg, "last_deliv_useful", 0)
                hist["redund"].append((nd - nu_) / nd if nd else 0.0)
            for m in METRICS:
                stacks[name][m].append(hist[m])
            print(f"  [seed {sd}] {name:18s} acc {hist['acc'][-1]:.3f} "
                  f"poor {hist['poor'][-1]:.3f} "
                  f"MB/rd {np.mean(hist['txmb']):.0f}", flush=True)
            del mfl, alg
            if device == "cuda":
                torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        for m in METRICS:
            out[f"{v}__{m}_all"] = np.array(stacks[v][m])
    np.savez("results/metrics_face_ablation_real_kitti.npz", **out)
    print("saved results/metrics_face_ablation_real_kitti.npz")


if __name__ == "__main__":
    main()
