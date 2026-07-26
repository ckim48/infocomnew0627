"""Native CENTRALIZED references for the two CFL-origin baselines
(out-of-model, infrastructure traffic), 3 seeds x 2 datasets:

- mmfedmc: top-10% clients by contribution upload their single best
  contribution-per-cost modality encoder; per-modality FedAvg; broadcast.
- autofed: top-10% clients by encoder quality upload ALL their modality
  encoders; per-modality FedAvg; broadcast. (Imputation module has no
  faithful counterpart here and is not ported -- disclosed in text.)

Both use blind adoption (no evaluation gate), as in their original
server-based deployments; architecture-homogeneous fleet assumed.
Output: newnewdata/metrics_cfl_{scheme}.npz with
{ds}__acc_all [seeds x 250], {ds}__mb_all, {ds}__poor [seeds].
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

from sim.config import Config
from sim.real_fl import RealMFL, _prep_data, _fedavg, _device
from sim.simulator import make_modality_availability, make_arch_assignment

SEL = 0.10
SEEDS = (2026, 2027, 2028)
ROUNDS = 250
device = _device()


def run_one(scheme, ds, seed):
    cfg = Config()
    cfg.num_vehicles = 180
    if ds == "nuscenes":
        cfg.modalities = ["camera", "lidar", "radar"]
        cfg.modality_prob = {"camera": 1.0, "lidar": 0.85, "radar": 0.7}
    else:
        cfg.modalities = ["camera", "lidar"]
        cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    torch.manual_seed(seed); np.random.seed(seed)
    data = _prep_data(cfg, cfg.seed, dataset=ds,
                      min_class_count=800 if ds == "nuscenes" else 0)
    avail = make_modality_availability(cfg, np.random.default_rng(seed + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(seed + 11), avail)
    mfl = RealMFL(cfg, np.random.default_rng(seed), avail, data, device=device)
    mfl.arch = arch
    n_top = max(int(SEL * mfl.N), 1)
    acc_h, mb_h = [], []
    for k in range(ROUNDS):
        mfl.local_train()
        mfl.refresh_strengths()
        score = np.array([np.mean([mfl.strength[(i, r)]
                                   for r in mfl.avail[i]])
                          if mfl.avail[i] else 0.0 for i in range(mfl.N)])
        top = np.argsort(score)[-n_top:]
        agg, mb = {}, 0.0
        for i in top:
            if scheme == "mmfedmc":
                mods = [max(mfl.avail[i],
                            key=lambda r: mfl.strength[(i, r)]
                            / cfg.encoder_size[r])]
            else:                              # autofed: all modalities
                mods = list(mfl.avail[i])
            for r in mods:
                sds, ws = agg.setdefault(r, ([], []))
                sds.append(mfl.enc[i][r].state_dict())
                ws.append(mfl.Dmr(i, r))
                mb += cfg.encoder_size[r]                   # uplink
        for r, (sds, ws) in agg.items():
            g = _fedavg(sds, ws)
            for i in range(mfl.N):
                if r in mfl.avail[i]:
                    mfl.enc[i][r].load_state_dict(
                        {kk: v.to(device) for kk, v in g.items()})
                    mb += cfg.encoder_size[r]               # downlink
        acc_h.append(float(mfl.evaluate("test").mean()))
        mb_h.append(mb)
    pm = mfl.poor_mask()
    poor = float(mfl.evaluate("test")[pm].mean())
    print(f"[{scheme}/{ds}/s{seed}] final {np.mean(acc_h[-20:]):.4f} "
          f"poor {poor:.4f} comm {sum(mb_h)/1024:.0f} GB", flush=True)
    del mfl
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.array(acc_h), np.array(mb_h), poor


for scheme in ("mmfedmc", "autofed"):
    out_path = f"newnewdata/metrics_cfl_{scheme}.npz"
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping")
        continue
    out = {}
    for ds in ("kitti", "nuscenes"):
        A, M, P = [], [], []
        for sd in SEEDS:
            a, m, p = run_one(scheme, ds, sd)
            A.append(a); M.append(m); P.append(p)
        out[f"{ds}__acc_all"] = np.stack(A)
        out[f"{ds}__mb_all"] = np.stack(M)
        out[f"{ds}__poor"] = np.array(P)
    np.savez(out_path, **out)
    print("saved", out_path, flush=True)
