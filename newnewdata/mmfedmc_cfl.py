"""Native CENTRALIZED mmFedMC reference (out-of-model, infrastructure-based):
per round, the top-10% contribution clients upload their highest
contribution-per-cost modality encoder to a server; the server FedAvg-
aggregates per modality and broadcasts the global encoder to every vehicle
(blind adoption, no evaluation gate -- as in the original CFL deployment;
architecture-homogeneous fleet assumed as in the original paper).
Communication = uplink + per-vehicle downlink bytes over infrastructure.

Purpose: price the centralized alternative next to the V2V schemes
(reference sentence in Sec. Evaluation), NOT a table row.
"""
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

# run after the NoComm nuScenes probe releases the GPU
while not os.path.exists("newnewdata/metrics_probe_nocomm_nusc.npz"):
    time.sleep(60)

from sim.config import Config
from sim.real_fl import RealMFL, _prep_data, _fedavg, _device
from sim.simulator import make_modality_availability, make_arch_assignment

SEL = 0.10          # client-selection fraction per round
SEED = 2026
ROUNDS = 250
device = _device()
out = {}
for ds in ("kitti", "nuscenes"):
    cfg = Config()
    cfg.num_vehicles = 180
    if ds == "nuscenes":
        cfg.modalities = ["camera", "lidar", "radar"]
        cfg.modality_prob = {"camera": 1.0, "lidar": 0.85, "radar": 0.7}
    else:
        cfg.modalities = ["camera", "lidar"]
        cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    torch.manual_seed(SEED); np.random.seed(SEED)
    data = _prep_data(cfg, cfg.seed, dataset=ds,
                      min_class_count=800 if ds == "nuscenes" else 0)
    avail = make_modality_availability(cfg, np.random.default_rng(SEED + 7))
    arch = make_arch_assignment(cfg, np.random.default_rng(SEED + 11), avail)
    mfl = RealMFL(cfg, np.random.default_rng(SEED), avail, data, device=device)
    mfl.arch = arch
    n_top = max(int(SEL * mfl.N), 1)
    acc_h, mb_h = [], []
    t0 = time.time()
    for k in range(ROUNDS):
        mfl.local_train()
        mfl.refresh_strengths()
        # contribution-ranked client selection (top-SEL by strength)
        score = np.array([np.mean([mfl.strength[(i, r)]
                                   for r in mfl.avail[i]])
                          if mfl.avail[i] else 0.0 for i in range(mfl.N)])
        top = np.argsort(score)[-n_top:]
        agg, mb = {}, 0.0
        for i in top:
            r = max(mfl.avail[i],
                    key=lambda r: mfl.strength[(i, r)] / cfg.encoder_size[r])
            sds, ws = agg.setdefault(r, ([], []))
            sds.append(mfl.enc[i][r].state_dict())
            ws.append(mfl.Dmr(i, r))
            mb += cfg.encoder_size[r]                       # uplink
        for r, (sds, ws) in agg.items():
            if not sds:
                continue
            g = _fedavg(sds, ws)
            for i in range(mfl.N):
                if r in mfl.avail[i]:
                    mfl.enc[i][r].load_state_dict(
                        {kk: v.to(device) for kk, v in g.items()})
                    mb += cfg.encoder_size[r]               # downlink
        accs = mfl.evaluate("test")
        acc_h.append(float(accs.mean()))
        mb_h.append(mb)
        if (k + 1) % 50 == 0:
            print(f"  [{ds}] round {k+1}/{ROUNDS} acc {acc_h[-1]:.3f} "
                  f"cum {sum(mb_h)/1024:.1f} GB", flush=True)
    pm = mfl.poor_mask()
    out[ds] = dict(acc=np.array(acc_h), mb=np.array(mb_h),
                   poor=float(mfl.evaluate("test")[pm].mean()))
    print(f"[{ds}] final acc {np.mean(acc_h[-20:]):.4f}  poor {out[ds]['poor']:.4f}  "
          f"total comm {sum(mb_h)/1024:.1f} GB  ({(time.time()-t0)/60:.0f} min)",
          flush=True)
    del mfl
    if device == "cuda":
        torch.cuda.empty_cache()
np.savez("newnewdata/metrics_mmfedmc_cfl.npz",
         **{f"{ds}__{k}": v for ds, d in out.items() for k, v in d.items()})
print("saved newnewdata/metrics_mmfedmc_cfl.npz", flush=True)
