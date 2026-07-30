"""EDGE-RSU centralized references (out-of-model): the CFL server lives at
K=4 roadside units placed at the densest 300 m zones of the evening45
trace (rule-based placement). Per round, the top-10% covered clients
upload (unicast, billed per encoder); the aggregate is broadcast ONCE per
RSU per modality (billed K_RSU x encoder size); only vehicles within the
V2I range receive/adopt that round. Inter-RSU aggregation uses wired
backhaul (not billed, disclosed).

Run with NUSC_FINE=1 NUSC_KEEPALL=1 for the fine nuScenes task.
Out: newnewdata/metrics_cfl_edge_{scheme}.npz
"""
import os, sys

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
ROUNDS = int(os.environ.get("CFL_ROUNDS", 400))
K_RSU = 4
R_V2I = 300.0                     # V2I range (m)
device = _device()

tr = np.load("newnewdata/v2x_seoul_trace_evening45.npz", allow_pickle=True)
XY = tr["veh_xy"]                                  # rounds x N x 2 (m)
KR, NV = XY.shape[0], XY.shape[1]

# rule-based RSU placement: the K densest 300 m zones by mean occupancy
cell = 300.0
gx = np.floor(XY[..., 0] / cell).astype(int)
gy = np.floor(XY[..., 1] / cell).astype(int)
from collections import Counter
cnt = Counter(zip(gx.ravel().tolist(), gy.ravel().tolist()))
rsus = np.array([[ (cx + 0.5) * cell, (cy + 0.5) * cell]
                 for (cx, cy), _ in cnt.most_common(K_RSU)])
print(f"[edge] RSUs at zones: {[(c, n) for c, n in cnt.most_common(K_RSU)]}")

COVER = np.zeros((KR, NV), dtype=bool)
for k in range(KR):
    dist = np.linalg.norm(XY[k][:, None, :] - rsus[None], axis=2).min(1)
    COVER[k] = dist <= R_V2I
print(f"[edge] mean coverage {COVER.mean():.2%}")


def run_one(scheme, ds, seed):
    cfg = Config()
    cfg.num_vehicles = NV
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
        cov = COVER[k % KR]
        mfl.local_train()
        mfl.refresh_strengths()
        score = np.array([np.mean([mfl.strength[(i, r)]
                                   for r in mfl.avail[i]])
                          if mfl.avail[i] and cov[i] else -1.0
                          for i in range(mfl.N)])
        top = [i for i in np.argsort(score)[-n_top:] if score[i] > -1.0]
        agg, mb = {}, 0.0
        for i in top:
            if scheme == "mmfedmc":
                mods = [max(mfl.avail[i],
                            key=lambda r: mfl.strength[(i, r)]
                            / cfg.encoder_size[r])]
            else:
                mods = list(mfl.avail[i])
            for r in mods:
                sds, ws = agg.setdefault(r, ([], []))
                sds.append(mfl.enc[i][r].state_dict())
                ws.append(mfl.Dmr(i, r))
                mb += cfg.encoder_size[r]                   # uplink unicast
        for r, (sds, ws) in agg.items():
            g = _fedavg(sds, ws)
            mb += cfg.encoder_size[r] * K_RSU               # broadcast/RSU
            for i in range(mfl.N):
                if r in mfl.avail[i] and cov[i]:            # covered only
                    mfl.enc[i][r].load_state_dict(
                        {kk: v.to(device) for kk, v in g.items()})
        acc_h.append(float(mfl.evaluate("test").mean()))
        mb_h.append(mb)
    pm = mfl.poor_mask()
    poor = float(mfl.evaluate("test")[pm].mean())
    print(f"[edge-{scheme}/{ds}/s{seed}] final {np.mean(acc_h[-20:]):.4f} "
          f"poor {poor:.4f} comm {sum(mb_h)/1024:.0f} GB", flush=True)
    del mfl
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.array(acc_h), np.array(mb_h), poor


for scheme in ("mmfedmc", "autofed"):
    out_path = f"newnewdata/metrics_cfl_edge_{scheme}.npz"
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping"); continue
    out = {}
    for ds in ("kitti", "nuscenes"):
        A, M, P = [], [], []
        for sd in SEEDS:
            a, m, p = run_one(scheme, ds, sd)
            A.append(a); M.append(m); P.append(p)
        out[f"{ds}__acc_all"] = np.stack(A)
        out[f"{ds}__mb_all"] = np.stack(M)
        out[f"{ds}__poor"] = np.array(P)
    np.savez_compressed(out_path, **out)
    print("saved", out_path, flush=True)
print("[edge] ALL DONE", flush=True)
