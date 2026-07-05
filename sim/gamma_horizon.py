"""
Gamma-vs-horizon probe: does the road-segment-aware future-contact score
Gamma_j (Eq. 10) get *more* predictive as we look further ahead over the road
graph?  This isolates the two ingredients of Gamma -- road **segmentation**
(propagation along the segment topology) and the prediction **horizon** H --
without touching the FL backend, so it is cheap and dataset-independent
(the Seoul V2X mobility is shared across KITTI / nuScenes).

For each round k we score every vehicle with three predictors and compare the
ranking against the *realized* future co-locations over the next W rounds,
restricted to vehicles NOT already in range at k -- the store-carry-forward
reach that only mobility prediction can anticipate:

  * Gamma(H): the paper's score computed with horizon H = 1..H_max.
  * blind density: cohort count on each vehicle's current segment
    (traffic-aware but topology-blind -- the foil named in hgat.py).

Quality is Spearman rank correlation, averaged over sampled rounds.
Caches to results/gamma_horizon.npz for the figure.
"""
import os
import numpy as np
import torch
from scipy.stats import spearmanr

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .v2x_trace import build_v2x_trace
from .hgat import train_hgat, future_contact_scores


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute(cfg=None, device=None, horizons=(1, 2, 3, 4), window=8,
            stride=2, out="results/gamma_horizon.npz"):
    cfg = cfg or Config()
    device = device or _device()
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    K, N = mob.Krounds, mob.N
    R = cfg.comm_range
    xy = mob.veh_xy                                     # [K, N, 2]
    print(f"[gamma-horizon] K={K} N={N} R={R} window={window}")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)

    def adj(t):
        d = np.linalg.norm(xy[t][:, None, :] - xy[t][None, :, :], axis=2)
        A = d <= R
        np.fill_diagonal(A, False)
        return A

    def realized(k, beyond):
        """distinct future co-locations over k+1..k+W. beyond=True keeps only
        vehicles NOT in range now (the store-carry-forward reach)."""
        seen = np.zeros((N, N), dtype=bool)
        for t in range(k + 1, min(k + window, K - 1) + 1):
            seen |= adj(t)
        if beyond:
            seen &= ~adj(k)
        np.fill_diagonal(seen, False)
        return seen.sum(1).astype(float)

    def blind_density(k):
        seg = mob.veh_seg[k]
        cnt = np.bincount(seg, minlength=road.V)
        return cnt[seg].astype(float)

    REGIMES = ("all", "beyond")
    rounds = list(range(3, K - window, stride))
    gamma_c = {r: {H: [] for H in horizons} for r in REGIMES}
    blind_c = {r: [] for r in REGIMES}
    for k in rounds:
        mob.k = k
        gts = {r: realized(k, r == "beyond") for r in REGIMES}
        if gts["beyond"].std() == 0:
            continue
        bd = blind_density(k)
        preds = {}
        for H in horizons:
            cfg.H_max = H
            preds[H] = future_contact_scores(cfg, road, mob, model, road_ei,
                                             device=device)
        for r in REGIMES:
            gt = gts[r]
            if gt.std() == 0:
                continue
            if bd.std() > 0:
                blind_c[r].append(spearmanr(bd, gt).correlation)
            for H in horizons:
                if np.std(preds[H]) > 0:
                    gamma_c[r][H].append(spearmanr(preds[H], gt).correlation)

    H = np.array(horizons, float)
    kw = {}
    for r in REGIMES:
        sfx = "" if r == "beyond" else "_all"   # 'beyond' keeps legacy names
        kw[f"gamma_mean{sfx}"] = np.array([np.mean(gamma_c[r][h]) for h in horizons])
        kw[f"gamma_std{sfx}"] = np.array([np.std(gamma_c[r][h]) for h in horizons])
        kw[f"blind_mean{sfx}"] = float(np.mean(blind_c[r]))
        kw[f"blind_std{sfx}"] = float(np.std(blind_c[r]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, horizons=H, window=window,
             n_rounds=len(blind_c["beyond"]), **kw)
    for r in REGIMES:
        sfx = "" if r == "beyond" else "_all"
        print(f"[gamma-horizon] {r:7s} blind={kw['blind_mean'+sfx]:.3f}  "
              + "  ".join(f"H{h}={m:.3f}"
                          for h, m in zip(horizons, kw['gamma_mean'+sfx])))
    print(f"[gamma-horizon] saved {out} (n_rounds={len(blind_c['beyond'])})")
    return out


if __name__ == "__main__":
    compute()
