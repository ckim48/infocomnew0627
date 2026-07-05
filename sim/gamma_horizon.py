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
from sklearn.metrics import roc_auc_score

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

    def rs(pred, gt):
        return spearmanr(pred, gt).correlation

    def auc(pred, gt):
        """P(pred ranks a true high-contact vehicle above a low one); the
        top-quartile by realized contacts are the positives. 0.5 = random."""
        lab = (gt >= np.quantile(gt, 0.75)).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            return np.nan
        return roc_auc_score(lab, pred)

    METRICS = {"rs": rs, "auc": auc}
    REGIMES = ("all", "beyond")
    rounds = list(range(3, K - window, stride))
    # collector[metric][regime]['blind' | H] -> list over rounds
    coll = {m: {r: {"blind": [], **{H: [] for H in horizons}}
                for r in REGIMES} for m in METRICS}
    for k in rounds:
        mob.k = k
        gts = {r: realized(k, r == "beyond") for r in REGIMES}
        if gts["beyond"].std() == 0:
            continue
        preds = {"blind": blind_density(k)}
        for H in horizons:
            cfg.H_max = H
            preds[H] = future_contact_scores(cfg, road, mob, model, road_ei,
                                             device=device)
        for r in REGIMES:
            gt = gts[r]
            if gt.std() == 0:
                continue
            for name, p in preds.items():
                if np.std(p) == 0:
                    continue
                for m, fn in METRICS.items():
                    coll[m][r][name].append(fn(p, gt))

    H = np.array(horizons, float)
    kw = {}
    for m in METRICS:
        mp = "" if m == "rs" else f"{m}_"        # 'rs' keeps legacy key names
        for r in REGIMES:
            sfx = "" if r == "beyond" else "_all"
            c = coll[m][r]
            kw[f"gamma_{mp}mean{sfx}"] = np.array(
                [np.nanmean(c[h]) for h in horizons])
            kw[f"gamma_{mp}std{sfx}"] = np.array(
                [np.nanstd(c[h]) for h in horizons])
            kw[f"blind_{mp}mean{sfx}"] = float(np.nanmean(c["blind"]))
            kw[f"blind_{mp}std{sfx}"] = float(np.nanstd(c["blind"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_rounds = len(coll["rs"]["beyond"]["blind"])
    np.savez(out, horizons=H, window=window, n_rounds=n_rounds, **kw)
    for m in METRICS:
        mp = "" if m == "rs" else f"{m}_"
        for r in REGIMES:
            sfx = "" if r == "beyond" else "_all"
            print(f"[gamma-horizon] {m:4s} {r:7s} "
                  f"blind={kw[f'blind_{mp}mean'+sfx]:.3f}  "
                  + "  ".join(f"H{h}={x:.3f}" for h, x
                              in zip(horizons, kw[f'gamma_{mp}mean'+sfx])))
    print(f"[gamma-horizon] saved {out} (n_rounds={n_rounds})")
    return out


if __name__ == "__main__":
    compute()
