"""
Component ablation of the NEW FACE system model on the real Seoul V2X trace
(abstract MDFL backend for speed). Each variant disables one mechanism of the
revised algorithm; everything else (mobility, modality availability, data
strengths, seeds) is identical, so differences are attributable to the
mechanism:

  FACE (full)        all mechanisms of the new model
  w/o future value   F == 0 (Eq. 16 off): forwarding only to direct ESVs,
                     no encoder ferrying via relays
  w/o coverage       Omega == 1 (Eq. 15 off): copies ignore what background
                     copies already cover -> redundant placement
  w/o tickets        K_x = inf: unbounded replication, no custody transfer
  w/o ridge gain     mean-bandit gain instead of the optimistic ridge (Eq. 10)
  w/o cache refresh  LRU eviction instead of the coverage knapsack (Eq. 19)
"""

import os
import numpy as np
import torch

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .mfl import MultimodalFL
from .simulator import make_modality_availability, make_arch_assignment
from .v2x_trace import build_v2x_trace
from .face import FACE

VARIANTS = {
    "FACE (full)":        {},
    "w/o relay ferrying": dict(use_relay=False),
    "w/o demand":         dict(use_demand=False),
    "w/o future value":   dict(use_future=False),
    "w/o coverage":       dict(use_coverage=False),
    "w/o tickets":        dict(use_tickets=False),
    "w/o ticket split":   dict(use_split=False),
    "w/o ridge gain":     dict(use_ridge=False),
    "w/o cache refresh":  dict(refresh="lru"),
}

METRICS = ["acc", "poor", "tx", "txmb", "beyond", "adopt"]


def _partition_strengths(cfg, mfl, mob, sd):
    """Motivation scenario (Sec. II-B): encoder-carrier vehicles (CVs) are
    spatially concentrated -- strong encoders only among vehicles that START
    in the west half of the region, so strong encoders must be ferried to
    the demand in the east. Overall CV fraction is kept at cfg.frac_good."""
    x0 = mob.veh_xy[0, :, 0]
    west = x0 <= np.quantile(x0, 0.5)
    rng = np.random.default_rng(sd + 13)
    p_good = cfg.frac_good / max(west.mean(), 1e-9)
    for i in range(cfg.num_vehicles):
        good_i = bool(west[i]) and (rng.random() < p_good)
        for r in mfl.avail[i]:
            q = rng.uniform(0.80, 1.00) if good_i else rng.uniform(0.10, 0.35)
            D = mfl.D[(i, r)]
            s = float(np.clip(q * (0.70 + 0.30 * (D / cfg.data_max)),
                              0.05, 0.97))
            mfl.Q[(i, r)] = q
            mfl.strength[(i, r)] = s
            mfl.theta[(i, r)] = s
            mfl.acquired[(i, r)] = {i}


def run(seeds=(2026, 2027, 2028), num_vehicles=180, rounds=None,
        variants=None, merge=False, partitioned=False):
    cfg = Config()
    cfg.num_vehicles = num_vehicles
    os.makedirs(cfg.results_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    K = rounds or mob.Krounds
    tag = "_part" if partitioned else ""
    print(f"[face-abl{tag}] Seoul V2X: |V|={road.V} N={mob.N} rounds={K}")

    todo = {n: f for n, f in VARIANTS.items()
            if variants is None or n in variants}
    stacks = {v: {m: [] for m in METRICS} for v in todo}
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        arch = make_arch_assignment(cfg, np.random.default_rng(sd + 11), avail)
        for name, flags in todo.items():
            rng = np.random.default_rng(sd)        # paired conditions
            mfl = MultimodalFL(cfg, rng, avail)
            mfl.arch = arch                        # architecture families chi
            if partitioned:
                _partition_strengths(cfg, mfl, mob, sd)
            alg = FACE(cfg, mfl, mob, seed=sd, flags=flags)
            hist = {m: [] for m in METRICS}
            for k in range(K):
                mob.k = k % mob.Krounds
                mfl.local_train()
                alg.run_round(k)
                hist["acc"].append(mfl.mean_accuracy())
                hist["poor"].append(mfl.poor_accuracy())
                hist["tx"].append(alg._n_tx)
                hist["txmb"].append(alg.last_tx_mb)
                hist["beyond"].append(alg._n_beyond_adopt)
                hist["adopt"].append(alg._n_adopt)
            for m in METRICS:
                stacks[name][m].append(hist[m])
            print(f"  [seed {sd}] {name:18s} acc {hist['acc'][-1]:.3f} "
                  f"poor {hist['poor'][-1]:.3f} tx/rd {np.mean(hist['tx']):.1f} "
                  f"MB/rd {np.mean(hist['txmb']):.0f} "
                  f"beyond {np.sum(hist['beyond'])}", flush=True)

    path = os.path.join(cfg.results_dir,
                        f"metrics_face_ablation_v2x{tag}.npz")
    out = dict(np.load(path)) if (merge and os.path.exists(path)) else {}
    results = {}
    for v in todo:
        results[v] = {}
        for m in METRICS:
            arr = np.stack(stacks[v][m])
            results[v][m] = arr.mean(0)
            results[v][m + "_std"] = arr.std(0)
            results[v][m + "_all"] = arr
    out.update({f"{v}__{k}": val for v, d in results.items()
                for k, val in d.items()})
    np.savez(path, **out)
    print(f"=== FACE component ablation (Seoul V2X{tag}) ===")
    print(f"{'variant':20s} {'acc':>6s} {'poor':>6s} {'tx/rd':>7s} "
          f"{'MB/rd':>7s} {'beyond':>7s}")
    for v in todo:
        d = results[v]
        print(f"{v:20s} {d['acc'][-1]:6.3f} {d['poor'][-1]:6.3f} "
              f"{np.mean(d['tx']):7.1f} {np.mean(d['txmb']):7.0f} "
              f"{np.sum(d['beyond']):7.0f}")
    print("  saved", path)
    return results


if __name__ == "__main__":
    import sys
    kw = {}
    args = [a for a in sys.argv[1:] if a != "part"]
    kw["partitioned"] = "part" in sys.argv[1:]
    if len(args) > 0:
        kw["rounds"] = int(args[0])
    if len(args) > 1:
        kw["seeds"] = tuple(int(s) for s in args[1].split(","))
    run(**kw)
