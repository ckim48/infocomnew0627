"""
End-to-end MDFL simulator over the real InTAS trace.

Trains the hierarchical-GAT mobility predictor once, then runs the proposed
scheme and the three baselines on identical learning conditions (same modality
availability, data, optima, and initialisation) and records per-round metrics.
"""

import os
import numpy as np
import torch

from .config import Config, SCHEMES
from .intas_trace import get_or_build_trace
from .mobility import RoadNetwork, MobilitySim
from .mfl import MultimodalFL
from .hgat import train_hgat, future_contact_scores
from .algorithm import CachingForwarding


def make_modality_availability(cfg, rng):
    """Per-vehicle sensing-modality sets. With cfg.vehicle_types set, vehicles
    are drawn from a typed sensor-suite mixture reflecting real fleets (e.g.,
    vision-only vehicles without LiDAR, camera+radar ADAS suites, and full
    robotaxi suites); otherwise falls back to independent per-modality draws."""
    types = getattr(cfg, "vehicle_types", None)
    avail = []
    if types:
        w = np.array([t[0] for t in types], dtype=float)
        w = w / w.sum()
        for i in range(cfg.num_vehicles):
            kind = rng.choice(len(types), p=w)
            s = set(types[kind][1]) & set(cfg.modalities)
            avail.append(s or {cfg.modalities[0]})
        return avail
    for i in range(cfg.num_vehicles):
        s = [r for r in cfg.modalities if rng.random() < cfg.modality_prob[r]]
        if not s:
            s = [rng.choice(cfg.modalities)]
        avail.append(set(s))
    return avail


def make_arch_assignment(cfg, rng, avail):
    """Architecture-family label per (vehicle, modality). Vehicles with
    high computational capability run the large encoder family (1), others
    the lightweight family (0); parameter-level aggregation is possible only
    within the same family (compatibility chi in the system model)."""
    if not getattr(cfg, "use_arch_families", True):
        return None
    arch = {}
    for i in range(cfg.num_vehicles):
        fam = 1 if rng.random() < cfg.arch_high_frac else 0
        for r in avail[i]:
            arch[(i, r)] = fam
    return arch


def prepare(cfg, device):
    """Build the InTAS trace, train the GAT, and precompute Gamma (seed-shared)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print("[1/4] Loading InTAS mobility trace ...")
    cache_path = os.path.join(cfg.results_dir, f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache_path, begin=34050.0, dt=2.0, warmup_s=480.0)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    print(f"      road segments |V|={road.V}, vehicles N={mob.N}, rounds K={mob.Krounds}")

    print("[2/4] Training hierarchical GAT mobility predictor ...")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)

    print("[3/4] Predicting future contact opportunities Gamma ...")
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei, device=device))
    return road, mob, np.array(gammas)


def run_schemes(cfg, mob, gammas, seed, verbose=True):
    """Run all schemes for one seed on identical learning conditions."""
    avail_rng = np.random.default_rng(seed + 7)
    modality_avail = make_modality_availability(cfg, avail_rng)
    results = {}
    for scheme in SCHEMES:
        rng = np.random.default_rng(seed)              # identical init per scheme
        mfl = MultimodalFL(cfg, rng, modality_avail)
        alg = CachingForwarding(cfg, mfl, mob, scheme, seed=seed)

        loss_h, acc_h, tail_h, tx_h, q_h = [], [], [], [], []
        for k in range(mob.Krounds):
            mob.k = k
            mfl.local_train()
            g = gammas[k] if alg.flags["use_dis"] or alg.flags["cache_policy"] == "psi" \
                else np.zeros(mob.N)
            selected = alg.run_round(k, g)
            loss_h.append(mfl.mean_val_loss())
            acc_h.append(mfl.mean_accuracy())
            tail_h.append(mfl.poor_accuracy())
            tx_h.append(len(selected))
            q_h.append(np.mean(list(alg.Q.values())))
        results[scheme] = dict(loss=np.array(loss_h), acc=np.array(acc_h),
                               tail=np.array(tail_h), tx=np.array(tx_h), qlen=np.array(q_h))
        if verbose:
            print(f"      [seed {seed}] {scheme:16s} acc {acc_h[-1]:.3f}  "
                  f"tail {tail_h[-1]:.3f}  tx/round {np.mean(tx_h):.1f}")
    return results


def run_all(cfg=None, device=None, seeds=None):
    """Run the full pipeline; average metrics over `seeds` (default [seed])."""
    cfg = cfg or Config()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.results_dir, exist_ok=True)
    seeds = seeds if seeds is not None else [cfg.seed]

    road, mob, gammas = prepare(cfg, device)

    print(f"[4/4] Running MDFL schemes over seeds {seeds} ...")
    metric_keys = ["loss", "acc", "tail", "tx", "qlen"]
    stacks = {s: {m: [] for m in metric_keys} for s in SCHEMES}
    for sd in seeds:
        res = run_schemes(cfg, mob, gammas, sd)
        for s in SCHEMES:
            for m in metric_keys:
                stacks[s][m].append(res[s][m])

    results = {}
    for s in SCHEMES:
        results[s] = {}
        for m in metric_keys:
            arr = np.stack(stacks[s][m])
            results[s][m] = arr.mean(0)
            results[s][m + "_std"] = arr.std(0)
        print(f"      {s:16s} final acc {results[s]['acc'][-1]:.3f} "
              f"(±{results[s]['acc_std'][-1]:.3f})  tail {results[s]['tail'][-1]:.3f}")

    np.savez(os.path.join(cfg.results_dir, "metrics.npz"),
             **{f"{s}__{k}": v for s, d in results.items() for k, v in d.items()})
    return results, cfg


if __name__ == "__main__":
    run_all(seeds=[2026, 2027, 2028])
