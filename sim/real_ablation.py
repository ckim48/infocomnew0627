"""
Component ablation of FACE on the REAL multimodal FL backend (KITTI over the
InTAS mobility): each variant disables one mechanism of the full algorithm,
everything else (data, seeds, mobility, operating point) identical to the
main comparison in sim/real_fl.py. Results feed Tables/tab_ablation.tex.
"""

import os
import numpy as np
import torch

from .config import Config
from .algorithm import CachingForwarding, SCHEME_FLAGS
from .simulator import prepare, make_modality_availability
from .real_fl import RealMFL, _prep_data, _device, main_config

FULL = SCHEME_FLAGS["Proposed"]
VARIANTS = {
    "FACE (full)":  FULL,
    "w/o caching":  {**FULL, "carry": False, "cache_policy": "own"},
    "w/o demand":   {**FULL, "demand_aware": False, "cache_policy": "lru"},
    "w/o queue":    {**FULL, "use_queue": False},
}


def run(cfg=None, seeds=(2026, 2027, 2028), dataset="kitti", device=None):
    cfg = cfg or main_config()
    cfg.modalities = ["camera", "lidar"]              # match run_real_all
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = device or _device()
    road, mob, gammas = prepare(cfg, device)
    data = _prep_data(cfg, cfg.seed, dataset=dataset)

    metric_keys = ["acc", "poor", "tx"]
    stacks = {v: {m: [] for m in metric_keys} for v in VARIANTS}
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        for name, flags in VARIANTS.items():
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            alg = CachingForwarding(cfg, mfl, mob, "Proposed", seed=sd)
            alg.flags = flags
            pm = mfl.poor_mask()
            acc_h, poor_h, tx_h = [], [], []
            for k in range(mob.Krounds):
                mob.k = k
                mfl.local_train()
                mfl.refresh_strengths()
                g = gammas[k] if flags["use_dis"] or flags["cache_policy"] == "psi" \
                    else np.zeros(mob.N)
                selected = alg.run_round(k, g)
                accs = mfl.evaluate("test")
                acc_h.append(float(accs.mean()))
                poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
                tx_h.append(len(selected))
            for m, h in zip(metric_keys, [acc_h, poor_h, tx_h]):
                stacks[name][m].append(h)
            print(f"  [ablation seed {sd}] {name:14s} acc {acc_h[-1]:.3f} "
                  f"poor {poor_h[-1]:.3f} tx/round {np.mean(tx_h):.1f}", flush=True)
            del mfl, alg
            if device == "cuda":
                torch.cuda.empty_cache()

    results = {}
    for v in VARIANTS:
        results[v] = {}
        for m in metric_keys:
            arr = np.stack(stacks[v][m])
            results[v][m] = arr.mean(0)
            results[v][m + "_std"] = arr.std(0)
            results[v][m + "_all"] = arr
    np.savez(os.path.join(cfg.results_dir, f"metrics_real_ablation_{dataset}.npz"),
             **{f"{v}__{k}": val for v, d in results.items() for k, val in d.items()})
    print(f"=== REAL ablation ({dataset}) final ===")
    for v in VARIANTS:
        print(f"  {v:14s} acc {results[v]['acc'][-1]:.3f} "
              f"poor {results[v]['poor'][-1]:.3f}")
    return results


if __name__ == "__main__":
    run()
