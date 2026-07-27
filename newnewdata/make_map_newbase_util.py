"""mmFedMC / AutoFed statistical-utility map, matching fig_seoul_map_compact.

Recomputes the ABSTRACT backend (q_eff utility) for the two new baselines
under the exact fig_seoul_map protocol: seed 2026 cohort, N=180, 250 wrapped
rounds, snapshot k=249, same GAT/Gamma; positions/headings and the utility
colour scale (0.2-1.0) come from/match results/v2x_map_cache.npz.

Out:  results/v2x_map_cache_newbase.npz (acc_mmFedMC, acc_AutoFed)
      Figures/fig_seoul_map_newbase.{png,pdf}  (1x2, 'Achieved statistical
      utility' colourbar -- replaces the earlier accuracy-metric version)
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

SCHEMES_NEW = ["mmFedMC", "AutoFed"]
CACHE_NEW = "results/v2x_map_cache_newbase.npz"

if not os.path.exists(CACHE_NEW):
    import torch
    from sim.config import Config
    from sim.mobility import RoadNetwork, MobilitySim
    from sim.hgat import train_hgat, future_contact_scores
    from sim.v2x_trace import build_v2x_trace
    from sim.map_viz import _run_one

    cfg = Config()
    cfg.num_vehicles = 180
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    rounds, snap_k = 250, 249

    print("  [util-newbase] training GAT + Gamma ...")
    model, road_ei = train_hgat(cfg, road, mob, device="cpu",
                                warmup_rounds=30)
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei,
                                            device="cpu"))
    gammas = np.array(gammas)

    utils = {}
    for s in SCHEMES_NEW:
        print(f"  [util-newbase] running {s} ...")
        utils[s], _ = _run_one(cfg, mob, gammas, s, snap_k, rounds=rounds)
    np.savez(CACHE_NEW, **{f"acc_{s}": utils[s] for s in SCHEMES_NEW})
    print("  saved", CACHE_NEW)

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "make_map_newbase", os.path.join(ROOT, "newnewdata/make_map_newbase.py"))
_mmn = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mmn)
draw, VALS = _mmn.draw, _mmn.VALS

d = np.load(CACHE_NEW)
for s in SCHEMES_NEW:
    VALS[s] = d[f"acc_{s}"]

draw(SCHEMES_NEW, 1, 2, "fig_seoul_map_newbase", slim=True,
     vmin=0.2, vmax=1.0, cbar_label="Achieved statistical utility",
     mean_label="mean utility")
