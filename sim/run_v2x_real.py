"""
REAL multimodal FL (real SGD encoders + FedAvg + real KITTI classification
accuracy) over the real Seoul V2X mobility trace. This is the technically
rigorous counterpart of sim/run_v2x.py (which uses the abstract coverage proxy):
here accuracy is genuine test accuracy and exhibits real convergence.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, SCHEMES
from .real_fl import REAL_SCHEMES
from .mobility import RoadNetwork, MobilitySim
from .hgat import train_hgat, future_contact_scores
from .algorithm import CachingForwarding
from .simulator import make_modality_availability
from .real_fl import RealMFL, _prep_data, _device
from .v2x_trace import build_v2x_trace
from .plotting import STYLE as STY, disp


def _prepare_v2x(cfg, device):
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    print(f"      Seoul V2X: |V|={road.V}, N={mob.N}, K={mob.Krounds}")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei, device=device))
    return road, mob, np.array(gammas)


def run(cfg=None, seeds=None, device=None, num_vehicles=180, dataset="kitti",
        rounds=250, min_class_count=None, schemes=None, merge=False):
    """Run REAL FL until convergence. `rounds` may exceed the mobility trace
    length: the Seoul V2X window is replayed cyclically (steady-state traffic),
    while FL keeps training/propagating so the accuracy curve plateaus."""
    cfg = cfg or Config()
    cfg.num_vehicles = num_vehicles
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = device or _device()
    seeds = seeds or [cfg.seed]
    os.makedirs(cfg.results_dir, exist_ok=True)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print("[1/3] Building real Seoul V2X mobility + GAT ...")
    road, mob, gammas = _prepare_v2x(cfg, device)
    total = rounds or mob.Krounds
    print(f"      running {total} FL rounds (trace K={mob.Krounds}, "
          f"replayed cyclically)")
    print("[2/3] Loading real KITTI multimodal data ...")
    if min_class_count is None:                # match the InTAS setup
        min_class_count = 800 if dataset == "nuscenes" else 0
    data = _prep_data(cfg, cfg.seed, dataset=dataset,
                      min_class_count=min_class_count)

    todo = schemes or REAL_SCHEMES
    keys = ["acc", "poor", "tx", "util"]
    stacks = {s: {m: [] for m in keys} for s in todo}
    print(f"[3/3] REAL FL over seeds {seeds} ...")
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        for scheme in todo:
            torch.manual_seed(sd)          # paired: same init/noise per seed
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            alg = CachingForwarding(cfg, mfl, mob, scheme, seed=sd)
            pm = mfl.poor_mask()
            acc_h, poor_h, tx_h, u_h = [], [], [], []
            for k in range(total):
                kk = k % mob.Krounds                    # replay the trace window
                mob.k = kk
                mfl.local_train()
                mfl.refresh_strengths()
                g = gammas[kk] if alg.flags["use_dis"] or alg.flags["cache_policy"] == "psi" \
                    else np.zeros(mob.N)
                selected = alg.run_round(k, g, gamma_eval=gammas[kk])
                accs = mfl.evaluate("test")
                acc_h.append(float(accs.mean()))
                poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
                tx_h.append(len(selected))
                u_h.append(alg.last_utility)
            stacks[scheme]["acc"].append(acc_h)
            stacks[scheme]["poor"].append(poor_h)
            stacks[scheme]["tx"].append(tx_h)
            stacks[scheme]["util"].append(u_h)
            print(f"  [seed {sd}] {scheme:16s} acc {acc_h[-1]:.3f} "
                  f"poor {poor_h[-1]:.3f} tx/round {np.mean(tx_h):.1f}", flush=True)
            del mfl, alg
            if device == "cuda":
                torch.cuda.empty_cache()

    results = {}
    for s in todo:
        results[s] = {}
        for m in keys:
            arr = np.stack(stacks[s][m])
            results[s][m] = arr.mean(0); results[s][m + "_std"] = arr.std(0)
    path = os.path.join(cfg.results_dir, f"metrics_v2x_real_{dataset}.npz")
    out = dict(np.load(path)) if (merge and os.path.exists(path)) else {}
    out.update({f"{s}__{k}": v for s, d in results.items() for k, v in d.items()})
    np.savez(path, **out)
    if not merge:
        _plot(results, cfg, dataset)
    print("=== REAL FL on Seoul V2X — final ===")
    for s in todo:
        print(f"  {disp(s):16s} acc {results[s]['acc'][-1]:.3f}  poor {results[s]['poor'][-1]:.3f}")
    return results


def _panel(ax, results, key, ylabel):
    K = len(results["Proposed"][key]); x = np.arange(1, K + 1)
    me = max(K // 11, 1)
    for s in REAL_SCHEMES:
        ax.plot(x, results[s][key], label=disp(s), markevery=me, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.2, **STY[s])
        ax.fill_between(x, results[s][key] - results[s][key + "_std"],
                        results[s][key] + results[s][key + "_std"],
                        color=STY[s]["color"], alpha=0.12, lw=0)
    ax.set_xlabel("Global round $k$"); ax.set_ylabel(ylabel)
    ax.set_xlim(0, K); ax.grid(True, ls="--", lw=0.6, alpha=0.5)


def _plot(results, cfg, dataset="kitti"):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _panel(axes[0], results, "acc", "Test accuracy")
    _panel(axes[1], results, "poor", "Poor-data accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.4, handlelength=2.6, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_infocom_v2x_real_{dataset}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_infocom_v2x_real")


if __name__ == "__main__":
    run()
