"""
Run the MDFL caching/forwarding pipeline on the REAL Seoul V2X mobility trace
(instead of the Ingolstadt InTAS trace). Same GAT predictor and same schemes;
only the mobility substrate changes -> a real-data mobility experiment.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, SCHEMES
from .mobility import RoadNetwork, MobilitySim
from .hgat import train_hgat, future_contact_scores
from .simulator import run_schemes
from .v2x_trace import build_v2x_trace
from .plotting import STYLE as STY, disp


def run(cfg=None, device=None, num_vehicles=180, seeds=None):
    cfg = cfg or Config()
    cfg.num_vehicles = num_vehicles
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [cfg.seed]
    os.makedirs(cfg.results_dir, exist_ok=True)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print("[1/4] Building real Seoul V2X mobility trace ...")
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    print(f"      road |V|={road.V}, vehicles N={mob.N}, rounds K={mob.Krounds}")

    print("[2/4] Training GAT mobility predictor on Seoul roads ...")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=30)
    print("[3/4] Predicting future-contact Gamma ...")
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei, device=device))
    gammas = np.array(gammas)

    print(f"[4/4] Running schemes over seeds {seeds} ...")
    keys = ["loss", "acc", "tail", "tx", "qlen"]
    stacks = {s: {m: [] for m in keys} for s in SCHEMES}
    for sd in seeds:
        res = run_schemes(cfg, mob, gammas, sd)
        for s in SCHEMES:
            for m in keys:
                stacks[s][m].append(res[s][m])

    results = {}
    for s in SCHEMES:
        results[s] = {}
        for m in keys:
            arr = np.stack(stacks[s][m])
            results[s][m] = arr.mean(0); results[s][m + "_std"] = arr.std(0)
        print(f"      {s:16s} final acc {results[s]['acc'][-1]:.3f} "
              f"tail {results[s]['tail'][-1]:.3f}")

    out = os.path.join(cfg.results_dir, "metrics_v2x.npz")
    np.savez(out, **{f"{s}__{k}": v for s, d in results.items() for k, v in d.items()})
    print("  saved", out)
    _plot(results, cfg)
    return results


def _panel(ax, results, key, ylabel):
    K = len(results["Proposed"][key]); x = np.arange(1, K + 1)
    me = max(K // 11, 1)
    for s in SCHEMES:
        ax.plot(x, results[s][key], label=disp(s), markevery=me, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.2, **STY[s])
    ax.set_xlabel("Global round $k$"); ax.set_ylabel(ylabel)
    ax.set_xlim(0, K); ax.grid(True, ls="--", lw=0.6, alpha=0.5)


def _plot(results, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _panel(axes[0], results, "acc", "Test accuracy")
    _panel(axes[1], results, "tail", "Poor-data accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.4, handlelength=2.6, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_infocom_v2x_seoul.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_infocom_v2x_seoul")


if __name__ == "__main__":
    run()
