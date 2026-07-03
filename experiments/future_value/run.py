"""
Direct measurement of the future-forwarding element: does the destination-free
potential Gamma steer caching toward encoders that later reach vehicles the
owner NEVER directly encounters?  (The paper's motivating question -- encoder
dissemination beyond direct V2V encounters.)

Variants on the Seoul-Gangnam trace (KITTI, paired seeds):
  FACE (full)      -- psi-cache with the Gamma future term
  w/o prediction   -- identical but Gamma = 0 (cache by immediate term only)
  Caching (LRU)    -- recency caching baseline

Metrics per round: store-carry-forward relays (sender != owner) and
beyond-encounter deliveries (owner never met receiver), plus their USEFUL
variants (delivered encoder stronger than what the receiver held).
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

from sim.config import Config
from sim.mobility import RoadNetwork, MobilitySim
from sim.hgat import train_hgat, future_contact_scores
from sim.v2x_trace import build_v2x_trace
from sim.algorithm import CachingForwarding, SCHEME_FLAGS
from sim.simulator import make_modality_availability
from sim.real_fl import RealMFL, _prep_data, _device

ROUNDS = 250
SEEDS = (2026, 2027, 2028)
FULL = SCHEME_FLAGS["Proposed"]
VARIANTS = {
    "FACE": ("Proposed", FULL, True),
    "w/o prediction": ("Proposed", {**FULL, "no_gamma": True}, False),
    "Caching": ("Caching-assisted", SCHEME_FLAGS["Caching-assisted"], False),
}


def main(dataset="kitti"):
    device = _device()
    cfg = Config()
    cfg.num_vehicles = 180
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei,
                                            device=device))
    gammas = np.array(gammas)
    data = _prep_data(cfg, cfg.seed, dataset=dataset)

    keys = ["relay", "relay_u", "beyond", "beyond_u", "acc"]
    stacks = {v: {m: [] for m in keys} for v in VARIANTS}
    for sd in SEEDS:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        for name, (scheme, flags, use_gamma) in VARIANTS.items():
            torch.manual_seed(sd)
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            alg = CachingForwarding(cfg, mfl, mob, scheme, seed=sd)
            alg.flags = flags
            hh = {m: [] for m in keys}
            for k in range(ROUNDS):
                kk = k % mob.Krounds
                mob.k = kk
                mfl.local_train()
                mfl.refresh_strengths()
                g = gammas[kk] if (use_gamma and flags["cache_policy"] == "psi") \
                    else np.zeros(mob.N)
                alg.run_round(k, g)
                hh["relay"].append(alg._n_relay)
                hh["relay_u"].append(alg._n_relay_u)
                hh["beyond"].append(alg._n_beyond)
                hh["beyond_u"].append(alg._n_beyond_u)
                hh["acc"].append(float(mfl.evaluate("test").mean()))
            for m in keys:
                stacks[name][m].append(hh[m])
            print(f"  [fv seed {sd}] {name:16s} "
                  f"relay_u {np.sum(hh['relay_u'])} "
                  f"beyond_u {np.sum(hh['beyond_u'])} "
                  f"acc {hh['acc'][-1]:.3f}", flush=True)
            del mfl, alg
            if device == "cuda":
                torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        for m in keys:
            arr = np.stack(stacks[v][m]).astype(float)
            out[f"{v}__{m}"] = arr.mean(0)
            out[f"{v}__{m}_std"] = arr.std(0)
            out[f"{v}__{m}_all"] = arr
    np.savez(os.path.join(HERE, f"metrics_future_value_{dataset}.npz"), **out)
    figure(dataset)
    print("=== future-value probe done ===", flush=True)


def figure(dataset="kitti"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    d = np.load(os.path.join(HERE, f"metrics_future_value_{dataset}.npz"))
    STY = {"FACE": dict(color="#e8000b", ls="-", marker="o"),
           "w/o prediction": dict(color="#000000", ls=":", marker="^"),
           "Caching": dict(color="#1f9e3d", ls="--", marker="s")}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6))
    for ax, key, ylab in [
            (axes[0], "relay_u", "Cumulative useful relays"),
            (axes[1], "beyond_u",
             "Cumulative useful deliveries\nbeyond direct encounters")]:
        for v in VARIANTS:
            y = np.cumsum(d[f"{v}__{key}"])
            sd = np.sqrt(np.cumsum(d[f"{v}__{key}_std"] ** 2))
            K = len(y); x = np.arange(1, K + 1)
            ax.plot(x, y, label=v, markevery=max(K // 9, 1), markersize=5,
                    markerfacecolor="white", markeredgewidth=1.1, **STY[v])
            ax.fill_between(x, y - sd, y + sd, color=STY[v]["color"],
                            alpha=0.10, lw=0)
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, K)
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(1)
    axes[0].set_title("(a)", y=-0.34, fontsize=12)
    axes[1].set_title("(b)", y=-0.34, fontsize=12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1),
               columnspacing=1.6, handlelength=2.4, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_future_value_{dataset}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_future_value_{dataset}")


if __name__ == "__main__":
    main()
