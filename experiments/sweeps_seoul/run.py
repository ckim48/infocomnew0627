"""
2x2 design-sensitivity study of FACE on the real Seoul-Gangnam V2X trace
(KITTI, single seed, 250 rounds):
  (a) nu sweep (incl. nu=0)     -- weight of the future-forwarding term;
                                   nu=0 removes the destination-free
                                   prediction from the objective.
  (b) gain estimator            -- UCB (paper) vs plain mean vs oracle
                                   (true current strengths); links Thm 2.
  (c) cache capacity sweep      -- 15/45/90 MB (15 MB cannot carry anything);
                                   FACE vs the caching-blind LRU baseline.
  (d) H^max sweep               -- prediction-horizon robustness.

The shared base point (nu=1, oracle, 45 MB, H=4) is run once and reused.
Outputs (npz, figure) stay in this folder.
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
from sim.algorithm import CachingForwarding
from sim.simulator import make_modality_availability
from sim.real_fl import RealMFL, _prep_data, _device

SEED = 2026
ROUNDS = 250
TAIL = 20

NU_VALS = [0.0, 0.5, 1.0, 2.0, 4.0]
EST_VALS = ["oracle", "mean", "ucb"]
CACHE_VALS = [15.0, 45.0, 90.0]
H_VALS = [1, 2, 4, 6]


def _base_cfg():
    cfg = Config()
    cfg.num_vehicles = 180
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    return cfg


def _prepare(device):
    cfg = _base_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    model, road_ei = train_hgat(cfg, road, mob, device=device,
                                warmup_rounds=40)
    data = _prep_data(cfg, cfg.seed, dataset="kitti")
    return cfg, road, mob, model, road_ei, data


def _gammas(cfg, road, mob, model, road_ei, device, H):
    cfg.H_max = H
    gam = []
    for k in range(mob.Krounds):
        mob.k = k
        gam.append(future_contact_scores(cfg, road, mob, model, road_ei,
                                         device=device))
    return np.array(gam)


def _run(cfg, mob, gammas, data, device, scheme="Proposed", nu=None,
         cache=None, est="oracle"):
    import copy
    c = copy.copy(cfg)
    if nu is not None:
        c.nu = nu
    if cache is not None:
        c.cache_capacity_mb = cache
    avail = make_modality_availability(c, np.random.default_rng(SEED + 7))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mfl = RealMFL(c, rng, avail, data, device=device)
    alg = CachingForwarding(c, mfl, mob, scheme, seed=SEED)
    alg.gain_est = est
    pm = mfl.poor_mask()
    acc_h, poor_h = [], []
    for k in range(ROUNDS):
        kk = k % mob.Krounds
        mob.k = kk
        mfl.local_train()
        mfl.refresh_strengths()
        g = gammas[kk] if alg.flags["use_dis"] or alg.flags["cache_policy"] == "psi" \
            else np.zeros(mob.N)
        alg.run_round(k, g)
        accs = mfl.evaluate("test")
        acc_h.append(float(accs.mean()))
        poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
    del mfl, alg
    if device == "cuda":
        torch.cuda.empty_cache()
    a = np.array(acc_h)
    return float(a[-TAIL:].mean()), float(np.array(poor_h)[-TAIL:].mean())


def main():
    device = _device()
    cfg, road, mob, model, road_ei, data = _prepare(device)
    g4 = _gammas(cfg, road, mob, model, road_ei, device, 4)

    out = {}
    print("[base] nu=1 oracle 45MB H=4 ...", flush=True)
    base = _run(cfg, mob, g4, data, device)
    print(f"  base acc {base[0]:.3f} poor {base[1]:.3f}", flush=True)

    # (a) nu sweep
    for nu in NU_VALS:
        out[f"nu_{nu}"] = base if nu == 1.0 else \
            _run(cfg, mob, g4, data, device, nu=nu)
        print(f"  [nu {nu}] acc {out[f'nu_{nu}'][0]:.3f}", flush=True)

    # (b) estimator
    for est in EST_VALS:
        out[f"est_{est}"] = base if est == "oracle" else \
            _run(cfg, mob, g4, data, device, est=est)
        print(f"  [est {est}] acc {out[f'est_{est}'][0]:.3f}", flush=True)

    # (c) cache sweep, FACE and the LRU baseline
    for cap in CACHE_VALS:
        out[f"cacheF_{int(cap)}"] = base if cap == 45.0 else \
            _run(cfg, mob, g4, data, device, cache=cap)
        out[f"cacheC_{int(cap)}"] = _run(cfg, mob, g4, data, device,
                                         scheme="Caching-assisted", cache=cap)
        print(f"  [cache {cap}] FACE {out[f'cacheF_{int(cap)}'][0]:.3f} "
              f"Caching {out[f'cacheC_{int(cap)}'][0]:.3f}", flush=True)

    # (d) H^max sweep
    for H in H_VALS:
        if H == 4:
            out[f"H_{H}"] = base
        else:
            gH = _gammas(cfg, road, mob, model, road_ei, device, H)
            out[f"H_{H}"] = _run(cfg, mob, gH, data, device)
        print(f"  [H {H}] acc {out[f'H_{H}'][0]:.3f}", flush=True)

    np.savez(os.path.join(HERE, "metrics_sweeps_seoul.npz"),
             **{k: np.array(v) for k, v in out.items()})
    figure(out)
    print("=== sweeps done ===", flush=True)


def figure(out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    if out is None:
        d = np.load(os.path.join(HERE, "metrics_sweeps_seoul.npz"))
        out = {k: tuple(d[k]) for k in d.files}
    RED, GRN = "#e8000b", "#1f9e3d"
    fig, axg = plt.subplots(2, 2, figsize=(7.2, 6.6))

    ax = axg[0, 0]                                        # (a) nu sweep
    mu = [out[f"nu_{v}"][0] for v in NU_VALS]
    ax.plot(NU_VALS, mu, marker="o", color=RED, markerfacecolor="white",
            markeredgewidth=1.2)
    ax.set_xlabel(r"Future-forwarding weight $\nu$")
    ax.set_ylabel("Final test accuracy")

    ax = axg[0, 1]                                        # (b) estimator
    names = ["w/o UCB (mean)", "UCB", "Oracle"]
    keys = ["est_mean", "est_ucb", "est_oracle"]
    mu = [out[k][0] for k in keys]
    bars = ax.bar(range(3), mu, width=0.55,
                  color=["#9db8d9", RED, "#4f7ab8"], edgecolor="k",
                  linewidth=0.6)
    for b, v in zip(bars, mu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(min(mu) - 0.03, max(mu) + 0.02)

    ax = axg[1, 0]                                        # (c) cache sweep
    for key, lbl, col in [("cacheF", "FACE", RED), ("cacheC", "Caching", GRN)]:
        mu = [out[f"{key}_{int(c)}"][0] for c in CACHE_VALS]
        ax.plot(CACHE_VALS, mu, marker="o" if key == "cacheF" else "s",
                color=col, ls="-" if key == "cacheF" else "--",
                markerfacecolor="white", markeredgewidth=1.2, label=lbl)
    ax.set_xlabel("Cache capacity (MB)")
    ax.set_ylabel("Final test accuracy")
    ax.legend(fontsize=10)

    ax = axg[1, 1]                                        # (d) H sweep
    mu = [out[f"H_{H}"][0] for H in H_VALS]
    ax.plot(H_VALS, mu, marker="o", color=RED, markerfacecolor="white",
            markeredgewidth=1.2)
    ax.set_xticks(H_VALS)
    ax.set_xlabel(r"Prediction horizon $H^{\max}$")
    ax.set_ylabel("Final test accuracy")

    for i, ax in enumerate(axg.ravel()):
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(1)
        ax.set_title(f"({'abcd'[i]})", y=-0.30, fontsize=12)
    fig.tight_layout(h_pad=2.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_sweeps_seoul.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_sweeps_seoul")


if __name__ == "__main__":
    main()
