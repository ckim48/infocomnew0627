"""
Additional INFOCOM-style evaluation figures from the large-scale InTAS
simulation (fast, scalable): component ablation, sensitivity sweeps, and a
per-vehicle fairness CDF. Mobility + GAT are prepared once and reused across all
configurations that do not change the trace.
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, SCHEMES
from .simulator import prepare, make_modality_availability
from .algorithm import CachingForwarding, SCHEME_FLAGS
from .mfl import MultimodalFL

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})
RED, BLU, GRN, BLK, ORG = "#e8000b", "#1f5fd0", "#1f9e3d", "#000000", "#f0a020"
STY = {
    "Proposed":         dict(color=RED, ls="-",  marker="o"),
    "Caching-assisted": dict(color=GRN, ls="--", marker="s"),
    "V2V-aware":        dict(color=BLU, ls="-.", marker="D"),
    "Learning-aware":   dict(color=BLK, ls=":",  marker="^"),
}


def _run(cfg, mob, gammas, flags, seed, avail):
    rng = np.random.default_rng(seed)
    mfl = MultimodalFL(cfg, rng, avail)
    alg = CachingForwarding(cfg, mfl, mob, "Proposed", seed=seed)
    alg.flags = flags
    for k in range(mob.Krounds):
        mob.k = k
        mfl.local_train()
        g = gammas[k] if (flags["use_dis"] or flags["cache_policy"] == "psi") \
            else np.zeros(mob.N)
        alg.run_round(k, g)
    pv = np.array([np.mean([mfl.q_eff(i, r) for r in mfl.avail[i]]) for i in range(mob.N)])
    return mfl.mean_accuracy(), pv


def _avail(cfg, seed):
    return make_modality_availability(cfg, np.random.default_rng(seed + 7))


def main(device=None, seeds=(2026, 2027)):
    cfg = Config(); cfg.num_vehicles = 150; cfg.K = 150; cfg.comm_range = 150.0
    device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    os.makedirs(cfg.figures_dir, exist_ok=True)
    road, mob, gammas = prepare(cfg, device)
    FULL = SCHEME_FLAGS["Proposed"]

    # ===== 1) Ablation (algorithm components) =====
    variants = {
        "FACE":            FULL,
        "w/o caching":     {**FULL, "carry": False, "cache_policy": "own"},
        "w/o demand":      {**FULL, "demand_aware": False, "cache_policy": "lru"},
        "w/o queue":       {**FULL, "use_queue": False},
    }
    abl = {}
    for name, fl in variants.items():
        vals = [_run(cfg, mob, gammas, fl, s, _avail(cfg, s))[0] for s in seeds]
        abl[name] = (np.mean(vals), np.std(vals))
    print("[ablation]", {k: round(v[0], 3) for k, v in abl.items()})

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    names = list(variants.keys()); mu = [abl[n][0] for n in names]; sd = [abl[n][1] for n in names]
    cols = [RED, ORG, GRN, BLU, BLK]
    ax.bar(range(len(names)), mu, yerr=sd, color=cols, edgecolor="k", linewidth=0.6,
           capsize=3, width=0.65)
    for i, v in enumerate(mu):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, fontsize=10)
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(min(mu) * 0.95, max(mu) * 1.03)
    ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_ablation.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # ===== 2) Sensitivity sweeps (Proposed vs baselines) =====
    schemes3 = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]

    def sweep(setter, values, label):
        out = {s: [] for s in schemes3}
        for val in values:
            for s in schemes3:
                vals = []
                for sd in seeds:
                    c = copy.copy(cfg); setter(c, val)
                    vals.append(_run(c, mob, gammas, SCHEME_FLAGS[s], sd, _avail(c, sd))[0])
                out[s].append(np.mean(vals))
        return out

    def set_cache(c, v): c.cache_capacity_mb = v
    def set_fracgood(c, v): c.frac_good = v
    def set_contact(c, v): c.contact_time_per_round = v
    def set_mods(c, v):
        c.modalities = ["camera", "lidar", "radar", "gps"][:v]
        c.modality_prob = {m: 1.0 for m in c.modalities}

    cache_vals = [15, 25, 35, 45, 60]
    frac_vals = [0.10, 0.15, 0.20, 0.30, 0.40]
    contact_vals = [0.8, 1.2, 1.6, 2.4, 3.2]
    mod_vals = [1, 2, 3, 4]
    sw_cache = sweep(set_cache, cache_vals, "cache")
    sw_frac = sweep(set_fracgood, frac_vals, "frac")
    sw_cont = sweep(set_contact, contact_vals, "contact")
    sw_mods = sweep(set_mods, mod_vals, "mods")

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))
    def panel(a, xs, sw, xlabel, tag):
        for s in schemes3:
            a.plot(xs, sw[s], label=s, markersize=5.5, markerfacecolor="white",
                   markeredgewidth=1.2, **STY[s])
        a.set_xlabel(xlabel, labelpad=6); a.set_ylabel("Final test accuracy")
        a.grid(True, ls="--", lw=0.6, alpha=0.5); a.set_title(tag, y=-0.40, fontsize=12)
    panel(ax[0, 0], cache_vals, sw_cache, "Cache capacity (MB)", "(a)")
    panel(ax[0, 1], [f * 100 for f in frac_vals], sw_frac, "Strong-encoder owners (\\%)", "(b)")
    panel(ax[1, 0], mod_vals, sw_mods, "Number of modalities", "(c)")
    panel(ax[1, 1], contact_vals, sw_cont, "Contact-time budget (s)", "(d)")
    ax[1, 0].set_xticks(mod_vals)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06),
               columnspacing=1.3, fontsize=10)
    fig.tight_layout(); fig.subplots_adjust(hspace=0.5, wspace=0.34, top=0.93)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_sensitivity.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # ===== 3) Fairness: per-vehicle final accuracy CDF =====
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for s in schemes3:
        pvs = []
        for sd in seeds:
            pvs.append(_run(cfg, mob, gammas, SCHEME_FLAGS[s], sd, _avail(cfg, sd))[1])
        pv = np.sort(np.concatenate(pvs))
        cdf = np.arange(1, len(pv) + 1) / len(pv)
        ax.plot(pv, cdf, label=s, **{k: v for k, v in STY[s].items() if k != "marker"})
    ax.set_xlabel("Per-vehicle model accuracy"); ax.set_ylabel("CDF")
    ax.grid(True, ls="--", lw=0.6, alpha=0.5); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_fairness.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print("saved fig_ablation, fig_sensitivity, fig_fairness")


if __name__ == "__main__":
    main()
