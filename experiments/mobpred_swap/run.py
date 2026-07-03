"""
Mobility-predictor swap experiment: how does FACE's end-task performance
change when the destination-free mobility predictor behind Gamma (the future
contact-demand potential, Eq. 10 / paper Sec. III-D) is replaced?

Predictor variants (only the transition model q_phi changes; the occupancy
propagation, Gamma formula, and the FACE algorithm are identical):
  * HGAT (ours)  -- hierarchical road/vehicle graph-attention predictor
  * Markov       -- empirical first-order transition frequencies counted from
                    the observed trace (no learning, traffic-history only)
  * Topology     -- uniform over road successors (road topology only)
  * NoPred       -- Gamma = 0 (future term disabled; psi-cache keeps only the
                    immediate learning term)

Everything else matches the paper's main comparison (sim/real_fl.py):
real multimodal FL on KITTI over InTAS, N=80, K=150, 3 seeds.
Outputs (npz, figure, LaTeX table) stay inside this folder.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sim.config import Config
from sim.algorithm import CachingForwarding, SCHEME_FLAGS
from sim.simulator import make_modality_availability
from sim.mobility import RoadNetwork, MobilitySim
from sim.intas_trace import get_or_build_trace
from sim.hgat import train_hgat, future_contact_scores
from sim.real_fl import RealMFL, _prep_data, _device, main_config

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["HGAT", "Markov", "Topology", "NoPred"]


# ---------------------------------------------------------------------------
# Gamma computation with a pluggable transition model
# ---------------------------------------------------------------------------
def _gamma_from_trans(cfg, road, mob, trans):
    """Same discounted expected-co-location propagation as
    sim.hgat.future_contact_scores, with transition model `trans(i, e)`."""
    N = mob.N
    dists = [{int(mob.seg[i]): 1.0} for i in range(N)]
    gamma = np.zeros(N)
    for h in range(1, cfg.H_max + 1):
        new_dists = []
        for i in range(N):
            nd = {}
            for e, p_e in dists[i].items():
                if p_e <= 1e-6:
                    continue
                succ, pi = trans(i, e)
                for idx, e2 in enumerate(succ):
                    nd[e2] = nd.get(e2, 0.0) + p_e * float(pi[idx])
            new_dists.append(nd)
        dists = new_dists
        occ = {}
        for i in range(N):
            for e, p in dists[i].items():
                occ[e] = occ.get(e, 0.0) + p
        disc = cfg.gamma_disc ** h
        for i in range(N):
            s = 0.0
            for e, p in dists[i].items():
                s += p * (occ.get(e, 0.0) - p)
            gamma[i] += disc * s
    if gamma.max() > 0:
        gamma = gamma / (gamma.mean() + 1e-9)
    return gamma


def _markov_counts(road, mob):
    """Empirical first-order segment-transition counts over the whole trace."""
    counts = {}
    segs = mob.veh_seg                            # (K, N) segment ids per round
    K = segs.shape[0]
    for k in range(K - 1):
        for i in range(mob.N):
            e, e2 = int(segs[k, i]), int(segs[k + 1, i])
            if e == e2:
                continue
            if e2 in road.successors[e]:
                counts.setdefault(e, {}).setdefault(e2, 0)
                counts[e][e2] += 1
    return counts


def build_gammas(cfg, road, mob, variant, device):
    K = mob.Krounds
    if variant == "NoPred":
        return np.zeros((K, mob.N))
    if variant == "HGAT":
        model, road_ei = train_hgat(cfg, road, mob, device=device,
                                    warmup_rounds=40)
        gam = []
        for k in range(K):
            mob.k = k
            gam.append(future_contact_scores(cfg, road, mob, model, road_ei,
                                             device=device))
        return np.array(gam)

    if variant == "Markov":
        counts = _markov_counts(road, mob)

        def trans(i, e):
            succ = road.successors[e]
            if not succ:
                return succ, np.array([])
            c = np.array([counts.get(e, {}).get(e2, 0) + 1.0 for e2 in succ])
            return succ, c / c.sum()
    else:                                          # Topology: uniform successor
        def trans(i, e):
            succ = road.successors[e]
            if not succ:
                return succ, np.array([])
            return succ, np.full(len(succ), 1.0 / len(succ))

    gam = []
    for k in range(K):
        mob.k = k
        gam.append(_gamma_from_trans(cfg, road, mob, trans))
    return np.array(gam)


# ---------------------------------------------------------------------------
def main(seeds=(2026, 2027, 2028), dataset="kitti"):
    cfg = main_config()
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = _device()

    cache_path = os.path.join(cfg.results_dir,
                              f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache_path, begin=34050.0, dt=2.0,
                               warmup_s=480.0)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    data = _prep_data(cfg, cfg.seed, dataset=dataset)

    gammas = {}
    for v in VARIANTS:
        print(f"[gamma] building {v} ...", flush=True)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        gammas[v] = build_gammas(cfg, road, mob, v, device)

    metric_keys = ["acc", "poor", "tx"]
    stacks = {v: {m: [] for m in metric_keys} for v in VARIANTS}
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        for v in VARIANTS:
            torch.manual_seed(sd)          # paired: same init/noise per seed
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            alg = CachingForwarding(cfg, mfl, mob, "Proposed", seed=sd)
            pm = mfl.poor_mask()
            acc_h, poor_h, tx_h = [], [], []
            for k in range(mob.Krounds):
                mob.k = k
                mfl.local_train()
                mfl.refresh_strengths()
                selected = alg.run_round(k, gammas[v][k])
                accs = mfl.evaluate("test")
                acc_h.append(float(accs.mean()))
                poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
                tx_h.append(len(selected))
            for m, h in zip(metric_keys, [acc_h, poor_h, tx_h]):
                stacks[v][m].append(h)
            print(f"  [mobpred seed {sd}] {v:9s} acc {acc_h[-1]:.3f} "
                  f"poor {poor_h[-1]:.3f}", flush=True)
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
    np.savez(os.path.join(HERE, f"metrics_mobpred_{dataset}.npz"),
             **{f"{v}__{k}": val for v, d in results.items() for k, val in d.items()})

    _figure(results, dataset)
    _table(results, dataset)
    print("=== mobility-predictor swap final ===")
    for v in VARIANTS:
        print(f"  {v:9s} acc {results[v]['acc'][-1]:.3f} "
              f"poor {results[v]['poor'][-1]:.3f}")
    return results


def _figure(results, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    STY = {
        "HGAT":     dict(color="#e8000b", ls="-",  marker="o"),
        "Markov":   dict(color="#1f5fd0", ls="--", marker="s"),
        "Topology": dict(color="#1f9e3d", ls="-.", marker="D"),
        "NoPred":   dict(color="#000000", ls=":",  marker="^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6))
    for ax, key, ylab in [(axes[0], "acc", "Test accuracy"),
                          (axes[1], "poor", "Poor-data accuracy")]:
        K = len(results["HGAT"][key]); x = np.arange(1, K + 1)
        me = max(K // 11, 1)
        for v in VARIANTS:
            ax.plot(x, results[v][key], label=v, markevery=me, markersize=5.5,
                    markerfacecolor="white", markeredgewidth=1.2, **STY[v])
            sd = results[v][key + "_std"]
            ax.fill_between(x, results[v][key] - sd, results[v][key] + sd,
                            color=STY[v]["color"], alpha=0.10, lw=0)
        ax.set_xlabel("Global round $k$"); ax.set_ylabel(ylab)
        ax.set_xlim(0, K); ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.4, handlelength=2.6, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_mobpred_swap_{dataset}.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("  saved fig_mobpred_swap")


def _table(results, dataset, tail=20):
    st = {}
    for v in VARIANTS:
        st[v] = dict(acc=results[v]["acc"][-tail:].mean(),
                     acc_sd=results[v]["acc_std"][-tail:].mean(),
                     poor=results[v]["poor"][-tail:].mean(),
                     poor_sd=results[v]["poor_std"][-tail:].mean())
    best_acc = max(st[v]["acc"] for v in VARIANTS)
    best_poor = max(st[v]["poor"] for v in VARIANTS)
    multi = st["HGAT"]["acc_sd"] > 0

    def pm(v, k, best):
        cell = (f"{100*st[v][k]:.1f} $\\pm$ {100*st[v][k + '_sd']:.1f}"
                if multi else f"{100*st[v][k]:.1f}")
        return f"\\textbf{{{cell}}}" if st[v][k] == best else cell

    disp = {"HGAT": "HGAT (ours)", "Markov": "Markov", "Topology": "Topology",
            "NoPred": "w/o prediction"}
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Impact of the mobility predictor behind the"
        " destination-free forwarding potential $\\Gamma$ (FACE, real"
        " multimodal FL on KITTI over "
        + ("the sparse Seoul-Gangnam V2X trace" if dataset == "seoul"
           else "InTAS")
        + ("; single 250-round run" if not multi
           else "; mean $\\pm$ std over 3 seeds")
        + f", \\%, averaged over the final {tail} rounds).}}",
        "    \\label{tab:mobpred}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c}",
        "        \\hline",
        "        \\textsc{Predictor} & \\textsc{Acc} & \\textsc{Poor Acc} \\\\",
        "        \\hline",
    ]
    for v in ["Markov", "Topology", "NoPred"]:
        lines.append(f"        \\textsc{{{disp[v]}}} & {pm(v, 'acc', best_acc)}"
                     f" & {pm(v, 'poor', best_poor)} \\\\")
    lines += ["        \\hline",
              f"        \\textsc{{{disp['HGAT']}}} & {pm('HGAT', 'acc', best_acc)}"
              f" & {pm('HGAT', 'poor', best_poor)} \\\\",
              "        \\hline", "    \\end{tabular}", "\\end{table}"]
    tex = "\n".join(lines)
    with open(os.path.join(HERE, f"tab_mobpred_{dataset}.tex"), "w") as f:
        f.write(tex + "\n")
    print(tex)


def main_seoul(seeds=(2026,), rounds=250, dataset="kitti", num_vehicles=180):
    """Predictor swap over the sparse Seoul-Gangnam V2X trace (cyclic replay),
    where the future contact-demand term actually matters."""
    from sim.v2x_trace import build_v2x_trace
    cfg = Config()
    cfg.num_vehicles = num_vehicles
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = _device()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    data = _prep_data(cfg, cfg.seed, dataset=dataset)

    gammas = {}
    for v in VARIANTS:
        print(f"[gamma] building {v} (Seoul) ...", flush=True)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        gammas[v] = build_gammas(cfg, road, mob, v, device)

    metric_keys = ["acc", "poor", "tx"]
    stacks = {v: {m: [] for m in metric_keys} for v in VARIANTS}
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        for v in VARIANTS:
            torch.manual_seed(sd)          # paired: same init/noise per seed
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            alg = CachingForwarding(cfg, mfl, mob, "Proposed", seed=sd)
            pm = mfl.poor_mask()
            acc_h, poor_h, tx_h = [], [], []
            for k in range(rounds):
                kk = k % mob.Krounds
                mob.k = kk
                mfl.local_train()
                mfl.refresh_strengths()
                selected = alg.run_round(k, gammas[v][kk])
                accs = mfl.evaluate("test")
                acc_h.append(float(accs.mean()))
                poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
                tx_h.append(len(selected))
            for m, h in zip(metric_keys, [acc_h, poor_h, tx_h]):
                stacks[v][m].append(h)
            print(f"  [mobpred-seoul seed {sd}] {v:9s} acc {acc_h[-1]:.3f} "
                  f"poor {poor_h[-1]:.3f}", flush=True)
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
    np.savez(os.path.join(HERE, "metrics_mobpred_seoul.npz"),
             **{f"{v}__{k}": val for v, d in results.items() for k, val in d.items()})
    _figure(results, "seoul")
    _table(results, "seoul")
    print("=== mobility-predictor swap (Seoul) final ===")
    for v in VARIANTS:
        print(f"  {v:9s} acc {results[v]['acc'][-1]:.3f} "
              f"poor {results[v]['poor'][-1]:.3f}")
    return results


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "seoul":
        main_seoul()
    else:
        main()
