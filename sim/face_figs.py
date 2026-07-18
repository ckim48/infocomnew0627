"""
Additional paper figures for the revised FACE model, written to new_result/:

  fig_face_abl_bars   component-ablation bar chart (acc / poor acc, both
                      ECV-placement scenarios, error bars over seeds)
  fig_face_beyond     beyond-direct-encounter delivery ratio per scheme
  fig_face_pareto     test accuracy vs. cumulative encoder traffic
  fig_face_txrate     encoder transmissions per round (traffic restraint)

All read the npz/logs produced by the overnight rerun, so re-running this
module after any experiment refresh regenerates every figure.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import STYLE, disp
from .real_fl import REAL_SCHEMES
from .face_abl_table import _parse, LABELS, ORDER

OUT = "new_result"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.join(OUT, name))


def fig_abl_bars():
    uni = _parse("results/face_abl_uniform.log")
    part = _parse("results/face_abl_part.log")
    names = [n for n in ORDER if n in uni or n in part]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, res, title in ((axes[0], uni, "Uniform ECVs"),
                           (axes[1], part, "Partitioned ECVs")):
        xs = np.arange(len(names))
        for off, idx, lab, col in ((-0.2, 0, "Acc", "#4C72B0"),
                                   (0.2, 1, "Poor acc", "#DD8452")):
            mu = [100 * np.mean([v[idx] for v in res[n].values()])
                  if n in res else 0.0 for n in names]
            sd = [100 * np.std([v[idx] for v in res[n].values()])
                  if n in res else 0.0 for n in names]
            ax.bar(xs + off, mu, width=0.38, yerr=sd, capsize=2.5,
                   label=lab, color=col, edgecolor="black", lw=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[n][0].replace(r"\textbf{", "")
                            .replace("}", "").replace("w/o ", "w/o\n")
                            for n in names], fontsize=7, rotation=28,
                           ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
    axes[0].set_ylabel("Final accuracy (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_face_abl_bars")


def _load(dataset):
    p = f"results/metrics_v2x_real_{dataset}.npz"
    return np.load(p) if os.path.exists(p) else None


def fig_curves(key, ylabel, fname, cumulative_x=None):
    ds = [(d, _load(d)) for d in ("kitti", "nuscenes")]
    ds = [(d, z) for d, z in ds if z is not None]
    if not ds:
        return
    fig, axes = plt.subplots(1, len(ds), figsize=(4.4 * len(ds), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (d, z) in zip(axes, ds):
        for s in REAL_SCHEMES:
            if f"{s}__{key}" not in z.files:
                continue
            y = z[f"{s}__{key}"]
            if cumulative_x:
                x = np.cumsum(z[f"{s}__{cumulative_x}"]) / 1e3
                ax.set_xlabel("Cumulative encoder traffic (GB)")
            else:
                x = np.arange(1, len(y) + 1)
                ax.set_xlabel("Global round $k$")
            st = STYLE.get(s, {})
            ax.plot(x, y, label=disp(s), markevery=max(len(y) // 10, 1),
                    ms=4.5, markerfacecolor="white", **st)
        ax.set_title(d.upper() if d == "kitti" else "nuScenes", fontsize=10)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, fname)


ABL_NPZ = "results/metrics_face_ablation_v2x.npz"
EVENTS_NPZ = "results/metrics_v2x_real_kitti_events.npz"
ABL_ORDER = [
    # x labels use the manuscript's own terms/notation (Secs. III-D, IV)
    ("FACE (full)", "FACE\n(full)"),
    ("w/o relay ferrying", "w/o caching\n($c_{i,x}{\\equiv}0$)"),
    ("w/o demand", "w/o demand\n($\\widehat{v}_{i,x}{\\equiv}$ const)"),
    ("w/o future value",
     "w/o future-contact\nutility ($F_{a,x}{\\equiv}0$)"),
]


def fig_abl_2panel():
    """Component ablation, simplified to the three core mechanisms:
    (a) final accuracy (all vs high-demand vehicles), (b) communication
    volume split into useful and redundant deliveries."""
    d = np.load(ABL_NPZ)
    keys = [k for k, _ in ABL_ORDER if f"{k}__acc_all" in d.files]
    labs = [l for k, l in ABL_ORDER if f"{k}__acc_all" in d.files]
    xs = np.arange(len(keys))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))
    ax = axes[0]                                       # (a) accuracy
    for off, met, lab, col in (
            (-0.19, "acc", "All vehicles", "#4C72B0"),
            (0.19, "poor", "High-demand vehicles", "#DD8452")):
        a = np.stack([100 * d[f"{k}__{met}_all"][:, -1] for k in keys])
        ax.bar(xs + off, a.mean(1), width=0.36, yerr=a.std(1), capsize=2.5,
               label=lab, color=col, edgecolor="black", lw=0.4)
    ax.set_ylabel("Final accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[1]                                       # (b) communication
    gb, red_gb = [], []
    for k in keys:
        mb = d[f"{k}__txmb_all"]                       # seeds x rounds
        rr = d[f"{k}__redund_all"]
        gb.append(mb.sum(1) / 1024)
        red_gb.append((mb * rr).sum(1) / 1024)
    gb, red_gb = np.array(gb), np.array(red_gb)
    use_gb = gb - red_gb
    ax.bar(xs, use_gb.mean(1), width=0.5, label="Useful deliveries",
           color="#55A868", edgecolor="black", lw=0.4)
    ax.bar(xs, red_gb.mean(1), width=0.5, bottom=use_gb.mean(1),
           yerr=gb.std(1), capsize=2.5, label="Redundant deliveries",
           color="#C44E52", edgecolor="black", lw=0.4)
    ax.set_ylabel("Communication volume (GB)")
    ax.set_ylim(0, 62)
    ax.legend(fontsize=7.5, loc="upper right")
    for i, ax_ in enumerate(axes):
        ax_.set_xticks(xs)
        ax_.set_xticklabels(labs, fontsize=7.2)
        ax_.grid(True, axis="y", ls=":", alpha=0.5)
        ax_.text(0.5, -0.30, f"({'ab'[i]})", transform=ax_.transAxes,
                 ha="center", va="top", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_face_abl_2panel")


def fig_deadline(deadlines=(1, 2, 3, 5, 10, 20)):
    """Delivery-deadline success: probability that a high-demand vehicle
    receives a useful (compatible, stronger) encoder within d rounds,
    averaged over all window starts, vehicles, and seeds."""
    if not os.path.exists(EVENTS_NPZ):
        print("  [skip] fig_face_deadline: no events npz yet")
        return
    z = np.load(EVENTS_NPZ)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for s in REAL_SCHEMES:
        if f"{s}__udeliv_all" not in z.files:
            continue
        U_all = z[f"{s}__udeliv_all"]      # seeds x K x N
        pm_all = z[f"{s}__pmask_all"]      # seeds x N
        ys = []
        for U, pm in zip(U_all, pm_all):
            Up = U[:, pm]                  # K x n_hd
            K = Up.shape[0]
            row = []
            for dl in deadlines:
                w = np.array([Up[t:t + dl].any(0)
                              for t in range(K - dl + 1)])
                row.append(w.mean())
            ys.append(row)
        ys = np.array(ys)
        st = STYLE.get(s, {})
        ax.errorbar(deadlines, ys.mean(0), yerr=ys.std(0), label=disp(s),
                    ms=5, markerfacecolor="white", capsize=2.5, **st)
    ax.set_xscale("log")
    ax.set_xticks(deadlines)
    ax.set_xticklabels([str(x) for x in deadlines])
    ax.minorticks_off()
    ax.set_xlabel("Delivery deadline (rounds)")
    ax.set_ylabel("High-demand delivery success")
    ax.set_ylim(0, 0.7)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_face_deadline")


def fig_calib(warmup=30, nbins=10):
    """Gain-prediction reliability of FACE: mean realized adoption gain and
    adoption probability per predicted-gain decile (equal-count bins over
    out-of-sample evaluation pairs after the ridge warm-up). The realized
    gain is zero-inflated (most evaluated encoders do not improve the
    receiver), so equal-count bins are the meaningful calibration view."""
    if not os.path.exists(EVENTS_NPZ):
        print("  [skip] fig_face_calib: no events npz yet")
        return
    z = np.load(EVENTS_NPZ)
    if "Proposed__calib_all" not in z.files:
        print("  [skip] fig_face_calib: calib pairs missing")
        return
    c = z["Proposed__calib_all"]           # seed, round, predicted, realized
    m = c[:, 1] >= warmup
    pred, real = c[m, 2], c[m, 3]
    q = np.quantile(pred, np.linspace(0, 1, nbins + 1))
    px, mu, sd, pos = [], [], [], []
    for i in range(nbins):
        mm = (pred >= q[i]) & ((pred < q[i + 1]) if i < nbins - 1
                               else (pred <= q[i + 1]))
        px.append(pred[mm].mean())
        mu.append(real[mm].mean())
        sd.append(real[mm].std() / np.sqrt(mm.sum()))
        pos.append((real[mm] > 0).mean())
    px = np.array(px) * 1e3                # milli-gain axes: readable ticks
    mu = np.array(mu) * 1e3
    sd = np.array(sd) * 1e3
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    lim = 1.06 * max(px.max(), mu.max())
    ax.plot([0, lim], [0, lim], color="0.35", ls="--", lw=1.0,
            label="Perfect calibration", zorder=1)
    ax.errorbar(px, mu, yerr=sd, color="#d62728", marker="o", ms=5,
                capsize=2.5, lw=1.7, label="Realized gain (decile mean)",
                zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"Predicted gain $\widehat{v}_{i,x}$ "
                  r"($\times 10^{-3}$, decile mean)")
    ax.set_ylabel(r"Realized gain $v_{i,x}$ ($\times 10^{-3}$)")
    ax.grid(True, ls=":", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(px, pos, color="#4C72B0", marker="s", ms=4,
             markerfacecolor="white", lw=1.4, ls="-.",
             label="Adoption probability")
    ax2.set_ylabel(r"$\Pr(v_{i,x} > 0)$", color="#4C72B0")
    ax2.set_ylim(0, max(pos) * 1.5)
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    lift = mu[-1] / max(mu[0], 1e-9)
    ax.text(0.03, 0.97,
            f"$n$={len(pred):,} pairs\n"
            f"top/bottom-decile lift {lift:.1f}$\\times$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_face_calib")


def main():
    for fn in (fig_abl_bars,
               fig_abl_2panel,
               fig_deadline,
               fig_calib,
               lambda: fig_curves("mhop", "Beyond-encounter delivery ratio",
                                  "fig_face_beyond"),
               lambda: fig_curves("acc", "Test accuracy", "fig_face_pareto",
                                  cumulative_x="txmb"),
               lambda: fig_curves("tx", "Encoder transmissions / round",
                                  "fig_face_txrate")):
        try:
            fn()
        except Exception as e:
            print("  [face_figs] skipped:", e)


if __name__ == "__main__":
    main()
