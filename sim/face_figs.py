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


def main():
    for fn in (fig_abl_bars,
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
