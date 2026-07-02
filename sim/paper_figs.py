"""
INFOCOM-style result figures (two-panel, top legend, open markers, serif font),
matching the requested template. Reads results/metrics.npz (large-scale InTAS
simulation) and results/metrics_real.npz (real KITTI multimodal FL).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
})

SCHEMES = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]
# Display name in legends (the proposed scheme is FACE).
DISPLAY = {"Proposed": "FACE"}
# style matched to the template: Proposed=red solid o, then green--s, blue-.D, black:^
STY = {
    "Proposed":         dict(color="#e8000b", ls="-",  marker="o"),
    "Caching-assisted": dict(color="#1f9e3d", ls="--", marker="s"),
    "V2V-aware":        dict(color="#1f5fd0", ls="-.", marker="D"),
    "Learning-aware":   dict(color="#000000", ls=":",  marker="^"),
}


def _load(path):
    d = np.load(path)
    return {s: {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(s + "__")}
            for s in SCHEMES}


def _smooth(y, w):
    """Centered moving average with edge-aware normalization."""
    if w <= 1:
        return y
    num = np.convolve(y, np.ones(w), "same")
    den = np.convolve(np.ones_like(y), np.ones(w), "same")
    return num / den


def _plot_panel(ax, res, key, title, xlabel, ylabel, nmark=11, band=False, smooth=1):
    K = len(res["Proposed"][key]); x = np.arange(1, K + 1)
    me = max(K // nmark, 1)
    for s in SCHEMES:
        y = _smooth(res[s][key], smooth)
        ax.plot(x, y, label=DISPLAY.get(s, s), markevery=me, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.2, **STY[s])
        if band and (key + "_std") in res[s]:
            sd = res[s][key + "_std"]
            ax.fill_between(x, y - sd, y + sd, color=STY[s]["color"], alpha=0.10, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, K)
    ax.grid(True, ls="--", lw=0.6, alpha=0.5)
    ax.set_title(title, y=-0.32, fontsize=12)


def _legend(fig, axes):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.4, handlelength=2.6, fontsize=11)


def main(outdir="Figures"):
    os.makedirs(outdir, exist_ok=True)
    kitti = _load("results/metrics_real_kitti.npz")
    nusc = _load("results/metrics_real_nuscenes.npz")

    # ---- Figure 1: test accuracy, (a) KITTI, (b) nuScenes (two datasets) ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _plot_panel(axes[0], kitti, "acc", "(a) KITTI",
                "Global round $k$", "Test accuracy")
    _plot_panel(axes[1], nusc, "acc", "(b) nuScenes",
                "Global round $k$", "Test accuracy")
    _legend(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_infocom_accuracy.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure: 2x2 convergence -- test loss (top) and test accuracy
    # (bottom) on the two real multimodal datasets (columns) ----
    if "loss" in kitti["Proposed"] and "loss" in nusc["Proposed"]:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
        _plot_panel(axes[0, 0], kitti, "loss", "(a) KITTI",
                    "Global round $k$", "Training loss", smooth=5)
        _plot_panel(axes[0, 1], nusc, "loss", "(b) nuScenes",
                    "Global round $k$", "Training loss", smooth=5)
        _plot_panel(axes[1, 0], kitti, "acc", "(c) KITTI",
                    "Global round $k$", "Test accuracy")
        _plot_panel(axes[1, 1], nusc, "acc", "(d) nuScenes",
                    "Global round $k$", "Test accuracy")
        _legend(fig, [axes[0, 0]])
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=3.2)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"fig_infocom_convergence.{ext}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        print("  [skip] fig_infocom_convergence: no 'loss' in metrics "
              "(re-run sim.real_fl to record it)")

    # ---- Figure 2: poor-data (needy) vehicle accuracy, two datasets ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _plot_panel(axes[0], kitti, "poor", "(a) KITTI",
                "Global round $k$", "Poor-data accuracy")
    _plot_panel(axes[1], nusc, "poor", "(b) nuScenes",
                "Global round $k$", "Poor-data accuracy")
    _legend(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_infocom_poor.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: large-scale InTAS simulation (separate scope) ----
    sim = _load("results/metrics.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _plot_panel(axes[0], sim, "acc", "(a) Test accuracy",
                "Global round $k$", "Test accuracy")
    _plot_panel(axes[1], sim, "tail", "(b) Poor-data accuracy",
                "Global round $k$", "Poor-data accuracy")
    _legend(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_infocom_largescale.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=== final values (real datasets) ===")
    for s in SCHEMES:
        print(f"  {s:16s} KITTI acc {kitti[s]['acc'][-1]:.3f} poor {kitti[s]['poor'][-1]:.3f} | "
              f"nuScenes acc {nusc[s]['acc'][-1]:.3f} poor {nusc[s]['poor'][-1]:.3f}")
    print("saved fig_infocom_accuracy (KITTI/nuScenes) / fig_infocom_poor / fig_infocom_largescale")


if __name__ == "__main__":
    main()
