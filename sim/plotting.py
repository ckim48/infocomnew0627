"""
Generate the simulation-result figures (Sec. V-B) from saved metrics.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, SCHEMES

STYLE = {
    "Proposed":         dict(color="#d62728", marker="o", ls="-"),
    "Caching-assisted": dict(color="#1f77b4", marker="s", ls="--"),
    "V2V-aware":        dict(color="#2ca02c", marker="^", ls="-."),
    "Learning-aware":   dict(color="#7f7f7f", marker="d", ls=":"),
}

# Display name for each scheme in legends/labels (the proposed scheme is FACE).
DISPLAY = {"Proposed": "FACE"}


def disp(scheme):
    return DISPLAY.get(scheme, scheme)


def _load(cfg):
    d = np.load(os.path.join(cfg.results_dir, "metrics.npz"))
    res = {}
    for s in SCHEMES:
        res[s] = {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(s + "__")}
    return res


def _band(ax, x, mean, std, color):
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, lw=0)


def _marker_idx(n, step):
    return np.arange(0, n, step)


def plot_all(cfg=None):
    cfg = cfg or Config()
    os.makedirs(cfg.figures_dir, exist_ok=True)
    res = _load(cfg)
    K = len(res["Proposed"]["loss"])
    x = np.arange(1, K + 1)
    mi = _marker_idx(K, max(K // 12, 1))

    def newfig():
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        return fig, ax

    def finish(fig, ax, ylabel, fname, loc="best"):
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc=loc, fontsize=9)
        fig.tight_layout()
        p = os.path.join(cfg.figures_dir, fname)
        fig.savefig(p, dpi=200); fig.savefig(p.replace(".png", ".pdf"))
        plt.close(fig)
        print("  saved", p)

    # (a) mean validation loss vs round
    fig, ax = newfig()
    for s in SCHEMES:
        ax.plot(x, res[s]["loss"], label=disp(s), markevery=mi, ms=5, **STYLE[s])
        _band(ax, x, res[s]["loss"], res[s].get("loss_std"), STYLE[s]["color"])
    finish(fig, ax, "Mean validation loss", "fig_loss.png")

    # (b) mean accuracy vs round
    fig, ax = newfig()
    for s in SCHEMES:
        ax.plot(x, res[s]["acc"], label=disp(s), markevery=mi, ms=5, **STYLE[s])
        _band(ax, x, res[s]["acc"], res[s].get("acc_std"), STYLE[s]["color"])
    finish(fig, ax, "Mean model accuracy", "fig_accuracy.png", loc="lower right")

    # (c) worst-decile (most needy) accuracy vs round
    fig, ax = newfig()
    for s in SCHEMES:
        ax.plot(x, res[s]["tail"], label=disp(s), markevery=mi, ms=5, **STYLE[s])
        _band(ax, x, res[s]["tail"], res[s].get("tail_std"), STYLE[s]["color"])
    finish(fig, ax, "Poor-data vehicle accuracy", "fig_poor_accuracy.png", loc="lower right")

    # (d) cumulative successful encoder deliveries
    fig, ax = newfig()
    for s in SCHEMES:
        ax.plot(x, np.cumsum(res[s]["tx"]), label=disp(s), markevery=mi, ms=5, **STYLE[s])
    finish(fig, ax, "Cumulative forwarded encoders", "fig_forwarding.png", loc="upper left")

    # (e) virtual queue backlog (Lyapunov stability)
    fig, ax = newfig()
    for s in SCHEMES:
        ax.plot(x, res[s]["qlen"], label=disp(s), markevery=mi, ms=5, **STYLE[s])
    finish(fig, ax, "Avg. virtual queue backlog $\\bar Q(k)$", "fig_queue.png", loc="upper left")

    # (f) bar chart of final accuracy
    fig, ax = newfig()
    finals = [res[s]["acc"][-1] for s in SCHEMES]
    bars = ax.bar(range(len(SCHEMES)), finals,
                  color=[STYLE[s]["color"] for s in SCHEMES])
    ax.set_xticks(range(len(SCHEMES)))
    ax.set_xticklabels([disp(s) for s in SCHEMES], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Final mean accuracy")
    ax.set_ylim(min(finals) * 0.97, max(finals) * 1.01)
    for b, v in zip(bars, finals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    fig.tight_layout()
    p = os.path.join(cfg.figures_dir, "fig_final_acc_bar.png")
    fig.savefig(p, dpi=200); fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)
    print("  saved", p)


if __name__ == "__main__":
    plot_all()
