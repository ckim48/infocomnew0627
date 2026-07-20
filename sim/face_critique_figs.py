"""
INFOCOM-style figures answering the setup/theory critiques:

  fig_eps_pred   -- empirical prediction-error constants (MAE/RMSE/ECE,
                    |v_hat - v| percentiles = eps_pred) from the recorded
                    out-of-sample calibration pairs (Sec. IV / Theorem 1).
  fig_eps_int    -- empirical interaction error eps_int: |sum of solo
                    marginal gains - joint gain| of accepted aggregation
                    sets (results/face_eint_probe.npz).
  fig_setup_sens -- REAL-backend setup-sensitivity (carrier ratio, poor-data
                    quality, data-quantity skew, LiDAR availability), FACE
                    vs the strongest baseline (results/sens_setup/*.npz).
  fig_dynamic    -- dynamic sensing environments: zone-conditioned
                    corruption while vehicles move; accuracy + deliveries
                    routed to currently-degraded vehicles
                    (results/face_dynamic_probe.npz).

Run:  python3 -m sim.face_critique_figs [eps_pred|eps_int|setup|dynamic|all]
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "Figures"
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 11,
    "axes.linewidth": 0.9, "lines.linewidth": 1.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False,
}
C_FACE, C_BASE = "#d62728", "#1f77b4"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def fig_eps_pred(npz="results/metrics_v2x_real_kitti_map.npz"):
    """Reliability diagram + |error| CDF of the out-of-sample reward
    prediction v_hat (real KITTI backend, all recorded delivery events)."""
    d = np.load(npz)
    c = d["Proposed__calib_all"]                 # seed, round, pred, realized
    p, r = c[:, 2], c[:, 3]
    err = np.abs(p - r)
    mae = float(err.mean())
    rmse = float(np.sqrt(((p - r) ** 2).mean()))
    m = p > 0
    q = np.quantile(p[m], np.linspace(0, 1, 11)); q[0] = 0; q[-1] += 1e-9
    mp, mr, ns = [], [], []
    for b in range(10):
        sel = m & (p >= q[b]) & (p < q[b + 1])
        if not sel.any():
            continue
        mp.append(p[sel].mean()); mr.append(r[sel].mean())
        ns.append(int(sel.sum()))
    mp, mr, ns = map(np.asarray, (mp, mr, ns))
    ece = float(np.sum(ns / ns.sum() * np.abs(mp - mr)))
    pct = {k: float(np.percentile(err, k)) for k in (50, 90, 95)}

    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
        # (a) reliability diagram over predicted-reward deciles
        lim = max(mp.max(), mr.max()) * 1.15
        a1.plot([0, lim], [0, lim], ls="--", lw=0.9, color="0.55",
                label="perfect calibration")
        a1.plot(mp, mr, marker="o", ms=4.5, color=C_FACE,
                markerfacecolor="white", label="decile mean")
        a1.set_xlabel(r"Predicted reward $\widehat{v}_{i,x}$ (decile mean)")
        a1.set_ylabel("Realized gain (mean)")
        a1.set_xlim(0, lim); a1.set_ylim(0, lim)
        a1.legend(loc="upper left", fontsize=9)
        a1.text(0.97, 0.06,
                f"ECE = {ece:.4f}\nlift (top/bottom) = "
                f"{mr[-1] / max(mr[0], 1e-9):.1f}$\\times$",
                transform=a1.transAxes, ha="right", va="bottom", fontsize=9)
        # (b) empirical eps_pred: CDF of |v_hat - v| (linear, covers p95+)
        xs = np.sort(err); ys = np.arange(1, len(xs) + 1) / len(xs)
        xmax = 0.06
        a2.plot(np.clip(xs, 0, xmax), ys, color=C_FACE)
        for k, ls in ((50, ":"), (90, "--"), (95, "-.")):
            a2.axvline(pct[k], ls=ls, lw=0.9, color="0.4")
            a2.text(pct[k] + 0.001, 0.06, f"p{k}={pct[k]:.3f}", rotation=90,
                    fontsize=8, ha="left", va="bottom", color="0.25")
        a2.set_xlim(0, xmax)
        a2.set_xlabel(r"$|\widehat{v}_{i,x} - v_{i,x}|$  "
                      r"(empirical $\epsilon_{\mathrm{pred}}$)")
        a2.set_ylabel("CDF")
        a2.set_ylim(0, 1.02)
        a2.text(0.96, 0.12,
                f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}\n"
                f"max = {err.max():.2f},  n = {len(err)}",
                transform=a2.transAxes, ha="right", va="bottom", fontsize=9)
        for ax, lab in ((a1, "(a)"), (a2, "(b)")):
            ax.grid(True, ls="--", lw=0.6, alpha=0.5)
            ax.text(0.5, -0.34, lab, transform=ax.transAxes, ha="center",
                    va="top", fontsize=11)
        fig.tight_layout()
        _save(fig, "fig_face_eps_pred")


def fig_eps_int(npz="results/face_eint_probe.npz"):
    """Empirical interaction error of accepted aggregation sets:
    eps_int = |sum_x v_solo(x) - v(A)|, recorded per aggregation event."""
    if not os.path.exists(npz):
        print("  [skip] fig_face_eps_int: run sim.face_eint_probe first")
        return
    d = np.load(npz)
    joint, solo, nsets = d["joint"], d["solo_sum"], d["nset"]
    multi = nsets >= 2                       # singletons have eps_int = 0
    eint = np.abs(solo - joint)
    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
        lim = max(joint.max(), solo.max()) * 1.1
        a1.plot([0, lim], [0, lim], ls="--", lw=0.9, color="0.55")
        a1.scatter(solo[~multi], joint[~multi], s=10, color="0.6",
                   alpha=0.4, lw=0, label=r"$|A|=1$")
        a1.scatter(solo[multi], joint[multi], s=12, color=C_FACE,
                   alpha=0.55, lw=0, label=r"$|A|\geq 2$")
        a1.set_xlabel(r"$\sum_{x \in A} v(\{x\})$  (sum of solo gains)")
        a1.set_ylabel(r"$v(A)$  (joint gain)")
        a1.legend(loc="upper left", fontsize=9)
        xs = np.sort(eint[multi]); ys = np.arange(1, len(xs) + 1) / len(xs)
        a2.plot(xs, ys, color=C_FACE)
        for k, ls in ((50, ":"), (90, "--"), (95, "-.")):
            v = float(np.percentile(eint[multi], k))
            a2.axvline(v, ls=ls, lw=0.9, color="0.4")
            a2.text(v, 0.04, f" p{k}={v:.3f}", rotation=90, fontsize=8,
                    ha="left", va="bottom", color="0.25")
        a2.set_xscale("symlog", linthresh=1e-3)
        a2.set_xlabel(r"$|\sum_x v(\{x\}) - v(A)|$  "
                      r"(empirical $\epsilon_{\mathrm{int}}$, $|A|\geq 2$)")
        a2.set_ylabel("CDF"); a2.set_ylim(0, 1.02)
        a2.text(0.97, 0.30,
                f"mean = {eint[multi].mean():.4f}\n"
                f"n = {int(multi.sum())} sets",
                transform=a2.transAxes, ha="right", va="bottom", fontsize=9)
        for ax, lab in ((a1, "(a)"), (a2, "(b)")):
            ax.grid(True, ls="--", lw=0.6, alpha=0.5)
            ax.text(0.5, -0.34, lab, transform=ax.transAxes, ha="center",
                    va="top", fontsize=11)
        fig.tight_layout()
        _save(fig, "fig_face_eps_int")


SENS_DIR = "results/sens_setup"
SENS_PANELS = [
    ("carrier", "Carrier fraction $\\rho$", 0.15,
     [0.05, 0.10, 0.15, 0.25, 0.40]),
    ("poorq", "Poor-data quality $Q^{\\mathrm{hi}}$", 0.30,
     [0.30, 0.50, 0.70]),
    ("skew", "Poor-vehicle data size (frames)", 8,
     [8, 30, 60]),
    ("plidar", "LiDAR availability $p_{\\mathrm{LiDAR}}$", 0.85,
     [0.60, 0.85, 1.00]),
]


def fig_setup_sens(tail=20):
    """2x2 REAL-backend setup sensitivity: final accuracy of FACE vs the
    strongest baseline as the data-heterogeneity setup varies."""
    with plt.rc_context(RC):
        fig, axg = plt.subplots(2, 2, figsize=(6.6, 4.6))
        for ax, (tag, xlabel, default, vals) in zip(axg.ravel(), SENS_PANELS):
            xs, yf, yb = [], [], []
            for v in vals:
                path = os.path.join(SENS_DIR, f"metrics_{tag}_{v}.npz")
                if not os.path.exists(path):
                    continue
                d = np.load(path)
                xs.append(v)
                yf.append(100 * d["Proposed__acc"][-tail:].mean())
                yb.append(100 * d["Learning-aware__acc"][-tail:].mean())
            if not xs:
                ax.set_axis_off(); continue
            ax.plot(xs, yf, marker="o", ms=4.5, color=C_FACE,
                    markerfacecolor="white", label="FACE")
            ax.plot(xs, yb, marker="s", ms=4.0, color=C_BASE,
                    markerfacecolor="white", label="Learning-aware")
            di = int(np.argmin(np.abs(np.asarray(xs) - default)))
            ax.plot([xs[di]], [yf[di]], marker="o", ms=5.5, color=C_FACE,
                    zorder=4)
            ax.set_xlabel(xlabel)
            ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        for ax in axg[:, 0]:
            ax.set_ylabel("Final accuracy (%)")
        h, l = axg[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.04), fontsize=10)
        for i, ax in enumerate(axg.ravel()):
            ax.text(0.5, -0.42, f"({'abcd'[i]})", transform=ax.transAxes,
                    ha="center", va="top", fontsize=11)
        fig.tight_layout(h_pad=2.6, rect=[0, 0, 1, 0.97])
        _save(fig, "fig_face_setup_sens")


def fig_dynamic(npz="results/face_dynamic_probe.npz", smooth=9):
    """Dynamic sensing environments: (a) accuracy under zone-conditioned
    corruption, (b) share of useful deliveries reaching vehicles whose
    current zone degrades one of their modalities."""
    if not os.path.exists(npz):
        print("  [skip] fig_face_dynamic: run sim.face_dynamic_probe first")
        return
    d = np.load(npz)
    schemes = [s for s in ("Proposed", "Caching-assisted", "V2V-aware",
                           "Learning-aware") if f"{s}__acc" in d.files]
    from .plotting import STYLE as STY, disp

    def sm(a):
        if smooth <= 1:
            return a
        ker = np.ones(smooth) / smooth
        return np.convolve(a, ker, mode="valid")

    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for s in schemes:
            a = d[f"{s}__acc"]
            x = np.arange(1, len(a) + 1)
            a1.plot(x[:len(sm(a))], 100 * sm(a), label=disp(s),
                    markevery=max(len(a) // 8, 1), markersize=5,
                    markerfacecolor="white", **STY[s])
            db = d[f"{s}__deg_deliv"]         # deliveries to degraded veh.
            a2.plot(x[:len(sm(db))], 100 * sm(db), label=disp(s),
                    markevery=max(len(db) // 8, 1), markersize=5,
                    markerfacecolor="white", **STY[s])
        frac = float(d["deg_frac"].mean())
        a2.axhline(100 * frac, ls="--", lw=0.9, color="0.45")
        a2.text(0.03, 100 * frac + 1.5, "fraction of degraded vehicles",
                fontsize=8, color="0.3",
                transform=a2.get_yaxis_transform())
        a1.set_xlabel("Global round $k$")
        a1.set_ylabel("Test accuracy (%)")
        a2.set_xlabel("Global round $k$")
        a2.set_ylabel("Useful deliveries to\ndegraded vehicles (%)")
        a1.legend(fontsize=8.5, loc="lower right")
        for ax, lab in ((a1, "(a)"), (a2, "(b)")):
            ax.grid(True, ls="--", lw=0.6, alpha=0.5)
            ax.text(0.5, -0.34, lab, transform=ax.transAxes, ha="center",
                    va="top", fontsize=11)
        fig.tight_layout()
        _save(fig, "fig_face_dynamic")


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("eps_pred", "all"):
        fig_eps_pred()
    if what in ("eps_int", "all"):
        fig_eps_int()
    if what in ("setup", "all"):
        fig_setup_sens()
    if what in ("dynamic", "all"):
        fig_dynamic()
