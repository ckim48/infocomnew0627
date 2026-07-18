"""
Motivation figures (data-driven) for the Introduction, computed from the real
Ingolstadt (InTAS) mobility trace and the real modality-encoder sizes.

Fig. 1 (Challenge 1 -- limited/short contacts under mobility):
  (a) CDF of realized V2V contact durations.
  (b) Fraction of contacts long enough to exchange all required modality
      encoders, vs. the number of modalities (unimodal is easy, multimodal is
      not) -- i.e. multiple encoders cannot all be shared in one contact.

Fig. 2 (Challenge 2 -- heterogeneous demand, limited direct reach):
  (a) Fraction of vehicles that obtain a strong modality encoder over time:
      direct V2V only vs. store-carry-forward (caching).
  (b) Final reach vs. the fraction of vehicles that own a strong encoder
      (the scarcer the strong encoders, the more direct-only sharing fails).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .intas_trace import get_or_build_trace
from .mobility import RoadNetwork, MobilitySim

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})
RED, BLU, GRN, BLK = "#e8000b", "#1f5fd0", "#1f9e3d", "#000000"


def _adj_stack(cfg, mob):
    K, N = mob.Krounds, mob.N
    A = np.zeros((K, N, N), dtype=bool)
    for k in range(K):
        mob.k = k
        A[k] = mob.v2v_graph() > 0
    return A


def _contact_durations(A, dt):
    """All V2V contact-span durations (s) from the adjacency stack."""
    K, N, _ = A.shape
    durs = []
    for i in range(N):
        for j in range(i + 1, N):
            seq = A[:, i, j]
            run = 0
            for k in range(K):
                if seq[k]:
                    run += 1
                elif run > 0:
                    durs.append(run * dt); run = 0
            if run > 0:
                durs.append(run * dt)
    return np.array(durs)


def _reach(A, sources, carry):
    """Fraction of vehicles served over rounds.
    carry=True: store-carry-forward (any vehicle meeting a server becomes a
    server); carry=False: served only by directly meeting an original source."""
    K, N, _ = A.shape
    served = np.zeros(N, dtype=bool); served[list(sources)] = True
    src = served.copy()
    out = []
    for k in range(K):
        nb = A[k]
        if carry:
            newly = nb[served].any(axis=0) if served.any() else np.zeros(N, bool)
            served = served | newly
        else:
            newly = nb[src].any(axis=0) if src.any() else np.zeros(N, bool)
            served = served | newly
        out.append(served.mean())
    return np.array(out)


def make(cfg=None, rate=2.0):
    # realistic urban C-V2X: NLOS-limited range ~100 m and effective
    # throughput ~16 Mbps (=2 MB/s) under congestion
    cfg = cfg or Config()
    cfg.num_vehicles = 150; cfg.K = 150; cfg.comm_range = 100.0
    os.makedirs(cfg.figures_dir, exist_ok=True)
    cache = os.path.join(cfg.results_dir, f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache, begin=34050.0, dt=2.0, warmup_s=480.0)
    road = RoadNetwork(trace); mob = MobilitySim(cfg, road, trace)
    A = _adj_stack(cfg, mob)
    dt = mob.dt

    # ---------- compute all panel data ----------
    durs = _contact_durations(A, dt)
    sizes = np.sort([cfg.encoder_size[r] for r in ["camera", "lidar", "radar", "gps"]])
    cum = np.cumsum(sizes)
    cap = durs * rate
    frac_all = [float((cap >= cum[n - 1]).mean()) for n in range(1, 5)]

    rng = np.random.default_rng(cfg.seed); N = mob.N; nseed = 8
    n_src = max(int(0.15 * N), 1)
    direct_runs, carry_runs = [], []
    for _ in range(nseed):
        src = set(rng.choice(N, n_src, replace=False))
        direct_runs.append(_reach(A, src, carry=False))
        carry_runs.append(_reach(A, src, carry=True))
    direct = np.mean(direct_runs, 0); carry = np.mean(carry_runs, 0)
    x = np.arange(1, mob.Krounds + 1); mi = np.arange(0, mob.Krounds, 12)

    fracs = [0.05, 0.10, 0.15, 0.20, 0.30]
    fd, fc = [], []
    for fg in fracs:
        ns = max(int(fg * N), 1); dd, cc = [], []
        for _ in range(nseed):
            src = set(rng.choice(N, ns, replace=False))
            dd.append(_reach(A, src, carry=False)[-1])
            cc.append(_reach(A, src, carry=True)[-1])
        fd.append(np.mean(dd)); fc.append(np.mean(cc))

    # ---------- single 2x2 figure ----------
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))
    # (a) contact-duration CDF
    xs = np.sort(durs); cdf = np.arange(1, len(xs) + 1) / len(xs)
    ax[0, 0].plot(xs, cdf, color=RED)
    ax[0, 0].axvline(np.median(durs), color=BLK, ls=":", lw=1.2)
    ax[0, 0].text(np.median(durs) * 1.07, 0.18, f"median\n{np.median(durs):.0f}s", fontsize=9)
    ax[0, 0].set_xlabel("V2V contact duration (s)", labelpad=6); ax[0, 0].set_ylabel("CDF")
    ax[0, 0].set_xlim(0, np.quantile(durs, 0.98)); ax[0, 0].set_ylim(0, 1)
    ax[0, 0].grid(True, ls="--", lw=0.6, alpha=0.5)
    ax[0, 0].set_title("(a)", y=-0.40, fontsize=12)
    # (b) multimodal exchange burden
    ax[0, 1].bar(range(1, 5), frac_all, color=[GRN, BLU, "#f0a020", RED], width=0.6,
                 edgecolor="k", linewidth=0.6)
    for n, f in zip(range(1, 5), frac_all):
        ax[0, 1].text(n, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)
    ax[0, 1].set_xticks(range(1, 5))
    ax[0, 1].set_xlabel("Number of modality encoders", labelpad=6)
    ax[0, 1].set_ylabel("Exchange success prob.")
    ax[0, 1].set_ylim(0, 1.08); ax[0, 1].grid(True, axis="y", ls="--", lw=0.6, alpha=0.5)
    ax[0, 1].set_title("(b)", y=-0.40, fontsize=12)
    # (c) reach over rounds
    ax[1, 0].plot(x, carry, color=RED, marker="o", markevery=mi, ms=5,
                  markerfacecolor="white", markeredgewidth=1.2, label="Store-carry-forward")
    ax[1, 0].plot(x, direct, color=BLK, ls=":", marker="^", markevery=mi, ms=5,
                  markerfacecolor="white", markeredgewidth=1.2, label="Direct V2V only")
    ax[1, 0].set_xlabel("Global round $k$", labelpad=6); ax[1, 0].set_ylabel("Ratio of reached vehicles")
    ax[1, 0].set_xlim(0, mob.Krounds); ax[1, 0].set_ylim(0, 1.02)
    ax[1, 0].grid(True, ls="--", lw=0.6, alpha=0.5); ax[1, 0].legend(fontsize=9, loc="lower right")
    ax[1, 0].set_title("(c)", y=-0.40, fontsize=12)
    # (d) effect of encoder scarcity
    ax[1, 1].plot(np.array(fracs) * 100, fc, color=RED, marker="o", ms=6,
                  markerfacecolor="white", markeredgewidth=1.2, label="Store-carry-forward")
    ax[1, 1].plot(np.array(fracs) * 100, fd, color=BLK, ls=":", marker="^", ms=6,
                  markerfacecolor="white", markeredgewidth=1.2, label="Direct V2V only")
    ax[1, 1].set_xlabel("Strong-encoder owners (\\%)", labelpad=6)
    ax[1, 1].set_ylabel("Ratio of reached vehicles")
    ax[1, 1].set_ylim(0, 1.02); ax[1, 1].grid(True, ls="--", lw=0.6, alpha=0.5)
    ax[1, 1].legend(fontsize=9, loc="lower right")
    ax[1, 1].set_title("(d)", y=-0.40, fontsize=12)

    fig.tight_layout(); fig.subplots_adjust(hspace=0.55, wspace=0.34)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_motivation.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=== motivation stats ===")
    print(f"  contacts: {len(durs)}, median duration {np.median(durs):.1f}s, "
          f"mean {durs.mean():.1f}s")
    print(f"  frac contacts exchanging all N encoders: "
          + ", ".join(f"N={n}:{f:.2f}" for n, f in zip(range(1, 5), frac_all)))
    print(f"  reach@end direct {direct[-1]:.2f} vs carry {carry[-1]:.2f}")
    print("  saved fig_motivation (2x2)")


def fig_seoul_encounter(hmax=20, out="fig_motiv_encounter"):
    """Sec. II motivation on the real Seoul-Gangnam trace: (a) probability
    that a high-demand vehicle encounters any vehicle / any encoder-carrier
    vehicle / one specific carrier within h rounds (windowed); (b) CDF of the
    first direct DV-CV meeting time over all pairs -- most pairs never meet
    during the whole trace, so encoders must be ferried via relays."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 11,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    cfg = Config()
    tr = np.load(os.path.join(cfg.results_dir, "v2x_seoul_trace.npz"))
    ev = np.load(os.path.join(cfg.results_dir,
                              "metrics_v2x_real_kitti_events.npz"))
    xy = tr["veh_xy"]                       # (K, N, 2), metres
    pm = ev["Proposed__pmask_all"][0]       # high-demand (DV) mask
    cv = ~pm                                # encoder carriers (ECVs)
    K, N, _ = xy.shape
    C = np.zeros((K, N, N), bool)
    for k in range(K):
        d = np.linalg.norm(xy[k][:, None, :] - xy[k][None, :, :], axis=2)
        C[k] = d <= cfg.comm_range
        np.fill_diagonal(C[k], False)
    dvi, cvi = np.where(pm)[0], np.where(cv)[0]
    anyc = C.any(2)[:, dvi]                 # (K, nDV) any-vehicle contact
    anycv = C[:, dvi][:, :, cvi].any(2)     # any-carrier contact
    pair = C[:, dvi][:, :, cvi]             # (K, nDV, nCV) specific pairs
    hs = np.arange(1, hmax + 1)
    p_any, p_cv, p_one = [], [], []
    for h in hs:
        w = K - h + 1
        p_any.append(np.mean([anyc[t:t + h].any(0) for t in range(w)]))
        p_cv.append(np.mean([anycv[t:t + h].any(0) for t in range(w)]))
        p_one.append(np.mean([pair[t:t + h].any(0) for t in range(w)]))
    # first DV-CV meeting time CDF over pairs
    met = np.zeros((len(dvi), len(cvi)), bool)
    first = np.full((len(dvi), len(cvi)), K + 1)
    for k in range(K):
        new = pair[k] & ~met
        first[new] = k + 1
        met |= pair[k]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.semilogy(hs, p_any, marker="o", ms=4, color="#4C72B0",
                markerfacecolor="white", label="Any vehicle")
    ax.semilogy(hs, p_cv, marker="s", ms=4, color="#DD8452",
                markerfacecolor="white", label="Any carrier (CV)")
    ax.semilogy(hs, p_one, marker="^", ms=4, color="#C44E52",
                markerfacecolor="white", label="One specific CV")
    ax.set_xlabel("Window $h$ (rounds)")
    ax.set_ylabel("Encounter probability")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(1e-3, 1.5)
    ax = axes[1]
    xs = np.arange(0, K + 1)
    cdf = [(first <= x).mean() for x in xs]
    ax.plot(xs, cdf, color="#C44E52", lw=1.8)
    ax.axhline(met.mean(), color="0.4", ls=":", lw=1.0)
    ax.text(2, met.mean() + 0.06,
            f"{100 * (1 - met.mean()):.0f}% of DV–CV pairs never\n"
            f"meet during the whole trace", fontsize=8.5, va="bottom")
    ax.set_xlabel("First direct DV–CV meeting (round)")
    ax.set_ylabel("CDF over pairs")
    ax.set_xlim(0, K)
    ax.set_ylim(0, 1)
    for i, ax in enumerate(axes):
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.text(0.5, -0.42, f"({'ab'[i]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=11)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06), columnspacing=1.2,
               handlelength=1.9, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        for d in ("Figures", "new_result"):
            os.makedirs(d, exist_ok=True)
            fig.savefig(os.path.join(d, f"{out}.{ext}"), dpi=300,
                        bbox_inches="tight")
    plt.close(fig)
    print("  saved", out,
          f"(P_any(1)={p_any[0]:.2f}, P_cv(1)={p_cv[0]:.2f}, "
          f"never-meet={100 * (1 - met.mean()):.1f}%)")


if __name__ == "__main__":
    make()
