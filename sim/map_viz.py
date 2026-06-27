"""
Spatial visualization on the real Ingolstadt (InTAS) map.

Renders the genuine road network and the vehicle cohort, contrasting the
proposed road-topology & traffic-aware scheme against a road/traffic-agnostic
scheme (Caching-assisted: same store-carry-forward, but no GAT future-contact
prediction). Vehicles are coloured by achieved model accuracy, so the broader
green coverage under the proposed scheme shows that road/traffic-aware
forwarding propagates strong encoders to more vehicles -- including those far
from the strong-encoder owners.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sumolib
from .config import Config, SCHEMES
from .intas_trace import get_or_build_trace, NET_FILE
from .mobility import RoadNetwork, MobilitySim
from .mfl import MultimodalFL
from .hgat import train_hgat, future_contact_scores
from .algorithm import CachingForwarding
from .simulator import make_modality_availability


def _edge_polylines(ctr):
    """Real road-segment polylines (centred by ctr) from the InTAS net."""
    net = sumolib.net.readNet(NET_FILE)
    edges = [e for e in net.getEdges() if not e.isSpecial()]
    segs = []
    raw_mid0 = None
    for k, e in enumerate(edges):
        shp = np.array(e.getShape())
        if k == 0:
            raw_mid0 = shp.mean(0)
        segs.append(shp - ctr)
    return segs, raw_mid0


def _run_one(cfg, mob, gammas, scheme, snap_k):
    """Run a scheme; return per-vehicle achieved accuracy at round snap_k."""
    avail_rng = np.random.default_rng(cfg.seed + 7)
    modality_avail = make_modality_availability(cfg, avail_rng)
    rng = np.random.default_rng(cfg.seed)
    mfl = MultimodalFL(cfg, rng, modality_avail)
    alg = CachingForwarding(cfg, mfl, mob, scheme, seed=cfg.seed)
    per_veh = None
    for k in range(mob.Krounds):
        mob.k = k
        mfl.local_train()
        g = gammas[k] if alg.flags["use_dis"] or alg.flags["cache_policy"] == "psi" \
            else np.zeros(mob.N)
        alg.run_round(k, g)
        if k == snap_k:
            per_veh = _per_vehicle_acc(mfl)
    return per_veh, mfl


def _per_vehicle_acc(mfl):
    acc = np.zeros(mfl.N)
    for i in range(mfl.N):
        qs = [mfl.q_eff(i, r) for r in mfl.avail[i]]
        acc[i] = np.mean(qs) if qs else 0.0
    return acc


def make_map_figure(cfg=None, device="cpu", snap_k=None):
    cfg = cfg or Config()
    os.makedirs(cfg.figures_dir, exist_ok=True)
    snap_k = snap_k if snap_k is not None else cfg.K - 1

    cache_path = os.path.join(cfg.results_dir, f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache_path, begin=34050.0, dt=2.0, warmup_s=480.0)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)

    # recover centring offset and load real road polylines
    import numpy as _np
    net_segs, raw_mid0 = _edge_polylines(_np.zeros(2))
    ctr = raw_mid0 - road.mid[0]
    net_segs = [s - ctr for s in net_segs]

    # train predictor + Gamma for the proposed scheme
    print("  [map] training GAT predictor ...")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei, device=device))
    gammas = np.array(gammas)

    # run the two contrasted schemes
    print("  [map] running Proposed and Caching-assisted ...")
    acc_prop, _ = _run_one(cfg, mob, gammas, "Proposed", snap_k)
    acc_cach, _ = _run_one(cfg, mob, gammas, "Caching-assisted", snap_k)

    # snapshot positions + traffic at the snapshot round
    mob.k = snap_k
    xy = mob.vehicle_xy()
    dens = mob.density()                                  # per-segment traffic density

    # identify strong-encoder owner vehicles (sources) for marking
    avail0 = make_modality_availability(cfg, np.random.default_rng(cfg.seed + 7))
    mfl0 = MultimodalFL(cfg, np.random.default_rng(cfg.seed), avail0)
    src = np.array([max([mfl0.strength[(i, r)] for r in mfl0.avail[i]]) >= 0.8
                    for i in range(mob.N)])

    # bounding box around the cohort
    pad = 600
    x0, x1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad
    y0, y1 = xy[:, 1].min() - pad, xy[:, 1].max() + pad

    # road segments within the viewport, with their traffic density
    seg_lines, seg_d = [], []
    for k, s in enumerate(net_segs):
        mx, my = s[:, 0].mean(), s[:, 1].mean()
        if x0 <= mx <= x1 and y0 <= my <= y1:
            seg_lines.append(s)
            seg_d.append(dens[k])
    seg_d = np.array(seg_d)
    # robust normalisation for the traffic heat (cap at 90th pct)
    hi = np.quantile(seg_d[seg_d > 0], 0.90) if (seg_d > 0).any() else 1.0
    seg_dn = np.clip(seg_d / (hi + 1e-9), 0, 1)
    busy = seg_dn > 0.05

    # vehicle motion trails over the last `tail` rounds (mobility)
    tail = 12
    t0 = max(0, snap_k - tail)
    trails = mob.veh_xy[t0:snap_k + 1]                    # [T, N, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharex=True, sharey=True)
    titles = [
        "(a) Proposed: road-topology & traffic-aware",
        "(b) Caching-assisted: road/traffic-agnostic",
    ]
    accs = [acc_prop, acc_cach]
    for ax, title, acc, road_aware in zip(axes, titles, accs, [True, False]):
        # base road network (both panels): light grey
        ax.add_collection(LineCollection(seg_lines, colors="0.82", linewidths=0.5,
                                         alpha=0.9, zorder=0))
        # traffic-density heat overlay (only the road/traffic-aware panel)
        if road_aware and busy.any():
            heat = [seg_lines[k] for k in np.where(busy)[0]]
            lc = LineCollection(heat, cmap="YlOrRd", linewidths=1.6, alpha=0.95, zorder=1)
            lc.set_array(seg_dn[busy])
            ax.add_collection(lc)
        # vehicle motion trails
        for i in range(mob.N):
            ax.plot(trails[:, i, 0], trails[:, i, 1], "-", color="0.55",
                    lw=0.5, alpha=0.5, zorder=2)
        # vehicles coloured by achieved accuracy
        sc = ax.scatter(xy[~src, 0], xy[~src, 1], c=acc[~src], cmap="RdYlGn",
                        vmin=0.2, vmax=1.0, s=30, edgecolors="k", linewidths=0.3, zorder=3)
        # strong-encoder owners (sources)
        ax.scatter(xy[src, 0], xy[src, 1], c="blue", marker="*", s=150,
                   edgecolors="k", linewidths=0.4, zorder=4, label="Strong-encoder owner")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.text(0.02, 0.02, f"mean acc = {acc.mean():.3f}", transform=ax.transAxes,
                fontsize=10, va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Vehicle model accuracy")
    # traffic-heat reference
    sm = plt.cm.ScalarMappable(cmap="YlOrRd")
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.06)
    cb2.set_label("Road traffic density (proposed)")
    cb2.set_ticks([])
    fig.suptitle(f"Ingolstadt (InTAS) — encoder propagation at round k={snap_k}",
                 fontsize=12, y=0.99)
    p = os.path.join(cfg.figures_dir, "fig_map_propagation.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", p)


if __name__ == "__main__":
    make_map_figure()
