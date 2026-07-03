"""
Meaningful 1x4 map subfigure on the real Seoul map: the SAME vehicle cohort,
coloured by the model accuracy each vehicle achieves under the Proposed
road/traffic-aware scheme (FACE) vs the three baselines. Greener = higher
accuracy; the broader green coverage under FACE shows strong encoders reach
more vehicles across the real Seoul road network.

Reuses the V2X trace (sim/v2x_trace) + the per-vehicle accuracy tracking from
sim/map_viz, and georeferences vehicles onto a contextily basemap.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .hgat import train_hgat, future_contact_scores
from .v2x_trace import build_v2x_trace, SEOUL_NET
from .map_viz import _run_one
from .plotting import disp

MAP_SCHEMES = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]


def _compute(cfg, device, snap_k, cache):
    """Heavy step: trace + GAT + per-vehicle accuracy + geo positions. Cached."""
    import sumolib
    from pyproj import Transformer
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    trace = build_v2x_trace(cfg)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    ctr = np.asarray(trace["ctr"], dtype=np.float64)
    snap_k = snap_k if snap_k is not None else mob.Krounds - 1

    print("  [v2x-map] training GAT + Gamma ...")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=30)
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei,
                                            device=device))
    gammas = np.array(gammas)

    accs = {}
    for s in MAP_SCHEMES:
        print(f"  [v2x-map] running {s} ...")
        accs[s], _ = _run_one(cfg, mob, gammas, s, snap_k)

    mob.k = snap_k
    net = sumolib.net.readNet(SEOUL_NET)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    net_xy = mob.vehicle_xy() + ctr
    vm = np.empty((mob.N, 2))
    for i, (x, y) in enumerate(net_xy):
        lon, lat = net.convertXY2LonLat(float(x), float(y))
        vm[i] = tf.transform(lon, lat)

    np.savez(cache, vm=vm, snap_k=snap_k,
             **{f"acc_{s}": accs[s] for s in MAP_SCHEMES})
    return vm, accs, snap_k


def make_v2x_map_subfig(cfg=None, device="cpu", num_vehicles=180, snap_k=None,
                        basemap="voyager", use_cache=True):
    import contextily as cx

    cfg = cfg or Config()
    cfg.num_vehicles = num_vehicles
    fig_dir = cfg.figures_dir
    os.makedirs(fig_dir, exist_ok=True)
    cache = os.path.join(cfg.results_dir, "v2x_map_cache.npz")

    d = np.load(cache) if (use_cache and os.path.exists(cache)) else None
    if d is not None and all(f"acc_{s}" in d.files for s in MAP_SCHEMES):
        vm = d["vm"]; snap_k = int(d["snap_k"])
        accs = {s: d[f"acc_{s}"] for s in MAP_SCHEMES}
        print(f"  [v2x-map] re-plotting from cache {cache}")
    else:
        vm, accs, snap_k = _compute(cfg, device, snap_k, cache)

    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    src = providers.get(basemap, providers["voyager"])

    fig, axgrid = plt.subplots(2, 2, figsize=(9.6, 9.2),
                               sharex=True, sharey=True)
    axes = axgrid.ravel()
    pad = 400
    xlim = (vm[:, 0].min() - pad, vm[:, 0].max() + pad)
    ylim = (vm[:, 1].min() - pad, vm[:, 1].max() + pad)
    sc = None
    for ax, s in zip(axes, MAP_SCHEMES):
        acc = accs[s]
        sc = ax.scatter(vm[:, 0], vm[:, 1], c=acc, cmap="RdYlGn", vmin=0.2,
                        vmax=1.0, s=30, edgecolors="k", linewidths=0.3,
                        alpha=0.95, zorder=4)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        cx.add_basemap(ax, crs="EPSG:3857", source=src, zoom=14,
                       attribution_size=4)
        # method label BELOW the panel (xlabel), so it never overlaps the dots
        ax.set_xlabel(f"{disp(s)}\nmean acc = {acc.mean():.3f}", fontsize=12)

    fig.subplots_adjust(wspace=0.04, hspace=0.16)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Vehicle model accuracy")

    out = os.path.join(fig_dir, "fig_infocom_v2x_map.png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out, " ".join(f"{disp(s)}={accs[s].mean():.3f}"
                                   for s in MAP_SCHEMES))
    return out


if __name__ == "__main__":
    make_v2x_map_subfig()
