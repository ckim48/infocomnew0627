"""
Meaningful 1x2 map subfigure on the real Seoul map: the SAME vehicle cohort,
coloured by the model accuracy each vehicle achieves under (a) the Proposed
road/traffic-aware scheme vs (b) the road/traffic-agnostic Caching-assisted
baseline. Greener = higher accuracy; the broader green coverage under Proposed
shows strong encoders reach more vehicles across the real Seoul road network.

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


def make_v2x_map_subfig(cfg=None, device="cpu", num_vehicles=180, snap_k=None,
                        basemap="positron"):
    import sumolib
    import contextily as cx
    from pyproj import Transformer

    cfg = cfg or Config()
    cfg.num_vehicles = num_vehicles
    fig_dir = cfg.figures_dir
    os.makedirs(fig_dir, exist_ok=True)

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

    print("  [v2x-map] running Proposed and Caching-assisted ...")
    acc_prop, _ = _run_one(cfg, mob, gammas, "Proposed", snap_k)
    acc_cach, _ = _run_one(cfg, mob, gammas, "Caching-assisted", snap_k)

    # vehicle positions at the snapshot -> lon/lat -> Web Mercator
    mob.k = snap_k
    net = sumolib.net.readNet(SEOUL_NET)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    net_xy = mob.vehicle_xy() + ctr
    vm = np.empty((mob.N, 2))
    for i, (x, y) in enumerate(net_xy):
        lon, lat = net.convertXY2LonLat(float(x), float(y))
        vm[i] = tf.transform(lon, lat)

    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    src = providers.get(basemap, providers["positron"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), sharex=True, sharey=True)
    pad = 400
    xlim = (vm[:, 0].min() - pad, vm[:, 0].max() + pad)
    ylim = (vm[:, 1].min() - pad, vm[:, 1].max() + pad)
    panels = [("(a) Proposed (road/traffic-aware)", acc_prop),
              ("(b) Caching-assisted (agnostic)", acc_cach)]
    sc = None
    for ax, (title, acc) in zip(axes, panels):
        sc = ax.scatter(vm[:, 0], vm[:, 1], c=acc, cmap="RdYlGn", vmin=0.2,
                        vmax=1.0, s=42, edgecolors="k", linewidths=0.3, zorder=4)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        cx.add_basemap(ax, crs="EPSG:3857", source=src, zoom=13,
                       attribution_size=5)
        ax.set_title(f"{title}\nmean acc = {acc.mean():.3f}", y=-0.16,
                     fontsize=12)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Vehicle model accuracy")
    fig.suptitle("Encoder propagation over the real Seoul road network "
                 f"(V2X cohort, round k={snap_k})", fontsize=13, y=0.99)

    out = os.path.join(fig_dir, "fig_infocom_v2x_map.png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}  (Proposed {acc_prop.mean():.3f} vs "
          f"Caching {acc_cach.mean():.3f})")
    return out


if __name__ == "__main__":
    make_v2x_map_subfig()
