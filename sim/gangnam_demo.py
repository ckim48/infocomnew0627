"""
Proof-of-concept: run our simulated environment on a Korean map (Gangnam, Seoul).

Builds a vehicle snapshot from a SUMO microsimulation over the real Gangnam road
network (imported from OpenStreetMap, UTM zone 52N) and overlays the road graph,
the vehicle cohort and per-road traffic density onto a real georeferenced
basemap (OSM streets / satellite), exactly like sim/map_overlay for Ingolstadt.

This demonstrates that the mobility/map layer of the pipeline transfers to
Gangnam; the multimodal-data layer would be supplied by a Korean sensor dataset
(e.g. AI-Hub 71784) via a loader mirroring sim/kitti_dataset.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

GANGNAM = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gangnam")
NET = os.path.join(GANGNAM, "gangnam.net.xml")
ROU = os.path.join(GANGNAM, "gangnam.rou.xml")


def _snapshot(warmup_s=300.0, net=NET, route=ROU, comm_range=150.0, seed=42):
    """Run SUMO to warmup_s and return (vehicle xy [N,2], per-edge density, net)."""
    import libsumo as traci
    import sumolib
    sumo_net = sumolib.net.readNet(net)
    traci.start(["sumo", "-n", net, "-r", route, "--no-warnings", "true",
                 "--seed", str(seed), "--step-length", "1.0"])
    step = 0
    while step < warmup_s:
        traci.simulationStep()
        step += 1
        if traci.vehicle.getIDCount() > 250:           # enough cars on screen
            break
    ids = traci.vehicle.getIDList()
    xy = np.array([traci.vehicle.getPosition(v) for v in ids], dtype=np.float64)
    edge_of = [traci.vehicle.getRoadID(v) for v in ids]
    # per-edge vehicle count -> density (veh per 100 m)
    counts = {}
    for e in edge_of:
        counts[e] = counts.get(e, 0) + 1
    traci.close()
    return xy, counts, sumo_net


def make_gangnam_overlay(basemap="osm", warmup_s=300.0, frac_strong=0.15, seed=42):
    import contextily as cx
    from pyproj import Transformer

    xy, counts, net = _snapshot(warmup_s=warmup_s, seed=seed)
    rng = np.random.default_rng(seed)
    N = len(xy)
    print(f"  [gangnam] snapshot: {N} vehicles")

    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def to_merc(pts):
        out = np.empty((len(pts), 2))
        for i, (x, y) in enumerate(pts):
            lon, lat = net.convertXY2LonLat(float(x), float(y))
            out[i] = tf.transform(lon, lat)
        return out

    veh_m = to_merc(xy)

    # road polylines + density in mercator
    edges = [e for e in net.getEdges() if not e.isSpecial()]
    seg_m, seg_d = [], []
    for e in edges:
        shp = np.array(e.getShape())
        seg_m.append(to_merc(shp))
        L = max(e.getLength(), 1.0)
        seg_d.append(counts.get(e.getID(), 0) / L * 100.0)
    seg_d = np.array(seg_d)
    hi = np.quantile(seg_d[seg_d > 0], 0.90) if (seg_d > 0).any() else 1.0
    seg_dn = np.clip(seg_d / (hi + 1e-9), 0, 1)
    busy = seg_dn > 0.05

    strong = rng.random(N) < frac_strong               # illustrative encoder roles

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.add_collection(LineCollection(seg_m, colors="#222222", linewidths=1.1,
                                     alpha=0.85, zorder=2))
    if busy.any():
        heat = [seg_m[k] for k in np.where(busy)[0]]
        lc = LineCollection(heat, cmap="YlOrRd", linewidths=3.2, alpha=0.95, zorder=3)
        lc.set_array(seg_dn[busy]); lc.set_clim(0, 1)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Road traffic density")
        cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])

    ax.scatter(veh_m[~strong, 0], veh_m[~strong, 1], s=26, c="#3a3a3a",
               edgecolors="white", linewidths=0.4, zorder=4, label="Vehicle")
    ax.scatter(veh_m[strong, 0], veh_m[strong, 1], s=150, marker="*",
               c="#1f77b4", edgecolors="white", linewidths=0.6, zorder=5,
               label="Strong-encoder vehicle")

    # tight viewport around the cohort
    pad = 250
    ax.set_xlim(veh_m[:, 0].min() - pad, veh_m[:, 0].max() + pad)
    ax.set_ylim(veh_m[:, 1].min() - pad, veh_m[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    cx.add_basemap(ax, crs="EPSG:3857",
                   source=providers.get(basemap, providers["osm"]),
                   zoom=16, attribution_size=6)
    ax.set_title("Simulated vehicles & traffic density on the real Gangnam "
                 "(Seoul) map", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    fig.tight_layout()

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"fig_gangnam_{basemap}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
    return out


if __name__ == "__main__":
    make_gangnam_overlay("osm")
