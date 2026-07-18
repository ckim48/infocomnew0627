"""
Paper map subfigure (2x2) on the real Seoul-Gangnam basemap: the SAME vehicle
cohort, drawn as heading-oriented car glyphs coloured by the model accuracy
each vehicle achieves under FACE vs three baselines. Greener = higher
accuracy; the broader green coverage under FACE shows strong encoders reach
more vehicles across the real road network.

Styling: clean CartoDB Positron tiles, top-view car silhouettes rotated to
each vehicle's instantaneous heading (conformal Mercator preserves angles),
soft accuracy-coloured halos, rounded chip labels, and an accent border on
the FACE panel.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import PathCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .config import Config
from .mobility import RoadNetwork, MobilitySim
from .hgat import train_hgat, future_contact_scores
from .v2x_trace import build_v2x_trace, SEOUL_NET
from .map_viz import _run_one
from .plotting import disp

MAP_SCHEMES = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]
# panel labels: full baseline names (paper style), regular weight
MAP_LABELS = {"Proposed": "FACE", "Caching-assisted": "Caching DFL",
              "V2V-aware": "V2V-aware", "Learning-aware": "Learning-aware"}

# top-view car silhouette pointing +x (unit length, ~0.45 width)
_CAR_BODY = np.array([
    (-0.50, -0.16), (-0.44, -0.225), (0.28, -0.225), (0.50, -0.12),
    (0.50, 0.12), (0.28, 0.225), (-0.44, 0.225), (-0.50, 0.16)])
_CAR_CABIN = np.array([
    (-0.20, -0.15), (0.10, -0.15), (0.16, 0.0), (0.10, 0.15), (-0.20, 0.15)])


def _glyphs(base, pos, ang, size):
    """Closed Paths of `base` rotated to `ang`, scaled, translated to `pos`."""
    out = []
    for (x, y), a in zip(pos, ang):
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        v = base @ R.T * size + (x, y)
        out.append(Path(v, closed=True))
    return out


def _compute(cfg, device, snap_k, cache):
    """Heavy step: trace + GAT + per-vehicle accuracy + geo positions +
    headings. Cached."""
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
    # instantaneous heading from the trace step (Mercator is conformal, so
    # the local-XY angle carries over to the map)
    prev = trace["veh_xy"][max(snap_k - 1, 0)]
    step = trace["veh_xy"][snap_k] - prev
    ang = np.arctan2(step[:, 1], step[:, 0])
    rng = np.random.default_rng(7)
    still = np.hypot(step[:, 0], step[:, 1]) < 1e-6
    ang[still] = rng.uniform(0, 2 * np.pi, int(still.sum()))

    np.savez(cache, vm=vm, ang=ang, snap_k=snap_k,
             **{f"acc_{s}": accs[s] for s in MAP_SCHEMES})
    return vm, ang, accs, snap_k


def make_v2x_map_subfig(cfg=None, device="cpu", num_vehicles=180, snap_k=None,
                        basemap="positron", use_cache=True):
    import contextily as cx

    cfg = cfg or Config()
    cfg.num_vehicles = num_vehicles
    fig_dir = cfg.figures_dir
    os.makedirs(fig_dir, exist_ok=True)
    cache = os.path.join(cfg.results_dir, "v2x_map_cache.npz")

    d = np.load(cache) if (use_cache and os.path.exists(cache)) else None
    if d is not None and "ang" in getattr(d, "files", []) \
            and all(f"acc_{s}" in d.files for s in MAP_SCHEMES):
        vm = d["vm"]; ang = d["ang"]; snap_k = int(d["snap_k"])
        accs = {s: d[f"acc_{s}"] for s in MAP_SCHEMES}
        print(f"  [v2x-map] re-plotting from cache {cache}")
    else:
        vm, ang, accs, snap_k = _compute(cfg, device, snap_k, cache)

    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    src = providers.get(basemap, providers["positron"])

    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=0.2, vmax=1.0)

    fig, axgrid = plt.subplots(2, 2, figsize=(6.6, 6.3),
                               sharex=True, sharey=True)
    axes = axgrid.ravel()
    pad = 400
    xlim = (vm[:, 0].min() - pad, vm[:, 0].max() + pad)
    ylim = (vm[:, 1].min() - pad, vm[:, 1].max() + pad)
    car_len = (xlim[1] - xlim[0]) / 42.0          # car glyph length (m)

    for ax, s in zip(axes, MAP_SCHEMES):
        acc = np.asarray(accs[s])
        colors = cmap(norm(acc))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        cx.add_basemap(ax, crs="EPSG:3857", source=src, zoom=15,
                       attribution_size=4)
        # soft accuracy halo under each car
        ax.scatter(vm[:, 0], vm[:, 1], c=acc, cmap=cmap, norm=norm,
                   s=80, lw=0, alpha=0.20, zorder=3)
        # car bodies, heading-oriented, coloured by accuracy
        bodies = PathCollection(_glyphs(_CAR_BODY, vm, ang, car_len),
                                facecolors=colors, edgecolors="black",
                                linewidths=0.4, zorder=4)
        ax.add_collection(bodies)
        # cabin / glasshouse overlay for the car look
        cabins = PathCollection(_glyphs(_CAR_CABIN, vm, ang, car_len),
                                facecolors="black", alpha=0.35,
                                edgecolors="none", zorder=5)
        ax.add_collection(cabins)
        # scheme name + cohort mean accuracy above the panel, off the map,
        # so the labels never occlude vehicles or roads
        name = MAP_LABELS.get(s, disp(s))
        ax.text(0.0, 1.02, f"{name}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10.5)
        ax.text(1.0, 1.02, f"mean acc {acc.mean():.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5)
        # accent border on the proposed scheme's panel
        if s == "Proposed":
            for sp in ax.spines.values():
                sp.set_edgecolor("#1f77b4"); sp.set_linewidth(2.2)
        else:
            for sp in ax.spines.values():
                sp.set_edgecolor("0.45"); sp.set_linewidth(0.8)

    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                        fraction=0.045, pad=0.04, aspect=45)
    cbar.set_label("Vehicle model accuracy")

    out = os.path.join(fig_dir, "fig_infocom_v2x_map.png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", "." + ext), dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved", out, " ".join(f"{disp(s)}={np.asarray(accs[s]).mean():.3f}"
                                   for s in MAP_SCHEMES))
    return out


if __name__ == "__main__":
    import sys
    make_v2x_map_subfig(basemap=sys.argv[1] if len(sys.argv) > 1 else "positron",
                        use_cache="fresh" not in sys.argv)
