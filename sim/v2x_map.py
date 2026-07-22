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
MAP_LABELS = {"Proposed": "FACE", "Caching-assisted": "Cached-DFL",
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
    # accuracy snapshot: end of the 250-round wrapped run (same protocol as
    # the main experiments); vehicle positions: last round of the raw trace
    rounds = 250
    snap_k = snap_k if snap_k is not None else rounds - 1
    pos_k = min(snap_k, mob.Krounds - 1)

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
        accs[s], _ = _run_one(cfg, mob, gammas, s, snap_k, rounds=rounds)

    mob.k = pos_k
    net = sumolib.net.readNet(SEOUL_NET)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    net_xy = mob.vehicle_xy() + ctr
    vm = np.empty((mob.N, 2))
    for i, (x, y) in enumerate(net_xy):
        lon, lat = net.convertXY2LonLat(float(x), float(y))
        vm[i] = tf.transform(lon, lat)
    # instantaneous heading from the trace step (Mercator is conformal, so
    # the local-XY angle carries over to the map)
    prev = trace["veh_xy"][max(pos_k - 1, 0)]
    step = trace["veh_xy"][pos_k] - prev
    ang = np.arctan2(step[:, 1], step[:, 0])
    rng = np.random.default_rng(7)
    still = np.hypot(step[:, 0], step[:, 1]) < 1e-6
    ang[still] = rng.uniform(0, 2 * np.pi, int(still.sum()))

    np.savez(cache, vm=vm, ang=ang, snap_k=snap_k,
             **{f"acc_{s}": accs[s] for s in MAP_SCHEMES})
    return vm, ang, accs, snap_k


def _draw_map(vm, ang, accs, out_name, fig_dir, basemap="positron",
              y_squeeze=1.0, vmin=0.2, vmax=1.0,
              cbar_label="Vehicle model accuracy", mean_label="mean acc",
              cmap_name="RdYlGn"):
    """Render the 2x2 accuracy-map panels from per-vehicle values `accs`
    (dict scheme -> array[N]); reused by the abstract-quality and the
    real-metric (test accuracy / LOO chi) variants."""
    import contextily as cx

    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    src = providers.get(basemap, providers["positron"])

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # two map rows are ~4.6 in of the 6.3 in height; shrink them by y_squeeze
    # (slack keeps the width constraint binding so panels stay wide, but is
    # capped -- the smaller squeezed-mode fonts absorb the residual narrowing)
    fig_h = 6.3 - 4.6 * (1.0 - y_squeeze) \
        + min(1.5 * (1.0 - y_squeeze), 0.25)
    fig, axgrid = plt.subplots(2, 2, figsize=(6.6, fig_h),
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
        ax.set_aspect(y_squeeze); ax.set_xticks([]); ax.set_yticks([])
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
        # strong squeezes narrow the panels; drop label sizes stepwise so
        # the longest title never collides with the mean-acc text
        tf, af = (10.5, 9.5) if y_squeeze >= 0.8 \
            else (9.5, 8.5) if y_squeeze >= 0.7 else (8.5, 7.5)
        ax.text(0.0, 1.045, f"{name}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=tf)
        ax.text(1.0, 1.045, f"{mean_label} {acc.mean():.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=af)
        # accent border on the proposed scheme's panel
        if s == "Proposed":
            for sp in ax.spines.values():
                sp.set_edgecolor("#1f77b4"); sp.set_linewidth(2.2)
        else:
            for sp in ax.spines.values():
                sp.set_edgecolor("0.45"); sp.set_linewidth(0.8)

    fig.subplots_adjust(wspace=0.04, hspace=0.20)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    # squeezed variants also slim the colorbar block to save height
    slim = y_squeeze < 0.8
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                        fraction=0.032 if slim else 0.045,
                        pad=0.02 if slim else 0.04, aspect=55 if slim else 45)
    cbar.set_label(cbar_label, fontsize=10 if slim else None)
    if slim:
        cbar.ax.tick_params(labelsize=9)

    out = os.path.join(fig_dir, out_name + ".png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", "." + ext), dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved", out, " ".join(f"{disp(s)}={np.asarray(accs[s]).mean():.3f}"
                                   for s in MAP_SCHEMES))
    return out


def make_v2x_map_subfig(cfg=None, device="cpu", num_vehicles=180, snap_k=None,
                        basemap="positron", use_cache=True, y_squeeze=1.0,
                        out_name="fig_infocom_v2x_map"):
    """Abstract-quality map (q_eff of the Seoul backend). y_squeeze < 1 draws
    map y-units compressed (shorter figure) under a separate out_name."""
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
    return _draw_map(vm, ang, accs, out_name, fig_dir, basemap=basemap,
                     y_squeeze=y_squeeze,
                     cbar_label="Achieved encoder quality",
                     mean_label="mean quality")


def make_v2x_map_real(dataset="kitti", metric="accveh", out_name=None,
                      basemap="positron", y_squeeze=1.0, npz=None,
                      cmap_name="RdYlGn", vmin=None, vmax=None):
    """Map painted with MEASURED per-vehicle values from the real-FL run
    (same protocol as the main table): metric='accveh' uses final test
    accuracy; 'chiveh' uses the leave-one-out encoder-contribution chi.
    Positions/headings come from the shared v2x_map_cache."""
    cfg = Config()
    fig_dir = cfg.figures_dir
    cache = np.load(os.path.join(cfg.results_dir, "v2x_map_cache.npz"))
    vm, ang = cache["vm"], cache["ang"]
    path = npz or os.path.join(cfg.results_dir,
                               f"metrics_v2x_real_{dataset}.npz")
    d = np.load(path)
    vals = {}
    for s in MAP_SCHEMES:
        a = d[f"{s}__{metric}_all"]          # seeds x N
        vals[s] = np.asarray(a, dtype=float).mean(0)
    allv = np.concatenate(list(vals.values()))
    # colorbar spans the empirical range (rounded to 0.05) for contrast
    if vmin is None:
        vmin = np.floor(np.nanpercentile(allv, 2) * 20) / 20
    if vmax is None:
        vmax = np.ceil(np.nanpercentile(allv, 98) * 20) / 20
    labels = {"accveh": "Per-vehicle test accuracy (real data)",
              "chiveh": "LOO encoder contribution $\\chi$ (real data)"}
    mean_labels = {"accveh": "mean acc", "chiveh": "mean $\\chi$"}
    out_name = out_name or f"fig_seoul_map_real{'' if metric=='accveh' else '_chi'}"
    return _draw_map(vm, ang, vals, out_name, fig_dir, basemap=basemap,
                     y_squeeze=y_squeeze, vmin=float(vmin), vmax=float(vmax),
                     cbar_label=labels[metric],
                     mean_label=mean_labels[metric], cmap_name=cmap_name)


if __name__ == "__main__":
    import sys
    make_v2x_map_subfig(basemap=sys.argv[1] if len(sys.argv) > 1 else "positron",
                        use_cache="fresh" not in sys.argv)
