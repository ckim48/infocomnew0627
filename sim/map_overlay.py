"""
Real-environment map with real camera images overlaid at their GPS positions.

Unlike the InTAS abstract road graph (sim/map_viz.py), this renders the genuine
nuScenes semantic map raster as the background and overlays the actual on-board
CAM_FRONT photographs at the ego-vehicle world coordinates where they were
captured. The result shows, on the true map, what each vehicle "sees" along its
drive -- a concrete picture of the multimodal sensor data our FL operates on.

nuScenes: H. Caesar et al., CVPR 2020. Map raster is 0.1 m/pixel; ego_pose
translations are world metres in the same frame.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

Image.MAX_IMAGE_PIXELS = None                      # nuScenes maps are huge rasters

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nuscenes")
MINI = os.path.join(ROOT, "v1.0-mini")
RES = 0.1                                           # metres per map pixel


def _load(name):
    with open(os.path.join(MINI, name)) as f:
        return json.load(f)


def _scene_track(scene_name):
    """Return (map_png, key-frame list) for a scene.

    Each key frame: dict(xy=[x,y] world metres, cam=abs CAM_FRONT image path).
    """
    scenes = {s["name"]: s for s in _load("scene.json")}
    if scene_name not in scenes:
        raise ValueError(f"scene {scene_name} not in {sorted(scenes)}")
    scene = scenes[scene_name]
    log_tok = scene["log_token"]
    location = {l["token"]: l["location"] for l in _load("log.json")}[log_tok]

    map_png = None
    for m in _load("map.json"):
        if log_tok in m["log_tokens"]:
            map_png = os.path.join(ROOT, m["filename"])
            break

    ego = {e["token"]: e["translation"] for e in _load("ego_pose.json")}
    sdata = _load("sample_data.json")
    cam_by_sample = {d["sample_token"]: d for d in sdata
                     if "/CAM_FRONT/" in d["filename"] and d["is_key_frame"]}

    # walk the sample linked list in capture order
    samples = {s["token"]: s for s in _load("sample.json")}
    track, tok = [], scene["first_sample_token"]
    while tok:
        d = cam_by_sample.get(tok)
        if d is not None:
            x, y, _ = ego[d["ego_pose_token"]]
            track.append({"xy": np.array([x, y]),
                          "cam": os.path.join(ROOT, d["filename"])})
        tok = samples[tok]["next"]
    return map_png, track, location


def make_overlay(scene_name="scene-0061", n_imgs=6, thumb_px=150,
                 figures_dir=None, pad_m=60.0):
    figures_dir = figures_dir or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    map_png, track, location = _scene_track(scene_name)
    xy = np.array([t["xy"] for t in track])                  # [T,2] world metres
    print(f"  [overlay] {scene_name} @ {location}: {len(track)} key frames")

    # viewport in metres around the driven trajectory
    x0, y0 = xy.min(0) - pad_m
    x1, y1 = xy.max(0) + pad_m

    # crop the semantic-map raster to that window (pixels = metres / RES)
    full = Image.open(map_png)
    W, H = full.size
    px0, px1 = int(x0 / RES), int(x1 / RES)
    # map image row 0 is the TOP, world-y grows upward -> flip rows
    py_top = H - int(y1 / RES)
    py_bot = H - int(y0 / RES)
    px0, px1 = max(0, px0), min(W, px1)
    py_top, py_bot = max(0, py_top), min(H, py_bot)
    crop = np.asarray(full.crop((px0, py_top, px1, py_bot)).convert("L"))

    extent = [px0 * RES, px1 * RES, (H - py_bot) * RES, (H - py_top) * RES]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, extent=extent,
              origin="upper", aspect="equal", alpha=0.55, zorder=0)

    # driven trajectory
    ax.plot(xy[:, 0], xy[:, 1], "-", color="#1f4e79",
            lw=2.4, alpha=0.9, zorder=2, label="Ego trajectory")
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#1f4e79", zorder=2)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=120, marker="o", c="white",
               edgecolors="#1f4e79", linewidths=2, zorder=3, label="Start")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=160, marker="*", c="#d62728",
               edgecolors="k", linewidths=0.6, zorder=3, label="End")

    # overlay real CAM_FRONT photos, evenly spaced along the drive
    idxs = np.linspace(0, len(track) - 1, n_imgs).round().astype(int)
    span = max(x1 - x0, y1 - y0)
    for j, i in enumerate(idxs):
        im = Image.open(track[i]["cam"]).convert("RGB")
        im.thumbnail((thumb_px, thumb_px))
        zoom = 1.0
        oi = OffsetImage(np.asarray(im), zoom=zoom)
        # fan the callouts around the trajectory so they don't overlap
        ang = 2 * np.pi * j / n_imgs
        off = 0.34 * span
        bx = track[i]["xy"][0] + off * np.cos(ang)
        by = track[i]["xy"][1] + off * np.sin(ang)
        ab = AnnotationBbox(
            oi, track[i]["xy"], xybox=(bx, by), xycoords="data",
            boxcoords="data", frameon=True, pad=0.25,
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.3,
                            connectionstyle="arc3,rad=0.0"),
            bboxprops=dict(edgecolor="#d62728", lw=1.6))
        ab.set_zorder(5)
        ax.add_artist(ab)
        ax.scatter([track[i]["xy"][0]], [track[i]["xy"][1]], s=55, marker="o",
                   c="#d62728", edgecolors="white", linewidths=1.0, zorder=4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(f"Real CAM_FRONT imagery overlaid on the nuScenes map "
                 f"({location}, {scene_name})", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    fig.tight_layout()

    out = os.path.join(figures_dir, f"fig_map_overlay_{scene_name}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
    return out


# ---------------------------------------------------------------------------
# Our simulated environment: real InTAS (Ingolstadt) map with the real camera
# frames each vehicle holds in the simulation, overlaid at its world position.
# ---------------------------------------------------------------------------

KITTI_IMG = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "kitti", "training", "image_2")


def _load_kitti_balanced(cfg, seed):
    """Mirror real_fl._prep_data (KITTI) but keep the KITTI frame id per sample,
    so a held sample index can be mapped back to its source image_2 frame."""
    from .kitti_dataset import build as build_kitti
    from .multimodal_model import NCLS
    img, lid, y, frame, boxh = build_kitti(cache="results/kitti_mm_all.npz")
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=NCLS)
    use_classes = [c for c in range(NCLS) if counts[c] >= 1]
    cap = int(min(counts[c] for c in use_classes))
    keep = []
    for c in use_classes:
        ci = np.where(y == c)[0]
        keep.append(rng.choice(ci, min(cap, len(ci)), replace=False))
    keep = rng.permutation(np.concatenate(keep))
    y = y[keep]; frame = frame[keep]
    n = len(y); perm = rng.permutation(n)
    n_test = int(0.20 * n); n_val = int(0.12 * n)
    train_idx = perm[n_test + n_val:]
    return frame, train_idx, y


def _assign_vehicles(cfg, seed, train_idx):
    """Reproduce RealMFL's rich/poor data partition (which samples each vehicle
    holds), identical to sim/real_fl.py so the overlay reflects the real run."""
    rng = np.random.default_rng(seed)
    N = cfg.num_vehicles
    riches = [rng.random() < cfg.frac_good for _ in range(N)]
    n_rich = max(sum(riches), 1)
    pool = rng.permutation(train_idx)
    poor_sizes = {i: int(rng.integers(4, 12)) for i in range(N) if not riches[i]}
    rich_budget = len(pool) - sum(poor_sizes.values())
    rich_each = max(rich_budget // n_rich, 30)
    local, cur = {}, 0
    for i in range(N):
        step = rich_each if riches[i] else poor_sizes[i]
        e = min(cur + step, len(pool))
        local[i] = pool[cur:e]; cur = e
        if len(local[i]) == 0:
            local[i] = pool[:5]
    return riches, local


def make_intas_overlay(cfg=None, device="cpu", snap_k=None,
                       n_rich_show=5, n_poor_show=3, thumb_px=170):
    """Overlay the real KITTI camera frames vehicles carry onto the genuine
    InTAS (Ingolstadt) map our simulation runs on, at their world positions."""
    from .config import Config
    from .intas_trace import get_or_build_trace
    from .mobility import RoadNetwork, MobilitySim
    from .map_viz import _net_segs_centred
    from matplotlib.collections import LineCollection

    cfg = cfg or Config()
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    snap_k = snap_k if snap_k is not None else cfg.K // 2

    cache_path = os.path.join(cfg.results_dir,
                              f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache_path, begin=34050.0, dt=2.0, warmup_s=480.0)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    net_segs, _ = _net_segs_centred(road)

    frame, train_idx, _ = _load_kitti_balanced(cfg, cfg.seed)
    riches, local = _assign_vehicles(cfg, cfg.seed, train_idx)
    rich_ids = np.array([i for i in range(mob.N) if riches[i]])
    poor_ids = np.array([i for i in range(mob.N) if not riches[i]])

    mob.k = snap_k
    xy = mob.vehicle_xy()                                 # [N,2] world metres
    dens = mob.density()                                  # per-segment traffic density

    # viewport over the cohort at this round (keep segment indices for the heat)
    pad = 350
    x0, x1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad
    y0, y1 = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    seg_in, seg_d = [], []
    for k, s in enumerate(net_segs):
        mx, my = s[:, 0].mean(), s[:, 1].mean()
        if x0 <= mx <= x1 and y0 <= my <= y1:
            seg_in.append(s)
            seg_d.append(dens[k])
    seg_d = np.array(seg_d)
    # robust normalisation for the traffic heat (cap at 90th pct of busy segments)
    hi = np.quantile(seg_d[seg_d > 0], 0.90) if (seg_d > 0).any() else 1.0
    seg_dn = np.clip(seg_d / (hi + 1e-9), 0, 1)
    busy = seg_dn > 0.05

    fig, ax = plt.subplots(figsize=(13, 9))
    # darker base road network
    ax.add_collection(LineCollection(seg_in, colors="0.45", linewidths=1.0,
                                     alpha=0.95, zorder=0))
    # traffic-density heat overlay on the busy segments
    if busy.any():
        heat = [seg_in[k] for k in np.where(busy)[0]]
        lc = LineCollection(heat, cmap="YlOrRd", linewidths=3.0, alpha=0.95, zorder=1)
        lc.set_array(seg_dn[busy])
        lc.set_clim(0, 1)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Road traffic density")
        cbar.set_ticks([0, 1]); cbar.set_ticklabels(["low", "high"])
    # the whole vehicle cohort
    ax.scatter(xy[poor_ids, 0], xy[poor_ids, 1], s=22, c="0.55", zorder=2,
               label="Vehicle (few/poor camera data)")
    ax.scatter(xy[rich_ids, 0], xy[rich_ids, 1], s=120, marker="*",
               c="#1f77b4", edgecolors="k", linewidths=0.5, zorder=3,
               label="Strong-encoder vehicle (rich camera data)")

    # choose vehicles spread across the viewport for legible callouts
    def _spread(ids, n):
        if len(ids) <= n:
            return list(ids)
        order = np.argsort(xy[ids, 0])                    # spread by x
        pick = order[np.linspace(0, len(ids) - 1, n).round().astype(int)]
        return list(ids[pick])

    show = [(i, True) for i in _spread(rich_ids, n_rich_show)] \
        + [(i, False) for i in _spread(poor_ids, n_poor_show)]

    span = max(x1 - x0, y1 - y0)
    for j, (i, is_rich) in enumerate(show):
        if len(local[i]) == 0:
            continue
        fr = int(frame[local[i][0]])
        png = os.path.join(KITTI_IMG, f"{fr:06d}.png")
        if not os.path.exists(png):
            continue
        im = Image.open(png).convert("RGB")
        im.thumbnail((thumb_px, thumb_px))
        oi = OffsetImage(np.asarray(im), zoom=1.0)
        ang = 2 * np.pi * j / len(show)
        off = 0.30 * span
        box = (xy[i, 0] + off * np.cos(ang), xy[i, 1] + off * np.sin(ang))
        ec = "#1f77b4" if is_rich else "#7f7f7f"
        ab = AnnotationBbox(
            oi, (xy[i, 0], xy[i, 1]), xybox=box, xycoords="data",
            boxcoords="data", frameon=True, pad=0.2,
            arrowprops=dict(arrowstyle="-", color=ec, lw=1.2),
            bboxprops=dict(edgecolor=ec, lw=1.8))
        ab.set_zorder(5)
        ax.add_artist(ab)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Real camera frames held by vehicles, overlaid on the simulated "
                 f"InTAS (Ingolstadt) map — round k={snap_k}", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    fig.tight_layout()
    out = os.path.join(figures_dir, "fig_intas_overlay.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
    return out


# ---------------------------------------------------------------------------
# Google-Maps-like overlay: the InTAS road network, vehicle cohort and traffic
# heat georeferenced onto a real OSM / satellite basemap of Ingolstadt.
# ---------------------------------------------------------------------------

NET_FILE_GEO = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "InTAS", "scenario", "ingolstadt.net.xml")


def _to_mercator(net, transformer, raw_xy):
    """Raw SUMO network metres -> Web-Mercator (EPSG:3857) metres."""
    out = np.empty_like(raw_xy, dtype=np.float64)
    for i, (x, y) in enumerate(raw_xy):
        lon, lat = net.convertXY2LonLat(float(x), float(y))
        out[i] = transformer.transform(lon, lat)
    return out


def make_intas_geo_overlay(cfg=None, snap_k=None, basemap="osm"):
    """Overlay the simulated InTAS network + vehicles + traffic heat on a real
    georeferenced basemap (Google-Maps-like), using the net's UTM projection."""
    import sumolib
    import contextily as cx
    from pyproj import Transformer
    from matplotlib.collections import LineCollection
    from .config import Config
    from .intas_trace import get_or_build_trace
    from .mobility import RoadNetwork, MobilitySim
    from .map_viz import _edge_polylines

    cfg = cfg or Config()
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    snap_k = snap_k if snap_k is not None else cfg.K // 2

    cache_path = os.path.join(cfg.results_dir,
                              f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache_path, begin=34050.0, dt=2.0, warmup_s=480.0)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)

    # raw SUMO coords <-> trace (centred) frame
    net = sumolib.net.readNet(NET_FILE_GEO)
    segs_raw, raw_mid0 = _edge_polylines(np.zeros(2))     # raw network polylines
    ctr = raw_mid0 - road.mid[0]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    mob.k = snap_k
    xy = mob.vehicle_xy()                                 # trace frame
    raw_xy = xy + ctr                                     # back to raw network metres
    veh_m = _to_mercator(net, transformer, raw_xy)        # vehicles in mercator
    dens = mob.density()

    # vehicle assignment (rich/poor) for marker styling — no images shown
    frame, train_idx, _ = _load_kitti_balanced(cfg, cfg.seed)
    riches, _ = _assign_vehicles(cfg, cfg.seed, train_idx)
    rich_ids = np.array([i for i in range(mob.N) if riches[i]])
    poor_ids = np.array([i for i in range(mob.N) if not riches[i]])

    # viewport (mercator) around the cohort, with padding
    pad = 350.0 * 1.55                                    # ~metres -> mercator scale near 48.8N
    bx0, by0 = veh_m.min(0) - pad
    bx1, by1 = veh_m.max(0) + pad

    # road segments whose raw midpoint is in the cohort's raw bbox, then project
    rx0, ry0 = raw_xy.min(0) - 350; rx1, ry1 = raw_xy.max(0) + 350
    seg_m, seg_d = [], []
    for k, s in enumerate(segs_raw):
        mx, my = s[:, 0].mean(), s[:, 1].mean()
        if rx0 <= mx <= rx1 and ry0 <= my <= ry1:
            seg_m.append(_to_mercator(net, transformer, s))
            seg_d.append(dens[k])
    seg_d = np.array(seg_d)
    hi = np.quantile(seg_d[seg_d > 0], 0.90) if (seg_d > 0).any() else 1.0
    seg_dn = np.clip(seg_d / (hi + 1e-9), 0, 1)
    busy = seg_dn > 0.05

    fig, ax = plt.subplots(figsize=(13, 10))
    # road network (drawn dark over the basemap)
    ax.add_collection(LineCollection(seg_m, colors="#222222", linewidths=1.2,
                                     alpha=0.85, zorder=2))
    # traffic-density heat
    if busy.any():
        heat = [seg_m[k] for k in np.where(busy)[0]]
        lc = LineCollection(heat, cmap="YlOrRd", linewidths=3.4, alpha=0.95, zorder=3)
        lc.set_array(seg_dn[busy]); lc.set_clim(0, 1)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Road traffic density")
        cbar.set_ticks([0, 1]); cbar.set_ticklabels(["low", "high"])
    # vehicles
    ax.scatter(veh_m[poor_ids, 0], veh_m[poor_ids, 1], s=26, c="#3a3a3a",
               edgecolors="white", linewidths=0.4, zorder=4,
               label="Vehicle")
    ax.scatter(veh_m[rich_ids, 0], veh_m[rich_ids, 1], s=150, marker="*",
               c="#1f77b4", edgecolors="white", linewidths=0.6, zorder=5,
               label="Strong-encoder vehicle")

    ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # real basemap tiles under everything
    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
        "satellite": cx.providers.Esri.WorldImagery,
    }
    src = providers.get(basemap, cx.providers.OpenStreetMap.Mapnik)
    cx.add_basemap(ax, crs="EPSG:3857", source=src, zoom=15, attribution_size=6)

    ax.set_title("Simulated InTAS vehicles & traffic density on the real "
                 f"Ingolstadt map — round k={snap_k}", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    fig.tight_layout()
    tag = basemap
    out = os.path.join(figures_dir, f"fig_intas_geo_{tag}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
    return out


if __name__ == "__main__":
    make_intas_geo_overlay()
