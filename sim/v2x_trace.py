"""
Build a MobilitySim-compatible trace from the real Seoul V2X feed.

Takes the live V2X trajectory log collected by sim.seoul_v2x.collect_trace
(per-terminal WGS84 positions over time) and a SUMO road network of Seoul
(imported from OpenStreetMap), snaps each vehicle to the real road segment it is
on each round, and emits the same trace dict that sim.intas_trace produces, so
the entire GAT + caching/forwarding pipeline runs unchanged on real Seoul
mobility instead of the Ingolstadt (InTAS) trace.
"""

import os
import numpy as np
import sumolib

from .intas_trace import build_road_graph_from_net

V2X_TRACE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "gangnam", "seoul_v2x_trace.npz")
SEOUL_NET = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "gangnam", "seoul.net.xml")


def _build_edge_kdtree(net, eid, step=10.0):
    """KDTree over points densely sampled along every road edge; each point maps
    back to its edge index. Fast nearest-edge snapping without an R-tree."""
    from scipy.spatial import cKDTree
    pts, owner = [], []
    for e in net.getEdges():
        if e.isSpecial():
            continue
        idx = eid[e.getID()]
        shp = np.array(e.getShape())
        for a, b in zip(shp[:-1], shp[1:]):
            seg = b - a
            L = np.hypot(*seg)
            n = max(int(L // step), 1)
            for t in np.linspace(0, 1, n + 1):
                pts.append(a + t * seg); owner.append(idx)
    return cKDTree(np.array(pts)), np.array(owner, dtype=np.int64)


def build_v2x_trace(cfg, cache=None, min_cov=0.9, net_file=SEOUL_NET,
                    v2x_file=V2X_TRACE, verbose=True):
    """Snap the real Seoul V2X log onto `net_file`; return an InTAS-style trace."""
    cache = cache or os.path.join(cfg.results_dir, "v2x_seoul_trace.npz")
    if os.path.exists(cache):
        if verbose:
            print(f"    [v2x-trace] loading cached {cache}")
        d = np.load(cache, allow_pickle=True)
        return {k: d[k].item() if d[k].dtype == object and d[k].shape == ()
                else d[k] for k in d.files}

    net = sumolib.net.readNet(net_file)
    road = build_road_graph_from_net(net)
    eid = road["eid"]
    (xmin, ymin), (xmax, ymax) = net.getBBoxXY()

    d = np.load(v2x_file, allow_pickle=True)
    pos, times = d["pos"], d["times"]                 # [T,M,2] lon/lat ; [T]
    T, M, _ = pos.shape
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 10.0

    # lon/lat -> net XY, only where inside the network bbox
    XY = np.full((T, M, 2), np.nan)
    inside = np.zeros((T, M), bool)
    for t in range(T):
        for j in range(M):
            lo, la = pos[t, j]
            if not np.isfinite(lo):
                continue
            x, y = net.convertLonLat2XY(float(lo), float(la))
            if xmin <= x <= xmax and ymin <= y <= ymax:
                XY[t, j] = (x, y); inside[t, j] = True

    cov = inside.sum(0) / T
    keep = np.where(cov >= min_cov)[0]
    if verbose:
        print(f"    [v2x-trace] {len(keep)} vehicles with coverage>={min_cov} "
              f"(of {M}); {T} rounds, dt~{dt:.0f}s")

    # subsample to a target cohort size if configured smaller
    rng = np.random.default_rng(cfg.seed)
    if cfg.num_vehicles and len(keep) > cfg.num_vehicles:
        keep = np.sort(rng.choice(keep, cfg.num_vehicles, replace=False))
    N = len(keep)

    # interpolate small gaps per vehicle (clamped at the ends) so every round
    # has a position; then snap each round to the real road segment.
    veh_xy = np.zeros((T, N, 2))
    for jj, j in enumerate(keep):
        valid = np.where(inside[:, j])[0]
        for c in (0, 1):
            veh_xy[:, jj, c] = np.interp(np.arange(T), valid, XY[valid, j, c])

    V = road["V"]
    if verbose:
        print("    [v2x-trace] building edge KDTree and snapping ...")
    tree, owner = _build_edge_kdtree(net, eid)
    flat = veh_xy.reshape(-1, 2)
    _, nn = tree.query(flat, k=1)                     # nearest sampled edge-point
    veh_seg = owner[nn].reshape(T, N).astype(np.int64)

    # speed from consecutive positions; per-edge density / mean speed
    veh_speed = np.zeros((T, N))
    veh_speed[1:] = np.linalg.norm(np.diff(veh_xy, axis=0), axis=2) / dt
    veh_speed[0] = veh_speed[1] if T > 1 else 0.0
    edge_count = np.zeros((T, V), dtype=np.float32)
    edge_spd = np.zeros((T, V), dtype=np.float32)
    for t in range(T):
        np.add.at(edge_count[t], veh_seg[t], 1.0)
        np.add.at(edge_spd[t], veh_seg[t], veh_speed[t])

    # centre coordinates for numerical stability (as in intas_trace)
    ctr = veh_xy.reshape(-1, 2).mean(0)
    veh_xy = veh_xy - ctr
    road["mid"] = road["mid"] - ctr

    trace = {
        **road,
        "veh_seg": veh_seg, "veh_xy": veh_xy, "veh_speed": veh_speed,
        "edge_count": edge_count, "edge_spd": edge_spd, "dt": dt,
        "ctr": ctr,                      # net-XY = veh_xy + ctr (for geo-mapping)
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **{k: (np.array(v, dtype=object) if isinstance(v, (list, dict))
                                      else v) for k, v in trace.items()})
    if verbose:
        print(f"    [v2x-trace] saved {cache}  (N={N}, K={T}, V={V})")
    return trace


if __name__ == "__main__":
    from .config import Config
    cfg = Config()
    cfg.num_vehicles = 180
    build_v2x_trace(cfg)
