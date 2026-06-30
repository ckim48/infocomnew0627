"""
Real Seoul V2X mobility source (data.go.kr / Seoul T-data, dataset 15102656).

Pulls live V2X vehicle-status messages (terminal id + WGS84 position) from the
Seoul Traffic Big-Data platform and (a) renders the real vehicle cohort on a
georeferenced basemap, and (b) can poll over a window to accumulate a real
mobility trace (per-terminal trajectories) -- a real-data replacement for the
synthetic SUMO demand used in sim/gangnam_demo.py.

API: http://t-data.seoul.go.kr/apig/apiman-gateway/tapi/
     v2xVehiclesStatusInformation/1.0  (apiKey, type=json, pageNo, numOfRows)
Fields used: trmnId (terminal id), vhcleLot (lon), vhcleLat (lat), trsmTm.
"""

import os
import json
import time
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API = ("http://t-data.seoul.go.kr/apig/apiman-gateway/tapi/"
       "v2xVehiclesStatusInformation/1.0")
API_KEY = os.environ.get("SEOUL_V2X_KEY", "9fa41bfb-c473-4db6-aa5e-cc7b158936a1")

# Gangnam-area bounding box (matches data/gangnam OSM extract region, widened)
GANGNAM_BBOX = (127.001, 37.449, 127.075, 37.516)        # lon0, lat0, lon1, lat1
SEOUL_BBOX = (126.76, 37.42, 127.18, 37.70)


def _get(page, rows, timeout=30):
    url = f"{API}?apiKey={API_KEY}&type=json&pageNo={page}&numOfRows={rows}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch_vehicles(pages=2, rows=1000):
    """Current snapshot: dict trmnId -> (lon, lat). De-duplicated, latest wins."""
    veh = {}
    for p in range(1, pages + 1):
        try:
            rowset = _get(p, rows)
        except Exception as e:
            print(f"  [v2x] page {p} error: {type(e).__name__}")
            break
        for x in rowset:
            lo, la = x.get("vhcleLot"), x.get("vhcleLat")
            if lo is None or la is None:
                continue
            veh[x["trmnId"]] = (float(lo), float(la))
        if len(rowset) < rows:
            break
    return veh


def collect_trace(duration_s=120, interval_s=10, out=None, bbox=None):
    """Poll the live API over a window; build per-terminal trajectories.

    Saves npz: ids[str], pos[T, M, 2] (lon,lat; NaN where absent), times[T].
    Use as a real mobility trace (replacing the synthetic SUMO snapshot).
    """
    out = out or os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "gangnam", "seoul_v2x_trace.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def _save(snaps, stamps):
        ids = sorted({k for s in snaps for k in s})
        idx = {k: i for i, k in enumerate(ids)}
        pos = np.full((len(snaps), len(ids), 2), np.nan)
        for t, s in enumerate(snaps):
            for k, (lo, la) in s.items():
                pos[t, idx[k]] = (lo, la)
        np.savez_compressed(out, ids=np.array(ids), pos=pos,
                            times=np.array(stamps))
        return pos.shape

    snaps, stamps = [], []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        try:
            veh = fetch_vehicles()
        except Exception as e:
            print(f"  [v2x] fetch error {type(e).__name__}; retrying", flush=True)
            time.sleep(interval_s); continue
        if bbox:
            lo0, la0, lo1, la1 = bbox
            veh = {k: v for k, v in veh.items()
                   if lo0 <= v[0] <= lo1 and la0 <= v[1] <= la1}
        snaps.append(veh)
        stamps.append(time.time() - t0)
        shp = _save(snaps, stamps)                       # incremental, crash-safe
        print(f"  [v2x] t={stamps[-1]:5.0f}s  vehicles={len(veh)}  trace{shp}",
              flush=True)
        time.sleep(interval_s)
    print(f"  [v2x] done -> {out}", flush=True)
    return out


def make_v2x_overlay(region="gangnam", basemap="osm", with_sumo_net=True,
                     pages=2):
    """Render the live V2X vehicle cohort on a real basemap of Seoul/Gangnam."""
    import contextily as cx
    from pyproj import Transformer

    bbox = GANGNAM_BBOX if region == "gangnam" else SEOUL_BBOX
    veh = fetch_vehicles(pages=pages)
    lo0, la0, lo1, la1 = bbox
    pts = np.array([(lo, la) for lo, la in veh.values()
                    if lo0 <= lo <= lo1 and la0 <= la <= la1])
    print(f"  [v2x] {region}: {len(pts)} vehicles in view (of {len(veh)} citywide)")

    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    vm = np.array([tf.transform(lo, la) for lo, la in pts])

    fig, ax = plt.subplots(figsize=(12, 11))

    # optional: overlay our Gangnam SUMO road network for context
    if with_sumo_net and region == "gangnam":
        net_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "gangnam", "gangnam.net.xml")
        if os.path.exists(net_path):
            import sumolib
            from matplotlib.collections import LineCollection
            net = sumolib.net.readNet(net_path)
            segs = []
            for e in net.getEdges():
                if e.isSpecial():
                    continue
                shp = np.array(e.getShape())
                ll = [net.convertXY2LonLat(float(x), float(y)) for x, y in shp]
                segs.append(np.array([tf.transform(lo, la) for lo, la in ll]))
            ax.add_collection(LineCollection(segs, colors="#444444",
                                             linewidths=0.8, alpha=0.6, zorder=2))

    ax.scatter(vm[:, 0], vm[:, 1], s=42, c="#d62728", edgecolors="white",
               linewidths=0.5, zorder=4, label="Live V2X vehicle")

    pad = 200 if region == "gangnam" else 600
    ax.set_xlim(vm[:, 0].min() - pad, vm[:, 0].max() + pad)
    ax.set_ylim(vm[:, 1].min() - pad, vm[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    zoom = 15 if region == "gangnam" else 13
    cx.add_basemap(ax, crs="EPSG:3857",
                   source=providers.get(basemap, providers["osm"]),
                   zoom=zoom, attribution_size=6)
    ax.set_title(f"Live Seoul V2X vehicles on the real map ({region})",
                 fontsize=13)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    fig.tight_layout()

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f"fig_v2x_{region}_{basemap}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
    return out


def _panel(ax, tf, pts, bbox_pad, caption, basemap, zoom, net_segs=None):
    """Render one V2X map panel (vehicles + optional road net) on a basemap."""
    import contextily as cx
    from matplotlib.collections import LineCollection
    providers = {
        "osm": cx.providers.OpenStreetMap.Mapnik,
        "satellite": cx.providers.Esri.WorldImagery,
        "positron": cx.providers.CartoDB.Positron,
        "voyager": cx.providers.CartoDB.Voyager,
    }
    if net_segs is not None:
        ax.add_collection(LineCollection(net_segs, colors="#444444",
                                         linewidths=0.7, alpha=0.6, zorder=2))
    vm = np.array([tf.transform(lo, la) for lo, la in pts])
    ax.scatter(vm[:, 0], vm[:, 1], s=20, c="#d62728", edgecolors="white",
               linewidths=0.4, zorder=4)
    ax.set_xlim(vm[:, 0].min() - bbox_pad, vm[:, 0].max() + bbox_pad)
    ax.set_ylim(vm[:, 1].min() - bbox_pad, vm[:, 1].max() + bbox_pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cx.add_basemap(ax, crs="EPSG:3857",
                   source=providers.get(basemap, providers["osm"]),
                   zoom=zoom, attribution_size=5)
    ax.set_title(caption, y=-0.10, fontsize=12)


def make_v2x_subfig(left_basemap="osm", right_basemap="positron", pages=2,
                    out_name="fig_infocom_korea_v2x"):
    """INFOCOM-style two-panel subfig: (a) Gangnam zoom, (b) Seoul citywide,
    from a single live V2X snapshot. Mirrors sim/paper_figs.py layout."""
    from pyproj import Transformer
    import sumolib

    veh = fetch_vehicles(pages=pages)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    glo0, gla0, glo1, gla1 = GANGNAM_BBOX
    gpts = np.array([(lo, la) for lo, la in veh.values()
                     if glo0 <= lo <= glo1 and gla0 <= la <= gla1])
    spts = np.array([(lo, la) for lo, la in veh.values()])
    print(f"  [v2x] subfig: {len(gpts)} Gangnam / {len(spts)} Seoul vehicles")

    # Gangnam SUMO road network for the left panel
    net_segs = None
    net_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "gangnam", "gangnam.net.xml")
    if os.path.exists(net_path):
        net = sumolib.net.readNet(net_path)
        net_segs = []
        for e in net.getEdges():
            if e.isSpecial():
                continue
            shp = np.array(e.getShape())
            ll = [net.convertXY2LonLat(float(x), float(y)) for x, y in shp]
            net_segs.append(np.array([tf.transform(lo, la) for lo, la in ll]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    _panel(axes[0], tf, gpts, 200, "(a) Gangnam (zoom)", left_basemap, 15,
           net_segs=net_segs)
    _panel(axes[1], tf, spts, 600, "(b) Seoul (city-wide)", right_basemap, 12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, out_name + ".png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", "." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved", out)
    return out


if __name__ == "__main__":
    make_v2x_subfig()
