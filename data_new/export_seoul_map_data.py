"""Export EVERYTHING needed to redraw fig_seoul_map(_realacc*) standalone
into data_new/: per-vehicle CSVs, the road-network polylines inside the
figure extent, the car-glyph geometry, and the figure metadata.

Run:  python3 data_new/export_seoul_map_data.py
Then: python3 data_new/plot_seoul_map.py --metric utility
"""
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

OUT = "data_new"
PAD = 400.0                       # figure margin around the cohort (m, EPSG:3857)


def _w(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  wrote {OUT}/{name}  ({len(rows)} rows)")


# 1) per-vehicle CSVs: identical to the plotted values (new_data exports)
for src in ("seoul_map_utility.csv", "seoul_map_realacc.csv"):
    shutil.copy(os.path.join("new_data", src), os.path.join(OUT, src))
    print(f"  copied new_data/{src} -> {OUT}/{src}")

# 2) figure extent from the shared position cache
cache = np.load("results/v2x_map_cache.npz")
vm = cache["vm"]
xlim = (vm[:, 0].min() - PAD, vm[:, 0].max() + PAD)
ylim = (vm[:, 1].min() - PAD, vm[:, 1].max() + PAD)

# 3) road polylines inside the extent (from the SUMO net used by the sim)
import sumolib
from pyproj import Transformer

net = sumolib.net.readNet("data/gangnam/seoul.net.xml")
inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
lo, la = inv.transform([xlim[0], xlim[1]], [ylim[0], ylim[1]])
(nx0, ny0) = net.convertLonLat2XY(float(lo[0]), float(la[0]))
(nx1, ny1) = net.convertLonLat2XY(float(lo[1]), float(la[1]))
nx0, nx1 = sorted((nx0, nx1)); ny0, ny1 = sorted((ny0, ny1))
M = 200.0                          # clip margin in net metres
rows = []
for e in net.getEdges():
    if e.isSpecial():
        continue
    shp = e.getShape()
    if not any(nx0 - M <= x <= nx1 + M and ny0 - M <= y <= ny1 + M
               for x, y in shp):
        continue
    eid, pr = e.getID(), e.getPriority()
    for seq, (x, y) in enumerate(shp):
        lon, lat = net.convertXY2LonLat(float(x), float(y))
        X, Y = fwd.transform(lon, lat)
        rows.append([eid, seq, pr, f"{lon:.6f}", f"{lat:.6f}",
                     f"{X:.1f}", f"{Y:.1f}"])
_w("seoul_roads.csv",
   ["edge_id", "seq", "priority", "lon", "lat",
    "x_web_mercator", "y_web_mercator"], rows)

# 4) car glyph (unit length, +x heading; scale by car_length_m, rotate by
#    heading_deg, translate to the vehicle position)
from sim.v2x_map import _CAR_BODY, _CAR_CABIN
rows = [["body", i, f"{x:.3f}", f"{y:.3f}"]
        for i, (x, y) in enumerate(_CAR_BODY)]
rows += [["cabin", i, f"{x:.3f}", f"{y:.3f}"]
         for i, (x, y) in enumerate(_CAR_CABIN)]
_w("seoul_car_glyph.csv", ["part", "seq", "x", "y"], rows)

# 5) figure metadata (everything else the drawing needs)
meta = [
    ("crs", "EPSG:3857 (Web Mercator); lon/lat are WGS84"),
    ("xlim_min", f"{xlim[0]:.1f}"), ("xlim_max", f"{xlim[1]:.1f}"),
    ("ylim_min", f"{ylim[0]:.1f}"), ("ylim_max", f"{ylim[1]:.1f}"),
    ("pad_m", f"{PAD:.0f}"),
    ("basemap_provider", "CartoDB Positron"),
    ("basemap_tile_url",
     "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"),
    ("basemap_zoom", "15"),
    ("panel_order", "FACE|Cached-DFL|V2V-aware|Learning-aware (2x2)"),
    ("colormap", "RdYlGn"),
    ("utility_vmin", "0.2"), ("utility_vmax", "1.0"),
    ("realacc_vmin", "0.05"), ("realacc_vmax", "0.95"),
    ("car_length_m", f"{(xlim[1] - xlim[0]) / 42.0:.1f}"),
    ("car_length_rule", "(xlim_max - xlim_min) / 42"),
    ("halo", "scatter s=80, alpha=0.20, same colormap"),
    ("car_body_style", "facecolor=cmap(norm(value)), edge black lw 0.4"),
    ("car_cabin_style", "facecolor black, alpha 0.35"),
    ("proposed_border", "#1f77b4 lw 2.2 (other panels: grey 0.45 lw 0.8)"),
    ("colorbar", "horizontal, fraction 0.045, pad 0.04, aspect 45"),
    ("figsize_in", "6.6 x 6.3"),
    ("utility_source", "results/v2x_map_cache.npz (abstract q_eff, seed 2026)"),
    ("realacc_source",
     "results/metrics_v2x_real_kitti_map400.npz (400-round real run)"),
]
_w("seoul_map_meta.csv", ["key", "value"], meta)
print("done")
