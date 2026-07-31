"""1x5 statistical-utility map (Learning-aware excluded) + CSV export.

Panels: FACE | Cached-DFL | V2V-aware | mmFedMC | AutoFed, all from the
fig_seoul_map_compact protocol (abstract q_eff backend, seed 2026, N=180,
snap_k=249) with the shared 0.2-1.0 RdYlGn colour scale.

Out:  Figures/fig_seoul_map_util_1x5.{png,pdf}
      newnewdata/seoul_map_utility_1x5.csv  (per-vehicle lon/lat, web-mercator
      x/y, heading, and the five per-vehicle utility columns)
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sim.v2x_map import _glyphs, _CAR_BODY, _CAR_CABIN

cache = np.load("results/v2x_map_cache.npz")
nb = np.load("results/v2x_map_cache_newbase.npz")
vm, ang = cache["vm"], cache["ang"]

PANELS = [
    ("FACE", cache["acc_Proposed"]),
    ("Cached-DFL", cache["acc_Caching-assisted"]),
    ("V2V-aware", cache["acc_V2V-aware"]),
    ("mmFedMC", nb["acc_mmFedMC"]),
    ("AutoFed", nb["acc_AutoFed"]),
]
VMIN, VMAX = 0.2, 1.0

# ---- CSV export (everything needed to redraw the figure) ----
from pyproj import Transformer
inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
lon, lat = inv.transform(vm[:, 0], vm[:, 1])
csv_path = "newnewdata/seoul_map_utility_1x5.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["vehicle", "lon", "lat", "x_web_mercator", "y_web_mercator",
                "heading_deg"] + [f"util_{n.replace('-', '')}"
                                  for n, _ in PANELS])
    for i in range(len(vm)):
        w.writerow([i, f"{lon[i]:.6f}", f"{lat[i]:.6f}",
                    f"{vm[i, 0]:.1f}", f"{vm[i, 1]:.1f}",
                    f"{np.degrees(ang[i]):.1f}"]
                   + [f"{float(v[i]):.4f}" for _, v in PANELS])
print("wrote", csv_path)

# ---- 1x5 figure ----
import contextily as cx
BASEMAP = os.environ.get("BASEMAP", "positron")   # positron | satellite
TILES = (cx.providers.Esri.WorldImagery if BASEMAP == "satellite"
         else cx.providers.CartoDB.Positron)
SUFFIX = "_sat" if BASEMAP == "satellite" else ""
cmap = plt.get_cmap("RdYlGn"); norm = Normalize(VMIN, VMAX)
pad = 400
xlim = (vm[:, 0].min() - pad, vm[:, 0].max() + pad)
ylim = (vm[:, 1].min() - pad, vm[:, 1].max() + pad)
car_len = (xlim[1] - xlim[0]) / 42.0

FIG_W = 7.16                       # IEEE \textwidth
pw = FIG_W / 5
ph = pw * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
fig, axes = plt.subplots(1, 5, figsize=(FIG_W, ph + 0.75),
                         sharex=True, sharey=True)
for ax, (name, vals) in zip(axes, PANELS):
    acc = np.asarray(vals)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
    cx.add_basemap(ax, crs="EPSG:3857",
                   source=TILES, zoom=15,
                   attribution=False if BASEMAP == "satellite" else None,
                   attribution_size=2)
    ax.scatter(vm[:, 0], vm[:, 1], c=acc, cmap=cmap, norm=norm,
               s=26, lw=0, alpha=0.20, zorder=3)
    ax.add_collection(PathCollection(
        _glyphs(_CAR_BODY, vm, ang, car_len), facecolors=cmap(norm(acc)),
        edgecolors="black", linewidths=0.25, zorder=4))
    ax.add_collection(PathCollection(
        _glyphs(_CAR_CABIN, vm, ang, car_len), facecolors="black",
        alpha=0.35, edgecolors="none", zorder=5))
    ax.text(0.0, 1.06, name, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5)
    ax.text(1.0, 1.06, f"{acc.mean():.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.0)
    for sp in ax.spines.values():
        if name == "FACE":
            sp.set_edgecolor("#1f77b4"); sp.set_linewidth(1.6)
        else:
            sp.set_edgecolor("0.45"); sp.set_linewidth(0.6)
fig.subplots_adjust(wspace=0.05)
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                    fraction=0.05, pad=0.035, aspect=70)
cbar.set_label("Achieved statistical utility (mean above each panel)",
               fontsize=7.5)
cbar.ax.tick_params(labelsize=7)
out = f"Figures/fig_seoul_map_util_1x5{SUFFIX}.png"
for ext in ("png", "pdf"):
    fig.savefig(out.replace(".png", "." + ext), dpi=260,
                bbox_inches="tight")
plt.close(fig)
print("saved", out, " ".join(f"{n}={np.asarray(v).mean():.3f}"
                             for n, v in PANELS))
