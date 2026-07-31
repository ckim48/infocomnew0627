"""Raw satellite basemap of the exact extent used by the 1x5 utility map
(no overlays). Out: Figures/fig_gangnam_satellite_raw.{png,pdf}"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as cx

vm = np.load("results/v2x_map_cache.npz")["vm"]
pad = 400
xlim = (vm[:, 0].min() - pad, vm[:, 0].max() + pad)
ylim = (vm[:, 1].min() - pad, vm[:, 1].max() + pad)

fig, ax = plt.subplots(figsize=(8, 8 * (ylim[1]-ylim[0]) / (xlim[1]-xlim[0])))
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.Esri.WorldImagery,
               zoom=16, attribution_size=4)
fig.tight_layout(pad=0.1)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_gangnam_satellite_raw.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_gangnam_satellite_raw.png")
