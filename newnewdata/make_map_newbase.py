"""mmFedMC / AutoFed accuracy maps in the exact fig_seoul_map_realacc3 style.

Sources (all same protocol, KITTI, 3 seeds, T=250, N=180):
- old 4 schemes: mean of results/metrics_v2x_real_kitti_map{,_s2027,_s2028}.npz
  (this reproduces the realacc3 panel means exactly: 0.615/0.522/0.526/0.585)
- mmFedMC/AutoFed: results/metrics_v2x_real_kitti_map_newbase.npz (merged 3-seed)

Out:  Figures/fig_seoul_map_newbase.{png,pdf}   (1x2: mmFedMC | AutoFed)
      Figures/fig_seoul_map_realacc6.{png,pdf}  (3x2: all six Table-I schemes)
"""
import os, sys

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
vm, ang = cache["vm"], cache["ang"]

_old = [np.load(f"results/metrics_v2x_real_kitti_map{sfx}.npz")
        for sfx in ("", "_s2027", "_s2028")]
_new = np.load("results/metrics_v2x_real_kitti_map_newbase.npz")


def _acc(scheme, src):
    if src == "old":
        return np.mean([d[f"{scheme}__accveh_all"].mean(0) for d in _old], 0)
    return _new[f"{scheme}__accveh_all"].mean(0)


VALS = {
    "FACE": _acc("Proposed", "old"),
    "Cached-DFL": _acc("Caching-assisted", "old"),
    "V2V-aware": _acc("V2V-aware", "old"),
    "Learning-aware": _acc("Learning-aware", "old"),
    "mmFedMC": _acc("mmFedMC", "new"),
    "AutoFed": _acc("AutoFed", "new"),
}

VMIN, VMAX = 0.05, 0.95   # realacc3 colour scale (seoul_map_meta.csv)


def draw(panels, nrows, ncols, out_name, slim=False, vmin=VMIN, vmax=VMAX,
         cbar_label="Per-vehicle test accuracy (real data, 3 seeds)",
         mean_label="mean acc"):
    import contextily as cx
    cmap = plt.get_cmap("RdYlGn"); norm = Normalize(vmin, vmax)
    pad = 400
    xlim = (vm[:, 0].min() - pad, vm[:, 0].max() + pad)
    ylim = (vm[:, 1].min() - pad, vm[:, 1].max() + pad)
    car_len = (xlim[1] - xlim[0]) / 42.0
    pw = 6.6 / ncols
    ph = pw * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig_h = nrows * (ph + 0.35) + (0.9 if not slim else 0.75)
    fig, axgrid = plt.subplots(nrows, ncols, figsize=(6.6, fig_h),
                               sharex=True, sharey=True, squeeze=False)
    axes = axgrid.ravel()
    for ax, name in zip(axes, panels):
        acc = np.asarray(VALS[name])
        colors = cmap(norm(acc))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.Positron, zoom=15,
                       attribution_size=4)
        ax.scatter(vm[:, 0], vm[:, 1], c=acc, cmap=cmap, norm=norm,
                   s=80, lw=0, alpha=0.20, zorder=3)
        ax.add_collection(PathCollection(
            _glyphs(_CAR_BODY, vm, ang, car_len), facecolors=colors,
            edgecolors="black", linewidths=0.4, zorder=4))
        ax.add_collection(PathCollection(
            _glyphs(_CAR_CABIN, vm, ang, car_len), facecolors="black",
            alpha=0.35, edgecolors="none", zorder=5))
        ax.text(0.0, 1.045, name, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10.5)
        ax.text(1.0, 1.045, f"{mean_label} {acc.mean():.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5)
        for sp in ax.spines.values():
            if name == "FACE":
                sp.set_edgecolor("#1f77b4"); sp.set_linewidth(2.2)
            else:
                sp.set_edgecolor("0.45"); sp.set_linewidth(0.8)
    fig.subplots_adjust(wspace=0.04, hspace=0.20)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                        fraction=0.045 if nrows > 1 else 0.06,
                        pad=0.04, aspect=45)
    cbar.set_label(cbar_label)
    out = os.path.join("Figures", out_name + ".png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", "." + ext), dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved", out,
          " ".join(f"{n}={VALS[n].mean():.3f}" for n in panels))


if __name__ == "__main__":
    # 1x2 accuracy variant kept under _realacc; the plain fig_seoul_map_newbase
    # name is owned by the utility-metric version (make_map_newbase_util.py)
    draw(["mmFedMC", "AutoFed"], 1, 2, "fig_seoul_map_newbase_realacc",
         slim=True)
    draw(["FACE", "Cached-DFL", "V2V-aware", "Learning-aware",
          "mmFedMC", "AutoFed"], 3, 2, "fig_seoul_map_realacc6")
