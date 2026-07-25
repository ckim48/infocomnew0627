"""Standalone redraw of fig_seoul_map from the data_new/ CSVs ONLY
(no simulator imports; works offline -- roads are drawn from
seoul_roads.csv; pass --tiles to use CartoDB Positron via contextily
instead, which reproduces the paper background exactly).

  python3 data_new/plot_seoul_map.py --metric utility
  python3 data_new/plot_seoul_map.py --metric realacc --tiles
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.colors import Normalize
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def read_csv(name):
    with open(os.path.join(HERE, name), newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=("utility", "realacc"),
                    default="utility")
    ap.add_argument("--tiles", action="store_true",
                    help="use CartoDB Positron tiles (needs contextily)")
    args = ap.parse_args()

    meta = {r["key"]: r["value"] for r in read_csv("seoul_map_meta.csv")}
    veh = read_csv(f"seoul_map_{args.metric}.csv")
    pre = "utility_" if args.metric == "utility" else "acc_"
    disp = {"FACE": "FACE", "CachedDFL": "Cached-DFL",
            "V2V": "V2V-aware", "LearningAware": "Learning-aware"}
    panels = [(disp.get(c[len(pre):], c[len(pre):]), c) for c in veh[0]
              if c.startswith(pre)]                      # (label, column)
    vm = np.array([[float(r["x_web_mercator"]),
                    float(r["y_web_mercator"])] for r in veh])
    ang = np.radians([float(r["heading_deg"]) for r in veh])

    glyph = {"body": [], "cabin": []}
    for r in read_csv("seoul_car_glyph.csv"):
        glyph[r["part"]].append((float(r["x"]), float(r["y"])))
    glyph = {k: np.array(v) for k, v in glyph.items()}

    roads = {}
    for r in read_csv("seoul_roads.csv"):
        roads.setdefault(r["edge_id"], []).append(
            (int(r["seq"]), float(r["x_web_mercator"]),
             float(r["y_web_mercator"]), int(r["priority"])))
    segs, prio = [], []
    for pts in roads.values():
        pts.sort()
        segs.append([(x, y) for _, x, y, _ in pts])
        prio.append(pts[0][3])
    prio = np.array(prio, float)
    lw = 0.25 + 0.9 * (prio - prio.min()) / max(prio.ptp(), 1)

    xlim = (float(meta["xlim_min"]), float(meta["xlim_max"]))
    ylim = (float(meta["ylim_min"]), float(meta["ylim_max"]))
    vmin = float(meta[f"{args.metric}_vmin"])
    vmax = float(meta[f"{args.metric}_vmax"])
    cmap = plt.get_cmap(meta["colormap"])
    norm = Normalize(vmin=vmin, vmax=vmax)
    car_len = float(meta["car_length_m"])

    def paths(base, size):
        out = []
        for (x, y), a in zip(vm, ang):
            c, s = np.cos(a), np.sin(a)
            v = base @ np.array([[c, s], [-s, c]]) * size + (x, y)
            out.append(Path(v, closed=True))
        return out

    fig, axgrid = plt.subplots(2, 2, figsize=(6.6, 6.3),
                               sharex=True, sharey=True)
    for ax, (label, col) in zip(axgrid.ravel(), panels):
        val = np.array([float(r[col]) for r in veh])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
        if args.tiles:
            import contextily as cx
            cx.add_basemap(ax, crs="EPSG:3857",
                           source=cx.providers.CartoDB.Positron,
                           zoom=int(meta["basemap_zoom"]),
                           attribution_size=4)
        else:
            ax.set_facecolor("#f6f5f3")
            ax.add_collection(LineCollection(segs, colors="#c9c4bd",
                                             linewidths=lw, zorder=1))
        ax.scatter(vm[:, 0], vm[:, 1], c=val, cmap=cmap, norm=norm,
                   s=80, lw=0, alpha=0.20, zorder=3)
        ax.add_collection(PathCollection(paths(glyph["body"], car_len),
                                         facecolors=cmap(norm(val)),
                                         edgecolors="black",
                                         linewidths=0.4, zorder=4))
        ax.add_collection(PathCollection(paths(glyph["cabin"], car_len),
                                         facecolors="black", alpha=0.35,
                                         edgecolors="none", zorder=5))
        ax.text(0.0, 1.045, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10.5)
        ax.text(1.0, 1.045, f"mean {val.mean():.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5)
        first = label == panels[0][0]
        for sp in ax.spines.values():
            sp.set_edgecolor("#1f77b4" if first else "0.45")
            sp.set_linewidth(2.2 if first else 0.8)
    fig.subplots_adjust(wspace=0.04, hspace=0.20)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    lbl = ("Achieved statistical utility" if args.metric == "utility"
           else "Per-vehicle test accuracy (real data)")
    fig.colorbar(sm, ax=axgrid.ravel().tolist(), orientation="horizontal",
                 fraction=0.045, pad=0.04, aspect=45).set_label(lbl)
    out = os.path.join(HERE, f"preview_seoul_map_{args.metric}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
