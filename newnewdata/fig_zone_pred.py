"""fig_zone_pred_1x2 (v3, forecast-vs-reality): (a) destination-zone hit
rate (top-1) and top-3 coverage of the learned transition model vs blind
guessing, FACE horizon banded; (b) the model's h=6 destination forecast
from the busiest zone (shaded cells, top-3 outlined) overlaid with where
test-split vehicles actually went (dots) -- the dots land in the shading.

Out: Figures/fig_zone_pred_1x2.{png,pdf}
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})

rows = list(csv.DictReader(open("newnewdata/zone_pred_movers.csv")))
HS = sorted({int(r["horizon"]) for r in rows})


def col(win, key):
    return np.array([100*float(r[key]) for h in HS for r in rows
                     if r["window"] == win and int(r["horizon"]) == h])


fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1),
                        gridspec_kw={"width_ratios": [1.0, 1.12]})
ax = axs[0]
BASE = 12.5
pk, ng = col("Rush-hour peak", "markov_top1"), col("Late-night off-peak", "markov_top1")
pk3, ng3 = col("Rush-hour peak", "markov_top3"), col("Late-night off-peak", "markov_top3")
# FACE operating band
ax.axvspan(0, 6, color="#55A868", alpha=0.10)
ax.text(3.1, 2.5, "used by FACE\n($h \\leq H{=}6$)", ha="center", fontsize=8,
        color="#2E6E45")
ax.plot(HS, pk3, color="#4C72B0", ls="--", lw=1.3, alpha=0.85)
ax.plot(HS, ng3, color="#DD8452", ls="--", lw=1.3, alpha=0.85)
ax.plot(HS, pk, color="#4C72B0", marker="o", markersize=4.5,
        markerfacecolor="white", label="Rush-hour peak")
ax.plot(HS, ng, color="#DD8452", marker="s", markersize=4.2,
        markerfacecolor="white", label="Late-night off-peak")
ax.axhline(BASE, color="0.25", ls=":", lw=1.4)
ax.text(7.2, BASE + 1.6, "blind guess (adjacent zone)", ha="left",
        fontsize=7, color="0.3")
# multiplier annotations vs blind guess
ax.annotate(f"${ng[0]/BASE:.1f}\\times$", xy=(1, ng[0]),
            xytext=(5.2, ng[0] + 4), fontsize=9.5, color="#B2182B",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.0))
ax.annotate(f"${ng[3]/BASE:.1f}\\times$", xy=(6, ng[3]),
            xytext=(10.2, ng[3] + 5), fontsize=9.5, color="#B2182B",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.0))
ax.set_xlabel("Look-ahead $h$ (rounds, $\\approx$10 s each)")
ax.set_ylabel("Destination-zone hit rate (%)")
ax.set_xlim(0, 31); ax.set_ylim(0, 95)
leg1 = ax.legend(fontsize=7.5, loc="upper right")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color="0.2", ls="-", lw=1.6, label="top-1"),
                   Line2D([], [], color="0.2", ls="--", lw=1.3,
                          label="top-3 coverage")],
          fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.52))
ax.add_artist(leg1)
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.335, "(a)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

# --- (b) forecast vs reality on the map (evening peak, h = 6) ---
import contextily as cx
import sumolib
from pyproj import Transformer
from sim.v2x_trace import SEOUL_NET

CELL, DECAY, H = 300.0, 0.97, 6
tr = np.load("newnewdata/v2x_seoul_trace_evening45.npz", allow_pickle=True)
XY = tr["veh_xy"]; ctr = np.asarray(tr["ctr"], float)
K, N = XY.shape[0], XY.shape[1]
Z = np.floor(XY / CELL).astype(int)
zid, zseq = {}, np.zeros((K, N), dtype=int)
for k in range(K):
    for i in range(N):
        zseq[k, i] = zid.setdefault((Z[k, i, 0], Z[k, i, 1]), len(zid))
nz = len(zid)
split = int(0.6 * K)                       # same train/test split as the CSV
C = np.zeros((nz, nz))
for k in range(1, split):
    C *= DECAY
    for i in range(N):
        C[zseq[k-1, i], zseq[k, i]] += 1.0
rs = C.sum(1, keepdims=True)
P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
Ph = np.linalg.matrix_power(P, H)

# busiest origin zone among test-split movers, and their realized destinations
cnt = np.zeros(nz, int)
for k in range(split, K - H):
    cur, fut = zseq[k], zseq[k + H]
    for i in np.where(cur != fut)[0]:
        cnt[cur[i]] += 1
c_star = int(cnt.argmax())
dest = np.zeros(nz, int)
for k in range(split, K - H):
    cur, fut = zseq[k], zseq[k + H]
    for i in np.where((cur == c_star) & (fut != c_star))[0]:
        dest[fut[i]] += 1
n_mov = int(dest.sum())

row = Ph[c_star].copy(); row[c_star] = 0.0  # mover-conditional destination dist
row /= row.sum()
top3 = np.argsort(row)[::-1][:3]
cov3 = 100.0 * dest[top3].sum() / n_mov

inv = {v: k for k, v in zid.items()}
net = sumolib.net.readNet(SEOUL_NET)
tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def merc(x, y):
    lon, lat = net.convertXY2LonLat(float(x), float(y))
    return tf.transform(lon, lat)


def cell_poly(zi):
    gx, gy = inv[zi]
    x0, y0 = gx * CELL + ctr[0], gy * CELL + ctr[1]
    return np.array([merc(x0, y0), merc(x0 + CELL, y0),
                     merc(x0 + CELL, y0 + CELL), merc(x0, y0 + CELL)])


def cell_ctr(zi):
    gx, gy = inv[zi]
    return merc((gx + .5) * CELL + ctr[0], (gy + .5) * CELL + ctr[1])


ax = axs[1]
show = np.where((row >= 0.01) | (dest > 0))[0]
pts = np.array([cell_ctr(z) for z in show] + [cell_ctr(c_star)])
pad = 1350
ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
ax.set_ylim(pts[:, 1].min() - 600, pts[:, 1].max() + 1600)
ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
try:
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron,
                   zoom=15, attribution_size=3)
except Exception as e:
    print("basemap skipped:", e)

cmap = plt.cm.Blues
pmax = row[show].max() if len(show) else 1.0
patches, colors = [], []
for z in show:
    if row[z] < 0.01:
        continue
    patches.append(MplPolygon(cell_poly(z), closed=True))
    colors.append(cmap(0.20 + 0.75 * row[z] / pmax))
ax.add_collection(PatchCollection(patches, facecolor=colors, alpha=0.60,
                                  edgecolor="none", zorder=2))
# outline the model's top-3 picks, rank tag in the cell corner
for rank, z in enumerate(top3, 1):
    poly = cell_poly(z)
    ax.add_patch(MplPolygon(poly, closed=True, fill=False, ec="#08306B",
                            lw=1.6, zorder=4))
    cx_, cy_ = poly[:, 0].min(), poly[:, 1].max()
    ax.text(cx_ + 55, cy_ - 55, str(rank), ha="left", va="top", fontsize=7.5,
            color="#08306B", fontweight="bold", zorder=8,
            bbox=dict(fc="white", ec="#08306B", lw=0.6, pad=1.3, alpha=0.9))
# origin zone
op = cell_poly(c_star)
ax.add_patch(MplPolygon(op, closed=True, fill=False, ec="0.15", lw=1.6,
                        ls=(0, (3, 1.6)), zorder=4))
ax.annotate("origin", xy=(op[:, 0].min(), op.mean(0)[1]),
            xytext=(0.04, 0.36), textcoords="axes fraction",
            fontsize=8, color="0.15", zorder=5,
            arrowprops=dict(arrowstyle="->", color="0.15", lw=0.9))
# realized destinations of the test-split movers
dz = np.where(dest > 0)[0]
dp = np.array([cell_ctr(z) for z in dz])
ax.scatter(dp[:, 0], dp[:, 1], s=18 + 150 * dest[dz] / dest.max(),
           c="#B2182B", alpha=0.85, lw=0.8, edgecolor="white", zorder=6)
ax.text(0.02, 0.975,
        f"shading: model forecast ($h{{=}}6$)\n"
        f"dots: realized destinations ($n{{=}}{n_mov}$)\n"
        f"top-3 zones catch {cov3:.0f}% of movers",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
        bbox=dict(fc="white", ec="0.6", lw=0.5, alpha=0.92), zorder=7)
ax.text(0.5, -0.055, "(b)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)
fig.tight_layout(w_pad=1.0)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_zone_pred_1x2.{ext}", dpi=300,
                bbox_inches="tight")
print(f"saved Figures/fig_zone_pred_1x2.png  (origin zone {c_star}, "
      f"n_movers {n_mov}, top-3 coverage {cov3:.1f}%)")
