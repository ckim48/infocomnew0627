"""fig_zone_pred_1x2: (a) mover-conditional destination-prediction accuracy
vs horizon (both windows, uniform-8 baseline, H=6 marked); (b) dominant
zone-to-zone flows learned by the transition model, drawn on the Gangnam
basemap (evening peak).

Out: Figures/fig_zone_pred_1x2.{png,pdf}
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.6,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})

import csv
rows = list(csv.DictReader(open("newnewdata/zone_pred_movers.csv")))
HS = sorted({int(r["horizon"]) for r in rows})


def series(win, key):
    return [100*float(r[key]) for h in HS for r in rows
            if r["window"] == win and int(r["horizon"]) == h]


fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0),
                        gridspec_kw={"width_ratios": [1.0, 1.15]})
ax = axs[0]
for win, col in [("Rush-hour peak", "#4C72B0"),
                 ("Late-night off-peak", "#DD8452")]:
    ax.plot(HS, series(win, "markov_top3"), color=col, marker="o",
            markersize=4, markerfacecolor="white",
            label=f"{win} (top-3)")
    ax.plot(HS, series(win, "markov_top1"), color=col, marker="s",
            markersize=3.6, ls="--", label=f"{win} (top-1)")
ax.axhline(12.5, color="0.4", ls=":", lw=1.2)
ax.text(29.5, 13.5, "uniform over adjacent zones", ha="right",
        fontsize=7, color="0.35")
ax.axvline(6, color="0.55", lw=0.9, ls="-.")
ax.text(6.6, 86, r"$H{=}6$", fontsize=8, color="0.3")
ax.set_xlabel("Prediction horizon $h$ (rounds)")
ax.set_ylabel("Destination-zone accuracy (\\%)")
ax.set_xlim(0, 31); ax.set_ylim(0, 92)
ax.legend(fontsize=6.8, loc="upper right")
ax.grid(True, ls="--", lw=0.6, alpha=0.5)
ax.text(0.5, -0.34, "(a)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

# --- (b) dominant flows on the basemap (evening peak) ---
import contextily as cx
import sumolib
from pyproj import Transformer
from sim.v2x_trace import SEOUL_NET

CELL, DECAY = 300.0, 0.97
tr = np.load("newnewdata/v2x_seoul_trace_evening45.npz", allow_pickle=True)
XY = tr["veh_xy"]; ctr = np.asarray(tr["ctr"], float)
K, N = XY.shape[0], XY.shape[1]
Z = np.floor(XY / CELL).astype(int)
zid, zseq = {}, np.zeros((K, N), dtype=int)
for k in range(K):
    for i in range(N):
        zseq[k, i] = zid.setdefault((Z[k, i, 0], Z[k, i, 1]), len(zid))
nz = len(zid)
C = np.zeros((nz, nz))
for k in range(1, K):
    C *= DECAY
    for i in range(N):
        C[zseq[k-1, i], zseq[k, i]] += 1.0
inv = {v: k for k, v in zid.items()}
net = sumolib.net.readNet(SEOUL_NET)
tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def merc(zi):
    cx_, cy_ = inv[zi]
    x, y = (cx_ + .5) * CELL + ctr[0], (cy_ + .5) * CELL + ctr[1]
    lon, lat = net.convertXY2LonLat(float(x), float(y))
    return tf.transform(lon, lat)


pos = np.array([merc(z) for z in range(nz)])
occ = C.sum(1)
ax = axs[1]
pad = 300
xl = (pos[:, 0].min() - pad, pos[:, 0].max() + pad)
yl = (pos[:, 1].min() - pad, pos[:, 1].max() + pad)
ax.set_xlim(*xl); ax.set_ylim(*yl)
ax.set_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron,
               zoom=14, attribution_size=3)
Coff = C.copy(); np.fill_diagonal(Coff, 0)
mx = Coff.max()
for a in range(nz):
    b = int(Coff[a].argmax())
    w = Coff[a, b]
    if w < mx * 0.02:
        continue
    ax.annotate("", xy=pos[b], xytext=pos[a],
                arrowprops=dict(arrowstyle="-|>", lw=0.5 + 2.8*w/mx,
                                color="#B2182B", alpha=0.35 + 0.6*w/mx,
                                shrinkA=1.5, shrinkB=1.5))
s = ax.scatter(pos[:, 0], pos[:, 1], s=6 + 60*occ/occ.max(),
               c="#2166AC", alpha=0.55, lw=0, zorder=3)
ax.text(0.02, 0.965, "dominant zone-to-zone flows\n(arrow width $\\propto$ frequency)",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(fc="white", ec="0.6", lw=0.5, alpha=0.85))
ax.text(0.5, -0.06, "(b)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)
fig.tight_layout(w_pad=1.0)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_zone_pred_1x2.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_zone_pred_1x2.png")
