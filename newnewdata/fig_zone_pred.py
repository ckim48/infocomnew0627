"""fig_zone_pred_1x2 (v4, conventional): (a) destination-zone hit rate of
the learned transition model vs look-ahead h, with the blind-guess
baseline; (b) grouped bars at the FACE horizons (h = 1, 3, 6): top-1
(dark) stacked to top-3 coverage (light) vs the blind-guess line.

Out: Figures/fig_zone_pred_1x2.{png,pdf}
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})

rows = list(csv.DictReader(open("newnewdata/zone_pred_movers.csv")))
HS = sorted({int(r["horizon"]) for r in rows})
BASE = 12.5                                   # uniform over 8 adjacent zones
C_PK, C_NG = "#4C72B0", "#DD8452"


def col(win, key, hs=HS):
    return np.array([100*float(r[key]) for h in hs for r in rows
                     if r["window"] == win and int(r["horizon"]) == h])


fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))

# --- (a) hit rate vs look-ahead ---
ax = axs[0]
pk = col("Rush-hour peak", "markov_top1")
ng = col("Late-night off-peak", "markov_top1")
ax.axvspan(0, 6, color="#55A868", alpha=0.10)
ax.text(3.1, 47.5, "FACE horizon\n($h \\leq 6$)", ha="center", fontsize=8,
        color="#2E6E45")
ax.plot(HS, pk, color=C_PK, marker="o", markersize=4.5,
        markerfacecolor="white", label="Rush-hour peak")
ax.plot(HS, ng, color=C_NG, marker="s", markersize=4.2,
        markerfacecolor="white", label="Late-night off-peak")
ax.axhline(BASE, color="0.25", ls="--", lw=1.3)
ax.text(30.3, BASE + 1.3, "blind guess (adjacent zone)", ha="right",
        fontsize=7, color="0.3")
ax.set_xlabel("Look-ahead $h$ (rounds, $\\approx$10 s each)")
ax.set_ylabel("Destination-zone hit rate (%)")
ax.set_xlim(0, 31); ax.set_ylim(0, 55)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.35, "(a)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

# --- (b) top-1 / top-3 vs blind guess at the FACE horizons ---
ax = axs[1]
HB = [1, 3, 6]
x = np.arange(len(HB))
w = 0.34
for off, win, c in [(-w/2, "Rush-hour peak", C_PK),
                    (+w/2, "Late-night off-peak", C_NG)]:
    t1 = col(win, "markov_top1", HB)
    t3 = col(win, "markov_top3", HB)
    ax.bar(x + off, t1, w, color=c, edgecolor="black", lw=0.6, zorder=3)
    ax.bar(x + off, t3 - t1, w, bottom=t1, color=c, alpha=0.35,
           edgecolor="black", lw=0.6, zorder=3)
ax.axhline(BASE, color="0.25", ls="--", lw=1.3, zorder=4)
ax.set_xticks(x); ax.set_xticklabels([f"$h={h}$" for h in HB])
ax.set_xlabel("Look-ahead within FACE horizon")
ax.set_ylabel("Destination-zone hit rate (%)")
ax.set_ylim(0, 100)
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(fc="0.35", label="top-1"),
                   Patch(fc="0.35", alpha=0.35, label="top-3 coverage"),
                   Line2D([], [], color="0.25", ls="--", lw=1.3,
                          label="blind guess")],
          fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.03),
          borderaxespad=0.2, handlelength=1.7)
ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.35, "(b)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

fig.tight_layout(w_pad=2.2)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_zone_pred_1x2.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_zone_pred_1x2.png")
