"""fig_zone_pred_1x2 (v5, 1x3): empirical validation of the zone-level
mobility statistics FACE relies on.
(a) destination-zone hit rate of the learned transition model vs
    look-ahead h, against the blind-guess baseline;
(b) top-1 / top-3 coverage at the FACE horizons (h = 1, 3, 6);
(c) accuracy as the transition statistics accumulate (learning curve at
    h = 6; unseen origins fall back to the adjacency prior).

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
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})

BASE = 12.5                                   # uniform over 8 adjacent zones
C_PK, C_NG = "#4C72B0", "#DD8452"
WINS = ("Rush-hour peak", "Late-night off-peak")

rows = list(csv.DictReader(open("newnewdata/zone_pred_movers.csv")))
HS = sorted({int(r["horizon"]) for r in rows})


def col(win, key, hs=HS):
    return np.array([100*float(r[key]) for h in hs for r in rows
                     if r["window"] == win and int(r["horizon"]) == h])


lc = list(csv.DictReader(open("newnewdata/zone_pred_learncurve.csv")))
LS = sorted({int(r["L_rounds"]) for r in lc})


def lcol(win, key):
    return np.array([100*float(r[key]) for L in LS for r in lc
                     if r["window"] == win and int(r["L_rounds"]) == L
                     and float(r["decay"]) == 0.97])


fig, axs = plt.subplots(1, 3, figsize=(9.9, 2.75))

# --- (a) hit rate vs look-ahead ---
ax = axs[0]
ax.axvspan(0, 6, color="#55A868", alpha=0.10)
ax.text(3.1, 47.5, "FACE horizon\n($h \\leq 6$)", ha="center", fontsize=8,
        color="#2E6E45")
ax.plot(HS, col(WINS[0], "markov_top1"), color=C_PK, marker="o",
        markersize=4.5, markerfacecolor="white", label=WINS[0])
ax.plot(HS, col(WINS[1], "markov_top1"), color=C_NG, marker="s",
        markersize=4.2, markerfacecolor="white", label=WINS[1])
ax.axhline(BASE, color="0.25", ls="--", lw=1.3)
ax.text(30.3, BASE + 1.3, "blind guess (adjacent zone)", ha="right",
        fontsize=7, color="0.3")
ax.set_xlabel("Look-ahead $h$ (rounds, $\\approx$10 s each)")
ax.set_ylabel("Destination-zone hit rate (%)")
ax.set_xlim(0, 31); ax.set_ylim(0, 55)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.38, "(a)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

# --- (b) top-1 / top-3 vs blind guess at the FACE horizons ---
ax = axs[1]
HB = [1, 3, 6]
x = np.arange(len(HB))
w = 0.34
for off, win, c in [(-w/2, WINS[0], C_PK), (+w/2, WINS[1], C_NG)]:
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
ax.legend(handles=[Patch(fc="0.35", label="top-1"),
                   Patch(fc="0.35", alpha=0.35, label="top-3 coverage"),
                   Line2D([], [], color="0.25", ls="--", lw=1.3,
                          label="blind guess")],
          fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 0.995),
          ncol=3, columnspacing=0.9, handlelength=1.4, handletextpad=0.5,
          borderaxespad=0.0)
ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.38, "(b)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

# --- (c) accuracy as the transition statistics accumulate ---
ax = axs[2]
mins = np.array(LS) / 6.0
ax.plot(mins, lcol(WINS[0], "top3"), color=C_PK, marker="o",
        markersize=4.5, markerfacecolor="white")
ax.plot(mins, lcol(WINS[1], "top3"), color=C_NG, marker="s",
        markersize=4.2, markerfacecolor="white")
ax.axhline(37.5, color="0.25", ls="--", lw=1.3)
ax.text(39.5, 35.4, "blind guess (3 of 8 zones)", ha="right", fontsize=7,
        color="0.3")
ax.text(0.03, 0.965, "top-3 coverage at $h{=}6$",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color="0.15")
ax.set_xlabel("Observed statistics (minutes)")
ax.set_ylabel("Destination-zone hit rate (%)")
ax.set_xlim(0, 42); ax.set_ylim(30, 64)
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.38, "(c)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

fig.tight_layout(w_pad=1.8)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_zone_pred_1x2.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_zone_pred_1x2.png")
