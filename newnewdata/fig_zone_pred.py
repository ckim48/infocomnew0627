"""fig_zone_pred_1x2 (v7): direction-conditioned zone statistics, 1x2.
(a) top-1 contact probability vs prediction horizon in seconds
    (direction-conditioned vs memoryless first-order vs blind guess;
    shaded region = FACE operating horizon, explained in the caption);
(b) contact probability as the statistics accumulate (top-3, h = 6).
Data: zone_pred_m3{,_learn}.csv (fair tie-breaking, mover-conditional).

Out: Figures/fig_zone_pred_1x2.{png,pdf}
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

rows = list(csv.DictReader(open("newnewdata/zone_pred_m3.csv")))
HS = sorted({int(r["horizon"]) for r in rows})
SEC = np.array(HS) * 10.0                     # rounds -> seconds


def col(win, model, key):
    return np.array([100*float(r[key]) for h in HS for r in rows
                     if r["window"] == win and int(r["horizon"]) == h
                     and r["model"] == model])


lc = list(csv.DictReader(open("newnewdata/zone_pred_m3_learn.csv")))
LS = sorted({int(r["L_rounds"]) for r in lc})


def lcol(win, key):
    return np.array([100*float(r[key]) for L in LS for r in lc
                     if r["window"] == win and int(r["L_rounds"]) == L])


fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))

# --- (a) contact probability vs prediction horizon ---
ax = axs[0]
ax.axvspan(0, 60, color="#55A868", alpha=0.10)
for win, c, mk in ((WINS[0], C_PK, "o"), (WINS[1], C_NG, "s")):
    ax.plot(SEC, col(win, "M1", "top1"), color=c, ls="--", lw=1.2, alpha=0.75)
    ax.plot(SEC, col(win, "M3", "top1"), color=c, marker=mk, markersize=4.4,
            markerfacecolor="white", label=win)
ax.axhline(BASE, color="0.25", ls=":", lw=1.4)
ax.text(303, BASE + 1.5, "blind guess (adjacent zone)", ha="right",
        fontsize=7, color="0.3")
leg1 = ax.legend(fontsize=7.5, loc="upper right")
ax.legend(handles=[Line2D([], [], color="0.2", lw=1.6,
                          label="direction-conditioned"),
                   Line2D([], [], color="0.2", ls="--", lw=1.2,
                          label="memoryless (1st-order)")],
          fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.56))
ax.add_artist(leg1)
ax.set_xlabel("Prediction horizon (s)")
ax.set_ylabel("Contact probability (%)")
ax.set_xlim(0, 310); ax.set_ylim(0, 88)
ax.set_xticks([0, 60, 120, 180, 240, 300])
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.35, "(a)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

# --- (b) contact probability as the statistics accumulate ---
ax = axs[1]
mins = np.array(LS) / 6.0
ax.plot(mins, lcol(WINS[0], "top3"), color=C_PK, marker="o",
        markersize=4.5, markerfacecolor="white")
ax.plot(mins, lcol(WINS[1], "top3"), color=C_NG, marker="s",
        markersize=4.2, markerfacecolor="white")
ax.axhline(37.5, color="0.25", ls=":", lw=1.4)
ax.text(41.0, 34.6, "blind guess (3 of 8 zones)", ha="right", fontsize=7,
        color="0.3")
ax.text(0.03, 0.965, "top-3, 60 s ahead",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color="0.15")
ax.set_xlabel("Observed statistics (minutes)")
ax.set_ylabel("Contact probability (%)")
ax.set_xlim(0, 42); ax.set_ylim(30, 80)
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
ax.text(0.5, -0.35, "(b)", transform=ax.transAxes, ha="center",
        va="top", fontsize=12)

fig.tight_layout(w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_zone_pred_1x2.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_zone_pred_1x2.png")
