"""KITTI validation-loss convergence, 1x2: (a) evening peak T=400,
(b) late-night off-peak T=250. Same 8-seed sources as the fixed-tau tables.

Out: Figures/fig_kitti_loss_1x2.{png,pdf}
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

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
from sim.paper_figs import STY

ORDER = [("Caching-assisted", "Cached-DFL"), ("V2V-aware", "V2V"),
         ("mmFedMC", "MFedMC"), ("AutoFed", "AutoFed"),
         ("Proposed", "FACE")]
PANELS = [
    ("newnewdata/metrics_v2x_real_kitti_evening45_8seed.npz",
     "Evening rush-hour peak"),
    ("newnewdata/metrics_v2x_real_kitti_night_8seed.npz",
     "Late-night off-peak"),
]

fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.0))
for col, (npz, title) in enumerate(PANELS):
    d = np.load(npz)
    ax = axs[col]
    for s, disp in ORDER:
        L = d[f"{s}__vloss_all"].mean(0)
        st = dict(STY.get(s, {}))
        st.pop("marker", None)
        ax.plot(np.arange(1, len(L) + 1), L, label=disp, **st)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation loss" if col == 0 else "")
    ax.grid(True, ls="--", lw=0.6, alpha=0.5)
    if col == 1:
        ax.yaxis.tick_right()
    ax.text(0.5, -0.38, f"({'ab'[col]})", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)
axs[0].legend(fontsize=7.5, loc="upper right")
fig.tight_layout(w_pad=0.5)
fig.subplots_adjust(wspace=0.05)
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_kitti_loss_1x2.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_kitti_loss_1x2.png")
