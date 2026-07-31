"""Convergence-horizon figure from the T=800 run (fine nuScenes,
evening45): FACE and Cached-DFL between the NoComm lower and FullContact
upper feasibility bounds. Out: Figures/fig_convergence_T800.{png,pdf}"""
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
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})

d = np.load("newnewdata/metrics_v2x_real_nuscenes_evening45_fine_T800.npz",
            allow_pickle=True)
acc = {m: 100*d[f"{m}__acc"] for m in
       ("Proposed", "Caching-assisted", "NoComm", "FullContact")}
r = np.arange(len(acc["Proposed"]))

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.fill_between(r, acc["NoComm"], acc["FullContact"], color="0.85",
                alpha=0.5, lw=0, zorder=1)
ax.plot(r, acc["FullContact"], color="0.25", ls="-.", lw=1.4,
        label="Full-contact (upper bound)", zorder=3)
ax.plot(r, acc["Proposed"], color="#C44E52", lw=1.9, label="FACE", zorder=5)
ax.plot(r, acc["Caching-assisted"], color="#55A868", ls="--", lw=1.5,
        label="Cached-DFL", zorder=4)
ax.plot(r, acc["NoComm"], color="0.45", ls=":", lw=1.6,
        label="No communication", zorder=3)
ax.axhline(60, color="0.6", ls=":", lw=0.9)
ax.text(792, 60.6, r"$\tau$", ha="right", fontsize=9, color="0.35")
ax.set_xlabel("Training round")
ax.set_ylabel("Test accuracy (%)")
ax.set_xlim(0, 800); ax.set_ylim(38, 78)
ax.set_xticks([0, 200, 400, 600, 800])
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(True, ls="--", lw=0.6, alpha=0.45)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_convergence_T800.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_convergence_T800.png")
