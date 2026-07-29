"""fig_face_abl_2panel with value labels + delta-vs-full annotations so the
demand / future-contact contributions are legible next to the caching
collapse. Same data (results/metrics_face_ablation_v2x.npz), styling only.

Out: Figures/fig_face_abl_2panel_labeled.{png,pdf}
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.face_figs import ABL_NPZ, ABL_ORDER

d = np.load(ABL_NPZ)
keys = [k for k, _ in ABL_ORDER if f"{k}__acc_all" in d.files]
labs = [l for k, l in ABL_ORDER if f"{k}__acc_all" in d.files]
xs = np.arange(len(keys))

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1))
ax = axes[0]
full_util = 100 * d["FACE (full)__acc_all"][:, -1].mean()
for off, met, lab, col in ((-0.19, "acc", "All vehicles", "#4C72B0"),
                           (0.19, "poor", "High-demand vehicles", "#DD8452")):
    a = np.stack([100 * d[f"{k}__{met}_all"][:, -1] for k in keys])
    bars = ax.bar(xs + off, a.mean(1), width=0.36, yerr=a.std(1),
                  capsize=2.5, label=lab, color=col, edgecolor="black",
                  lw=0.4)
    if met == "acc":
        for i, (x, v) in enumerate(zip(xs, a.mean(1))):
            txt = f"{v:.1f}" if i == 0 else f"{v:.1f}\n({v - full_util:+.1f})"
            ax.text(x, v + a.std(1)[i] + 1.2, txt, ha="center", va="bottom",
                    fontsize=7.0, fontweight="bold" if i else "normal")
ax.set_ylabel("Final statistical utility (%)")
ax.set_ylim(35, 97)
ax.set_yticks([40, 50, 60, 70, 80, 90])
ax.legend(fontsize=7.5, loc="upper right")

ax = axes[1]
gb, red_gb = [], []
for k in keys:
    mb = d[f"{k}__txmb_all"]
    rr = d[f"{k}__redund_all"]
    gb.append(mb.sum(1) / 1024)
    red_gb.append((mb * rr).sum(1) / 1024)
gb, red_gb = np.array(gb), np.array(red_gb)
use_gb = gb - red_gb
ax.bar(xs, use_gb.mean(1), width=0.5, label="Useful deliveries",
       color="#55A868", edgecolor="black", lw=0.4)
ax.bar(xs, red_gb.mean(1), width=0.5, bottom=use_gb.mean(1),
       yerr=gb.std(1), capsize=2.5, label="Redundant deliveries",
       color="#C44E52", edgecolor="black", lw=0.4)
for x, t, r in zip(xs, gb.mean(1), red_gb.mean(1) / gb.mean(1)):
    ax.text(x, t + 2.5, f"{t:.0f}\n({100*r:.0f}\\% red.)", ha="center",
            va="bottom", fontsize=7.0)
ax.set_ylabel("Communication volume (GB)")
ax.set_ylim(0, 62)
ax.legend(fontsize=7.5, loc="upper right")

for i, ax_ in enumerate(axes):
    ax_.set_xticks(xs)
    ax_.set_xticklabels(labs, fontsize=7.2)
    ax_.grid(True, axis="y", ls=":", alpha=0.5)
    ax_.text(0.5, -0.30, f"({'ab'[i]})", transform=ax_.transAxes,
             ha="center", va="top", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"Figures/fig_face_abl_2panel_labeled.{ext}", dpi=300,
                bbox_inches="tight")
print("saved Figures/fig_face_abl_2panel_labeled.png")
