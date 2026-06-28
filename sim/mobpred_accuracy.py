"""
Mobility-prediction accuracy: how well the hierarchical GAT predicts a vehicle's
next road segment (and multi-hop position) on the real InTAS trace, vs.\\ a
first-order Markov predictor, a go-straight heuristic, and random.

Top-1 next-segment accuracy is measured only at branching segments (>=2 feasible
successors), where prediction is non-trivial.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .mobility_prediction import _prepare_model

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
})
RED, BLU, GRN, BLK, ORG = "#e8000b", "#1f5fd0", "#1f9e3d", "#000000", "#f0a020"


def run(cfg=None, device="cuda"):
    cfg = cfg or Config()
    cfg.num_vehicles = 150; cfg.K = 150; cfg.comm_range = 150.0
    os.makedirs(cfg.figures_dir, exist_ok=True)
    road, mob, net_segs, model, road_ei = _prepare_model(cfg, device)
    turn = road.turn

    # empirical first-order Markov: most-frequent realized successor per segment
    succ_counts = {}
    for k in range(mob.Krounds):
        for i in range(mob.N):
            idx = mob.realized_idx[k, i]
            if idx < 0:
                continue
            e = int(mob.veh_seg[k, i])
            succ_counts.setdefault(e, {})
            succ_counts[e][idx] = succ_counts[e].get(idx, 0) + 1
    markov_pred = {e: max(d, key=d.get) for e, d in succ_counts.items()}

    # straight successor index per segment (turn label 0), else 0
    def straight_idx(e):
        succ = road.successors[e]
        for j, e2 in enumerate(succ):
            if turn.get((e, e2), 0) == 0:
                return j
        return 0

    # evaluate top-1 accuracy at branching segments, bucketed by #successors
    from .mobility_prediction import RoadAwarePredictor
    pred = RoadAwarePredictor(cfg, road, mob, net_segs, model, road_ei, device)

    tot = {m: 0 for m in ["GAT", "Markov", "Straight", "Random"]}
    cor = {m: 0.0 for m in tot}
    by_deg = {}                                   # deg -> [n, gat_correct, markov_correct]
    rounds = list(range(4, mob.Krounds, 2))
    for k in rounds:
        veh_emb, road_emb = pred.embeddings(k)
        for i in range(mob.N):
            idx = mob.realized_idx[k, i]
            if idx < 0:
                continue
            e = int(mob.veh_seg[k, i])
            succ = road.successors[e]
            if len(succ) < 2:
                continue
            with torch.no_grad():
                logit = model.transition_logits(veh_emb, road_emb, i, e, succ, turn)
            gat = int(torch.argmax(logit))
            mk = markov_pred.get(e, 0)
            st = straight_idx(e)
            tot["GAT"] += 1
            cor["GAT"] += (gat == idx)
            cor["Markov"] += (mk == idx)
            cor["Straight"] += (st == idx)
            cor["Random"] += 1.0 / len(succ)
            d = min(len(succ), 4)
            by_deg.setdefault(d, [0, 0.0, 0.0])
            by_deg[d][0] += 1
            by_deg[d][1] += (gat == idx)
            by_deg[d][2] += (mk == idx)
    n = tot["GAT"]
    acc = {m: cor[m] / n for m in cor}
    print("=== next-segment top-1 accuracy (branching segments, n=%d) ===" % n)
    for m in ["GAT", "Markov", "Straight", "Random"]:
        print(f"  {m:9s} {acc[m]:.3f}")

    # ---- figure: (a) overall top-1 bar, (b) accuracy vs #successors ----
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    methods = ["GAT", "Markov", "Straight", "Random"]
    cols = [RED, BLU, ORG, BLK]
    ax[0].bar(range(len(methods)), [acc[m] for m in methods], color=cols, edgecolor="k",
              linewidth=0.6, width=0.62)
    for i2, m in enumerate(methods):
        ax[0].text(i2, acc[m] + 0.01, f"{acc[m]:.2f}", ha="center", fontsize=9)
    ax[0].set_xticks(range(len(methods))); ax[0].set_xticklabels(methods, fontsize=10)
    ax[0].set_ylabel("Next-segment top-1 accuracy")
    ax[0].set_ylim(0, 1.0); ax[0].grid(True, axis="y", ls="--", lw=0.6, alpha=0.5)
    ax[0].set_title("(a)", y=-0.30, fontsize=12)

    degs = sorted(by_deg.keys())
    gat_d = [by_deg[d][1] / by_deg[d][0] for d in degs]
    mk_d = [by_deg[d][2] / by_deg[d][0] for d in degs]
    rnd_d = [1.0 / d for d in degs]
    ax[1].plot(degs, gat_d, "-o", color=RED, ms=6, markerfacecolor="white",
               markeredgewidth=1.2, label="GAT")
    ax[1].plot(degs, mk_d, "--s", color=BLU, ms=6, markerfacecolor="white",
               markeredgewidth=1.2, label="Markov")
    ax[1].plot(degs, rnd_d, ":^", color=BLK, ms=6, markerfacecolor="white",
               markeredgewidth=1.2, label="Random")
    ax[1].set_xticks(degs)
    ax[1].set_xlabel("Number of feasible successors", labelpad=6)
    ax[1].set_ylabel("Top-1 accuracy")
    ax[1].set_ylim(0, 1.0); ax[1].grid(True, ls="--", lw=0.6, alpha=0.5)
    ax[1].legend(fontsize=9, loc="upper right")
    ax[1].set_title("(b)", y=-0.34, fontsize=12)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.22, wspace=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_mobpred_acc.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_mobpred_acc")


if __name__ == "__main__":
    run()
