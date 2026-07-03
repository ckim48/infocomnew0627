"""
Per-modality vs fusion comparison (centralized training): how much does each
modality contribute, and how much does fusing them add? Motivates multimodal
FL -- no single sensor suffices, so vehicles benefit from receiving strong
encoders for every modality they carry.

KITTI: camera / lidar / camera+lidar.
nuScenes (mini, with radar): camera / lidar / radar / camera+lidar / all three
(radar covers only ~48% of objects -- a realistically weak but complementary
modality).

Outputs (npz, bar chart, LaTeX table) stay inside this folder.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

from sim.multimodal_model import make_encoder, FusionHead

EPOCHS = 15
BATCH = 128
SEEDS = (2026, 2027, 2028)


def _balanced(y, rng, min_count=0):
    counts = np.bincount(y)
    use = [c for c in range(len(counts)) if counts[c] >= max(min_count, 1)]
    cap = int(min(counts[c] for c in use))
    keep = []
    for c in use:
        ci = np.where(y == c)[0]
        keep.append(rng.choice(ci, min(cap, len(ci)), replace=False))
    return rng.permutation(np.concatenate(keep))


def _load(dataset):
    """Return dict modality->array plus labels, class-balanced."""
    rng = np.random.default_rng(2026)
    if dataset == "kitti":
        from sim.kitti_dataset import build
        img, lid, y, _, _ = build(cache="results/kitti_mm_all.npz")
        data = {"camera": img, "lidar": lid}
        keep = _balanced(y, rng)
    else:
        from sim.nuscenes_dataset import build
        img, lid, rad, y, _, _ = build(with_radar=True)
        data = {"camera": img, "lidar": lid, "radar": rad}
        keep = _balanced(y, rng, min_count=800)   # match the FL task (2 classes)
    y = y[keep]
    # remap labels to 0..C-1
    classes = np.unique(y)
    y = np.searchsorted(classes, y)
    data = {m: a[keep] for m, a in data.items()}
    n = len(y); perm = rng.permutation(n)
    n_test = int(0.2 * n)
    return data, y, perm[n_test:], perm[:n_test], len(classes)


def _train_eval(data, y, tr, te, ncls, mods, seed, device):
    torch.manual_seed(seed)
    t = {m: torch.tensor(data[m], device=device) for m in mods}
    yt = torch.tensor(y, device=device, dtype=torch.long)
    enc = {m: make_encoder(m).to(device) for m in mods}
    import sim.multimodal_model as MM
    head = FusionHead(mods, ncls=ncls).to(device)
    params = list(head.parameters())
    for m in mods:
        params += list(enc[m].parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        for _ in range(max(1, len(tr) // BATCH)):
            b = tr[rng.choice(len(tr), BATCH, replace=False)]
            feats = {m: enc[m](t[m][b]) for m in mods}
            loss = ce(head(feats), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        te_t = torch.tensor(te, device=device)
        feats = {m: enc[m](t[m][te_t]) for m in mods}
        acc = float((head(feats).argmax(1) == yt[te_t]).float().mean())
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plans = {
        "kitti": [("camera",), ("lidar",), ("camera", "lidar")],
        "nuscenes": [("camera",), ("lidar",), ("radar",),
                     ("camera", "lidar"), ("camera", "lidar", "radar")],
    }
    results = {}
    for ds, subsets in plans.items():
        data, y, tr, te, ncls = _load(ds)
        for mods in subsets:
            accs = [_train_eval(data, y, tr, te, ncls, list(mods), sd, device)
                    for sd in SEEDS]
            key = "+".join(m[:3] for m in mods)
            results[f"{ds}|{key}"] = (float(np.mean(accs)), float(np.std(accs)))
            print(f"  [{ds}] {'+'.join(mods):24s} "
                  f"acc {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)
    np.savez(os.path.join(HERE, "metrics_modality_fusion.npz"),
             **{k: np.array(v) for k, v in results.items()})
    _figure(results, plans)
    return results


def _figure(results, plans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False,
    })
    LBL = {"cam": "Camera", "lid": "LiDAR", "rad": "Radar",
           "cam+lid": "Cam+LiD", "cam+lid+rad": "All (fusion)"}
    COL = {1: "#9db8d9", 2: "#4f7ab8", 3: "#e8000b"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [3, 5]})
    for ax, (ds, subsets) in zip(axes, plans.items()):
        keys = ["+".join(m[:3] for m in mods) for mods in subsets]
        mu = [results[f"{ds}|{k}"][0] for k in keys]
        sd = [results[f"{ds}|{k}"][1] for k in keys]
        nm = [len(k.split("+")) for k in keys]
        cols = [COL[min(n, 3)] if n > 1 else COL[1] for n in nm]
        cols = ["#e8000b" if n == max(nm) else ("#4f7ab8" if n > 1 else "#9db8d9")
                for n in nm]
        bars = ax.bar(range(len(keys)), mu, yerr=sd, capsize=3, width=0.62,
                      color=cols, edgecolor="k", linewidth=0.6)
        for b, v in zip(bars, mu):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=9)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([LBL.get(k, k) for k in keys], rotation=18,
                           ha="right", fontsize=9)
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(min(mu) - 0.08, max(mu) + 0.06)
        ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.5)
        ax.set_title(f"({'ab'[list(plans).index(ds)]}) "
                     f"{'KITTI' if ds == 'kitti' else 'nuScenes'}",
                     y=-0.42, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_modality_fusion.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_modality_fusion")


if __name__ == "__main__":
    main()
