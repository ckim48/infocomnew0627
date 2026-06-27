"""
Road-awareness demonstration: road-constrained vs straight-line mobility
prediction on the real Ingolstadt (InTAS) network.

For each vehicle we predict its position H rounds ahead two ways:
  * straight-line  : constant-velocity Euclidean extrapolation (ignores roads),
  * road-aware     : advance the vehicle's travelled distance ALONG the road
                     network, choosing successors at intersections with the
                     learned GAT turn probabilities (Sec. III-E).
We compare both against the realized future position from the SUMO trace. Because
real vehicles are confined to the road graph and turn at intersections, the
road-aware prediction has far lower displacement error -- showing that
accounting for road topology is essential for mobility prediction, which is the
basis of the proposed future-contact-aware caching/forwarding.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch

from .config import Config
from .intas_trace import get_or_build_trace
from .mobility import RoadNetwork, MobilitySim
from .hgat import (train_hgat, build_features, add_self_loops)
from .map_viz import _net_segs_centred


# ---------- polyline geometry ----------
def _cumlen(poly):
    d = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    return np.concatenate([[0.0], np.cumsum(d)])


def _point_at(poly, cum, s):
    s = np.clip(s, 0.0, cum[-1])
    j = int(np.searchsorted(cum, s) - 1)
    j = max(0, min(j, len(poly) - 2))
    seg_len = cum[j + 1] - cum[j] + 1e-9
    t = (s - cum[j]) / seg_len
    return poly[j] + t * (poly[j + 1] - poly[j])


def _project(poly, p):
    """Arc-length of the closest point on the polyline to p."""
    best_s, best_d = 0.0, 1e18
    cum = _cumlen(poly)
    for j in range(len(poly) - 1):
        a, b = poly[j], poly[j + 1]
        ab = b - a
        L2 = (ab ** 2).sum() + 1e-12
        t = np.clip(((p - a) * ab).sum() / L2, 0.0, 1.0)
        proj = a + t * ab
        d = ((p - proj) ** 2).sum()
        if d < best_d:
            best_d, best_s = d, cum[j] + t * np.sqrt(L2)
    return best_s


class RoadAwarePredictor:
    def __init__(self, cfg, road, mob, net_segs, model, road_ei, device="cpu"):
        self.cfg, self.road, self.mob = cfg, road, mob
        self.segs = net_segs
        self.model, self.road_ei, self.device = model, road_ei, device
        self._poly_cache = {}

    def _poly(self, e):
        if e not in self._poly_cache:
            p = np.asarray(self.segs[e], dtype=float)
            self._poly_cache[e] = (p, _cumlen(p))
        return self._poly_cache[e]

    def embeddings(self, k):
        mob, model = self.mob, self.model
        mob.k = k
        z = torch.tensor(mob.traffic_state(), device=self.device)
        x, com_ei = build_features(mob)
        xt = torch.tensor(x, device=self.device)
        com = add_self_loops(torch.tensor(com_ei, device=self.device), mob.N, self.device)
        seg = torch.tensor(mob.seg, device=self.device, dtype=torch.long)
        with torch.no_grad():
            road_emb = model.encode_road(z, self.road_ei)
            veh_emb = model.encode_veh(xt, com, road_emb, seg)
        return veh_emb, road_emb

    def _next_seg(self, i, e, veh_emb, road_emb):
        succ = self.road.successors[e]
        if not succ:
            return None
        if len(succ) == 1:
            return succ[0]
        with torch.no_grad():
            logit = self.model.transition_logits(veh_emb, road_emb, i, e, succ, self.road.turn)
        return succ[int(torch.argmax(logit))]

    def predict(self, i, k, dist, veh_emb, road_emb, max_steps=40, return_path=False):
        """Predicted position after travelling `dist` metres along the roads."""
        e = int(self.mob.veh_seg[k, i])
        poly, cum = self._poly(e)
        s = _project(poly, self.mob.veh_xy[k, i])
        remaining = dist
        path = [_point_at(poly, cum, s)]
        for _ in range(max_steps):
            avail = cum[-1] - s
            if remaining <= avail:
                end = _point_at(poly, cum, s + remaining)
                # densify the final on-segment portion for a smooth path
                for ss in np.linspace(s, s + remaining, 4)[1:]:
                    path.append(_point_at(poly, cum, ss))
                return (end, np.array(path)) if return_path else end
            remaining -= avail
            path.append(_point_at(poly, cum, cum[-1]))
            nxt = self._next_seg(i, e, veh_emb, road_emb)
            if nxt is None:
                end = _point_at(poly, cum, cum[-1])
                return (end, np.array(path)) if return_path else end
            e = nxt
            poly, cum = self._poly(e)
            s = 0.0
            path.append(_point_at(poly, cum, 0.0))
        end = _point_at(poly, cum, s)
        return (end, np.array(path)) if return_path else end


def _straight_line(mob, i, k, H):
    """Constant-velocity Euclidean extrapolation (road-agnostic)."""
    kk = max(k - 3, 0)
    vel = (mob.veh_xy[k, i] - mob.veh_xy[kk, i]) / max(k - kk, 1)
    return mob.veh_xy[k, i] + vel * H


def _prepare_model(cfg, device):
    cache_path = os.path.join(cfg.results_dir, f"intas_trace_N{cfg.num_vehicles}_K{cfg.K}.npz")
    trace = get_or_build_trace(cfg, cache_path, begin=34050.0, dt=2.0, warmup_s=480.0)
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    print("  [mobpred] training GAT predictor ...")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)
    net_segs, _ = _net_segs_centred(road)
    return road, mob, net_segs, model, road_ei


def run(cfg=None, device="cpu", horizons=(3, 6, 10, 15, 20, 30), map_k=40, map_H=15):
    cfg = cfg or Config()
    os.makedirs(cfg.figures_dir, exist_ok=True)
    road, mob, net_segs, model, road_ei = _prepare_model(cfg, device)
    pred = RoadAwarePredictor(cfg, road, mob, net_segs, model, road_ei, device)

    # ---- displacement error vs horizon (averaged over base rounds & vehicles) ----
    # Both predictors are given the SAME travelled distance (recent per-round
    # displacement x H), so the comparison isolates road-following vs going
    # straight, not speed estimation. We also report the subset of vehicles that
    # turn (heading change > 25 deg), where road topology matters most.
    base = list(range(8, mob.Krounds - max(horizons) - 1, 6))
    err_road = {H: [] for H in horizons}
    err_line = {H: [] for H in horizons}
    err_road_turn = {H: [] for H in horizons}
    err_line_turn = {H: [] for H in horizons}

    for k in base:
        veh_emb, road_emb = pred.embeddings(k)
        kk = max(k - 3, 0)
        vel = (mob.veh_xy[k] - mob.veh_xy[kk]) / max(k - kk, 1)     # per-round displacement
        step = np.linalg.norm(vel, axis=1)
        for H in horizons:
            for i in range(mob.N):
                actual = mob.veh_xy[k + H, i]
                p0 = mob.veh_xy[k, i]
                dist = step[i] * H
                pr = pred.predict(i, k, dist, veh_emb, road_emb)
                pl = p0 + vel[i] * H
                e_r = np.linalg.norm(pr - actual)
                e_l = np.linalg.norm(pl - actual)
                err_road[H].append(e_r); err_line[H].append(e_l)
                # heading change of the realized path
                h0 = vel[i]
                h1 = mob.veh_xy[k + H, i] - mob.veh_xy[max(k + H - 3, 0), i]
                if np.linalg.norm(h0) > 1 and np.linalg.norm(h1) > 1:
                    cosang = np.dot(h0, h1) / (np.linalg.norm(h0) * np.linalg.norm(h1))
                    if cosang < np.cos(np.deg2rad(25)):
                        err_road_turn[H].append(e_r); err_line_turn[H].append(e_l)
    mr = [np.mean(err_road[H]) for H in horizons]
    ml = [np.mean(err_line[H]) for H in horizons]
    mrt = [np.mean(err_road_turn[H]) if err_road_turn[H] else np.nan for H in horizons]
    mlt = [np.mean(err_line_turn[H]) if err_line_turn[H] else np.nan for H in horizons]
    print("  [mobpred] mean displacement error (m)  [all vehicles | turning vehicles]:")
    for H, a, b, at, bt in zip(horizons, mr, ml, mrt, mlt):
        print(f"      H={H:3d} ({H*mob.dt:.0f}s): road {a:6.1f} / line {b:6.1f} "
              f"(imp {100*(b-a)/b:+5.1f}%)  | turn: road {at:6.1f} / line {bt:6.1f} "
              f"(imp {100*(bt-at)/bt:+5.1f}%)")

    hs = [H * mob.dt for H in horizons]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    # (a) turning vehicles -- where road topology matters
    ax1.plot(hs, mlt, "--s", color="#7f7f7f", label="Straight-line (road-agnostic)")
    ax1.plot(hs, mrt, "-o", color="#d62728", label="Road-aware (proposed)")
    ax1.set_title("(a) Turning vehicles", fontsize=11)
    ax1.set_xlabel("Prediction horizon (s)"); ax1.set_ylabel("Mean displacement error (m)")
    ax1.grid(True, ls=":", alpha=0.6); ax1.legend(fontsize=9)
    # (b) all vehicles
    ax2.plot(hs, ml, "--s", color="#7f7f7f", label="Straight-line (road-agnostic)")
    ax2.plot(hs, mr, "-o", color="#d62728", label="Road-aware (proposed)")
    ax2.set_title("(b) All vehicles", fontsize=11)
    ax2.set_xlabel("Prediction horizon (s)"); ax2.set_ylabel("Mean displacement error (m)")
    ax2.grid(True, ls=":", alpha=0.6); ax2.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(cfg.figures_dir, "fig_mobpred_error.png")
    fig.savefig(p, dpi=200); fig.savefig(p.replace(".png", ".pdf"))
    plt.close(fig)

    # ---- map figure: example trajectories at base round map_k, horizon map_H ----
    veh_emb, road_emb = pred.embeddings(map_k)
    kk = max(map_k - 3, 0)
    vel_map = (mob.veh_xy[map_k] - mob.veh_xy[kk]) / max(map_k - kk, 1)
    step_map = np.linalg.norm(vel_map, axis=1)
    # pick vehicles where straight-line errs most (i.e., that turn / follow curves)
    gaps = []
    for i in range(mob.N):
        actual = mob.veh_xy[map_k + map_H, i]
        pl = mob.veh_xy[map_k, i] + vel_map[i] * map_H
        gaps.append((np.linalg.norm(pl - actual), i))
    gaps.sort(reverse=True)
    picks = [i for _, i in gaps[:6]]

    allxy = mob.veh_xy[map_k:map_k + map_H + 1, picks].reshape(-1, 2)
    pad = 350
    x0, x1 = allxy[:, 0].min() - pad, allxy[:, 0].max() + pad
    y0, y1 = allxy[:, 1].min() - pad, allxy[:, 1].max() + pad
    seg_in = [s for s in net_segs
              if x0 <= np.mean(s[:, 0]) <= x1 and y0 <= np.mean(s[:, 1]) <= y1]

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.add_collection(LineCollection(seg_in, colors="0.82", linewidths=0.6, zorder=0))
    for n, i in enumerate(picks):
        actual_traj = mob.veh_xy[map_k:map_k + map_H + 1, i]
        p0 = mob.veh_xy[map_k, i]
        pr, rpath = pred.predict(i, map_k, step_map[i] * map_H, veh_emb, road_emb,
                                 return_path=True)
        pl = p0 + vel_map[i] * map_H
        lab = dict(label="Actual future path") if n == 0 else {}
        ax.plot(actual_traj[:, 0], actual_traj[:, 1], "-", color="#2ca02c", lw=2.6,
                zorder=3, **lab)
        # road-aware predicted PATH (follows the network)
        lab = dict(label="Road-aware prediction") if n == 0 else {}
        ax.plot(rpath[:, 0], rpath[:, 1], "-", color="#d62728", lw=1.8, zorder=2, **lab)
        ax.scatter([pr[0]], [pr[1]], c="#d62728", marker="o", s=45, edgecolors="k",
                   linewidths=0.4, zorder=4)
        # straight-line predicted path (cuts across blocks)
        lab = dict(label="Straight-line prediction") if n == 0 else {}
        ax.plot([p0[0], pl[0]], [p0[1], pl[1]], "--", color="#7f7f7f", lw=1.6, zorder=2,
                **lab)
        ax.scatter([pl[0]], [pl[1]], c="#7f7f7f", marker="X", s=55, edgecolors="k",
                   linewidths=0.4, zorder=4)
        ax.scatter([p0[0]], [p0[1]], c="navy", marker="s", s=42, edgecolors="k",
                   linewidths=0.4, zorder=5, label="Current position" if n == 0 else None)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_title(f"Mobility prediction on Ingolstadt (InTAS), horizon {map_H*mob.dt:.0f}s",
                 fontsize=12)
    fig.tight_layout()
    p2 = os.path.join(cfg.figures_dir, "fig_mobpred_map.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    fig.savefig(p2.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", p, "and", p2)


if __name__ == "__main__":
    run()
