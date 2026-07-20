"""
REAL multimodal FL (real SGD encoders + FedAvg + real KITTI classification
accuracy) over the real Seoul V2X mobility trace. This is the technically
rigorous counterpart of sim/run_v2x.py (which uses the abstract coverage proxy):
here accuracy is genuine test accuracy and exhibits real convergence.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, SCHEMES
from .real_fl import REAL_SCHEMES
from .mobility import RoadNetwork, MobilitySim
from .hgat import train_hgat, future_contact_scores
from .algorithm import CachingForwarding
from .face import FACE
from .simulator import make_modality_availability, make_arch_assignment
from .real_fl import RealMFL, _prep_data, _device
from .v2x_trace import build_v2x_trace
from .plotting import STYLE as STY, disp


def _prepare_v2x(cfg, device):
    trace = build_v2x_trace(cfg)
    n_tr = trace["veh_xy"].shape[1]
    if cfg.num_vehicles < n_tr:
        # density scenario: deterministic vehicle subset of the cached trace
        sel = np.random.default_rng(4242).choice(n_tr, cfg.num_vehicles,
                                                 replace=False)
        sel.sort()
        for kk in ("veh_seg", "veh_xy", "veh_speed"):
            trace[kk] = trace[kk][:, sel]
    road = RoadNetwork(trace)
    mob = MobilitySim(cfg, road, trace)
    cfg.K = mob.Krounds
    print(f"      Seoul V2X: |V|={road.V}, N={mob.N}, K={mob.Krounds}")
    model, road_ei = train_hgat(cfg, road, mob, device=device, warmup_rounds=40)
    gammas = []
    for k in range(mob.Krounds):
        mob.k = k
        gammas.append(future_contact_scores(cfg, road, mob, model, road_ei, device=device))
    return road, mob, np.array(gammas)


def run(cfg=None, seeds=None, device=None, num_vehicles=180, dataset="kitti",
        rounds=250, min_class_count=None, schemes=None, merge=False,
        out_name=None, partitioned=False, ttl=15, kx=6, lam=0.001,
        record_class=True, record_veh=False, loc=False):
    """Run REAL FL until convergence. `rounds` may exceed the mobility trace
    length: the Seoul V2X window is replayed cyclically (steady-state traffic),
    while FL keeps training/propagating so the accuracy curve plateaus.

    partitioned=True realizes the Sec. II motivation: encoder-carrier vehicles
    (data-rich taxis) are confined to the commercial west half of the region,
    so strong encoders must be ferried east to reach the demand there."""
    cfg = cfg or Config()
    cfg.num_vehicles = num_vehicles
    cfg.face_ttl = ttl          # real backend: versions expire, sources
    cfg.face_K_tickets = kx     # tighter replication: stale spread hurts here
    cfg.face_Qpub = 1           # republish updated encoders every round
    cfg.face_lam = lam          # communication price per MB
    if dataset == "deepsense":
        import sim.multimodal_model as _MM
        _MM.ENCODER_OVERRIDES.update({"radar": _MM.RadarMapEncoder,
                                      "gps": _MM.GPSEncoder})
        cfg.modalities = ["camera", "lidar", "radar", "gps"]
        cfg.modality_prob = {"camera": 1.0, "lidar": 0.85,
                             "radar": 0.75, "gps": 0.95}
        cfg.per_modality_strength = True   # chi per encoder (LOO val contrib)
    elif dataset == "nuscenes":       # camera + LiDAR + sparse radar returns
        cfg.modalities = ["camera", "lidar", "radar"]
        cfg.modality_prob = {"camera": 1.0, "lidar": 0.85, "radar": 0.7}
    else:                             # KITTI: camera + LiDAR
        cfg.modalities = ["camera", "lidar"]
        cfg.modality_prob = getattr(cfg, "modality_prob_override", None) \
            or {"camera": 1.0, "lidar": 0.85}
    # per-modality availability draws realize missing-modality vehicles
    # (e.g., P(lidar)=0.85 -> 15% vision-only); typed sensor-suite mixtures
    # are available via cfg.vehicle_types (see config.py note)
    device = device or _device()
    seeds = seeds or [cfg.seed]
    os.makedirs(cfg.results_dir, exist_ok=True)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print("[1/3] Building real Seoul V2X mobility + GAT ...")
    road, mob, gammas = _prepare_v2x(cfg, device)
    total = rounds or mob.Krounds
    print(f"      running {total} FL rounds (trace K={mob.Krounds}, "
          f"replayed cyclically)")
    print("[2/3] Loading real KITTI multimodal data ...")
    if min_class_count is None:                # match the InTAS setup
        min_class_count = 800 if dataset == "nuscenes" else 0
    data = _prep_data(cfg, cfg.seed, dataset=dataset,
                      min_class_count=min_class_count, loc=loc)
    if partitioned:
        # data-rich (ECV) candidates confined to the west half of the region,
        # by each vehicle's starting longitude in the Seoul trace
        x0 = mob.veh_xy[0, :, 0]
        data["rich_mask"] = (x0 <= np.median(x0)).astype(bool)
        print(f"      [partitioned] ECV-eligible vehicles: "
              f"{int(data['rich_mask'].sum())}/{mob.N} (west half)")

    todo = schemes or REAL_SCHEMES
    keys = ["acc", "poor", "tx", "util", "utl", "utf", "vloss", "sat", "usat", "txmb", "mhop", "avail"]
    stacks = {s: {m: [] for m in keys} for s in todo}
    print(f"[3/3] REAL FL over seeds {seeds} ...")
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        arch = make_arch_assignment(cfg, np.random.default_rng(sd + 11), avail)
        for scheme in todo:
            torch.manual_seed(sd)          # paired: same init/noise per seed
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            mfl.arch = arch                # architecture families (chi)
            # ALL schemes run under the same system-model protocol (encoder
            # versions, copy tickets, evaluation-gated adoption, Sec. III);
            # they differ only in the forwarding policy (SCHEME_FACE_FLAGS)
            alg = FACE(cfg, mfl, mob, scheme, seed=sd)
            pm = mfl.poor_mask()
            acc_h, poor_h, tx_h, u_h, vl_h = [], [], [], [], []
            sat_h, usat_h, mb_h = [], [], []
            utl_h, utf_h, mh_h, av_h = [], [], [], []
            ud_h = []                      # per-round useful-delivery receivers
            cls_h, clsp_h = [], []         # per-class (service-level) accuracy
            veh_h = []                     # per-round per-vehicle accuracy
            iou_h, iouhd_h = [], []        # RoI localization quality
            for k in range(total):
                kk = k % mob.Krounds                    # replay the trace window
                mob.k = kk
                mfl.local_train()
                mfl.refresh_strengths()
                # paper-defined validation loss L^val = (1 - Q^eff)^2, with
                # Q^eff the real per-vehicle validation accuracy
                vl_h.append(float(np.mean((1.0 - mfl.acc) ** 2)))
                g = gammas[kk] if alg.flags.get("use_dis") \
                    or alg.flags.get("cache_policy") == "psi" \
                    else np.zeros(mob.N)
                selected = alg.run_round(k, g, gamma_eval=gammas[kk])
                accs = mfl.evaluate("test")
                acc_h.append(float(accs.mean()))
                poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
                tx_h.append(len(selected))
                u_h.append(alg.last_utility)
                utl_h.append(alg.last_utility_learn)
                utf_h.append(alg.last_utility_fwd)
                sat_h.append(getattr(alg, "last_satisfaction", 0.0))
                usat_h.append(getattr(alg, "last_useful_sat", 0.0))
                mh_h.append(alg._n_beyond / max(alg._n_deliv, 1))
                av_h.append(getattr(alg, "last_avail", 0.0))
                mb_h.append(sum(cfg.encoder_size[e[3]] for e in selected))
                urow = np.zeros(mob.N, dtype=bool)
                urow[getattr(alg, "last_useful_receivers", [])] = True
                ud_h.append(urow)
                if record_class:
                    ac = mfl.evaluate_class("test")    # N x C
                    cls_h.append(ac.mean(0))
                    clsp_h.append(ac[pm].mean(0) if pm.any()
                                  else ac.mean(0))
                if record_veh:                # readiness curves; reuses the
                    veh_h.append(accs.astype(np.float16))   # existing eval
                if loc:
                    iou = mfl.evaluate_loc("test")
                    iou_h.append(float(iou.mean()))
                    iouhd_h.append(float(iou[pm].mean()) if pm.any()
                                   else float(iou.mean()))
            stacks[scheme]["acc"].append(acc_h)
            stacks[scheme]["poor"].append(poor_h)
            stacks[scheme]["tx"].append(tx_h)
            stacks[scheme]["util"].append(u_h)
            stacks[scheme]["utl"].append(utl_h)
            stacks[scheme]["utf"].append(utf_h)
            stacks[scheme]["vloss"].append(vl_h)
            stacks[scheme]["sat"].append(sat_h)
            stacks[scheme]["usat"].append(usat_h)
            stacks[scheme]["mhop"].append(mh_h)
            stacks[scheme]["avail"].append(av_h)
            stacks[scheme]["txmb"].append(mb_h)
            stacks[scheme].setdefault("accveh", []).append(
                mfl.evaluate("test"))          # per-vehicle final accuracies
            # per-vehicle LOO encoder-contribution chi at the final round
            # (mean of strength chi_{i,r} over the vehicle's modalities)
            chi = np.array([np.mean([mfl.strength[(i, r)]
                                     for r in mfl.avail[i]])
                            if len(mfl.avail[i]) else np.nan
                            for i in range(mob.N)])
            stacks[scheme].setdefault("chiveh", []).append(chi)
            # event-level extras: useful-delivery matrix [K,N], high-demand
            # mask, and (round, predicted, realized) gain-calibration pairs
            stacks[scheme].setdefault("udeliv", []).append(np.array(ud_h))
            stacks[scheme].setdefault("pmask", []).append(np.asarray(pm))
            stacks[scheme].setdefault("calib", []).append(
                np.array(alg.calib, dtype=np.float32))
            if record_class:
                stacks[scheme].setdefault("accclass", []).append(
                    np.array(cls_h))
                stacks[scheme].setdefault("accclass_hd", []).append(
                    np.array(clsp_h))
            if record_veh:
                stacks[scheme].setdefault("accveht", []).append(
                    np.array(veh_h))
            if loc:
                stacks[scheme].setdefault("iou", []).append(np.array(iou_h))
                stacks[scheme].setdefault("iou_hd", []).append(
                    np.array(iouhd_h))
            print(f"  [seed {sd}] {scheme:16s} acc {acc_h[-1]:.3f} "
                  f"poor {poor_h[-1]:.3f} tx/round {np.mean(tx_h):.1f}", flush=True)
            del mfl, alg
            if device == "cuda":
                torch.cuda.empty_cache()

    results = {}
    for s in todo:
        results[s] = {}
        for m in keys:
            arr = np.stack(stacks[s][m])
            results[s][m] = arr.mean(0); results[s][m + "_std"] = arr.std(0)
            results[s][m + "_all"] = arr            # per-seed (exact +- stats)
        results[s]["accveh_all"] = np.stack(stacks[s]["accveh"])  # seeds x N
        if stacks[s].get("chiveh"):
            results[s]["chiveh_all"] = np.stack(stacks[s]["chiveh"])
        if stacks[s].get("udeliv"):
            results[s]["udeliv_all"] = np.stack(stacks[s]["udeliv"])
            results[s]["pmask_all"] = np.stack(stacks[s]["pmask"])
            cal = [np.concatenate([np.full((len(c), 1), si, dtype=np.float32),
                                   c], axis=1)
                   for si, c in enumerate(stacks[s]["calib"]) if len(c)]
            if cal:                      # columns: seed, round, pred, realized
                results[s]["calib_all"] = np.concatenate(cal)
        if stacks[s].get("accclass"):    # seeds x K x C (service classes)
            results[s]["accclass_all"] = np.stack(stacks[s]["accclass"])
            results[s]["accclass_hd_all"] = np.stack(stacks[s]["accclass_hd"])
        if stacks[s].get("accveht"):     # seeds x K x N (readiness curves)
            results[s]["accveht_all"] = np.stack(stacks[s]["accveht"])
        if stacks[s].get("iou"):         # RoI localization curves
            for kk in ("iou", "iou_hd"):
                arr = np.stack(stacks[s][kk])
                results[s][kk] = arr.mean(0)
                results[s][kk + "_std"] = arr.std(0)
                results[s][kk + "_all"] = arr
    path = os.path.join(cfg.results_dir,
                        out_name or f"metrics_v2x_real_{dataset}.npz")
    out = dict(np.load(path)) if (merge and os.path.exists(path)) else {}
    out.update({f"{s}__{k}": v for s, d in results.items() for k, v in d.items()})
    np.savez(path, **out)
    if not merge:
        _plot(results, cfg, dataset)
    print("=== REAL FL on Seoul V2X — final ===")
    for s in todo:
        print(f"  {disp(s):16s} acc {results[s]['acc'][-1]:.3f}  poor {results[s]['poor'][-1]:.3f}")
    return results


def _panel(ax, results, key, ylabel):
    K = len(results["Proposed"][key]); x = np.arange(1, K + 1)
    me = max(K // 11, 1)
    for s in [s for s in REAL_SCHEMES if s in results]:
        ax.plot(x, results[s][key], label=disp(s), markevery=me, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.2, **STY[s])
        ax.fill_between(x, results[s][key] - results[s][key + "_std"],
                        results[s][key] + results[s][key + "_std"],
                        color=STY[s]["color"], alpha=0.12, lw=0)
    ax.set_xlabel("Global round $k$"); ax.set_ylabel(ylabel)
    ax.set_xlim(0, K); ax.grid(True, ls="--", lw=0.6, alpha=0.5)


def _plot(results, cfg, dataset="kitti"):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _panel(axes[0], results, "acc", "Test accuracy")
    _panel(axes[1], results, "poor", "Poor-data accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.4, handlelength=2.6, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"fig_infocom_v2x_real_{dataset}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_infocom_v2x_real")


if __name__ == "__main__":
    run()
