"""
Export the exact plotted data of four paper figures as CSVs (new_data/),
so the figures can be re-drawn externally:

  seoul_map_realacc.csv   -- fig_seoul_map_realacc_loo(_compact): one row
                             per vehicle (lon/lat, WebMercator x/y, heading)
                             with per-scheme test accuracy (400-round run).
  seoul_deadline.csv      -- fig_seoul_deadline: delivery success mean/std
                             per dataset x scheme x deadline.
  seoul_calib.csv         -- fig_seoul_calib: per dataset x predicted-gain
                             decile: mean predicted / realized gain + SEM
                             (raw units; the figure shows x1e3).
  face_abl_2panel.csv     -- fig_face_abl_2panel(_sep): per ablation
                             variant: final acc / high-demand acc (%),
                             useful & redundant delivery volume (GB),
                             means and stds over seeds.

Run:  python3 -m sim.export_fig_csv
"""

import csv
import os
import numpy as np

OUT = "new_data"
MAP_SCHEMES = ["Proposed", "Caching-assisted", "V2V-aware", "Learning-aware"]
SCHEMES6 = MAP_SCHEMES + ["mmFedMC", "AutoFed"]
DEADLINES = (1, 2, 3, 5, 10, 20)


def _w(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def export_map(npz="results/metrics_v2x_real_kitti_map400.npz"):
    from pyproj import Transformer
    cache = np.load("results/v2x_map_cache.npz")
    vm, ang = cache["vm"], cache["ang"]
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(vm[:, 0], vm[:, 1])
    d = np.load(npz)
    accs = {s: d[f"{s}__accveh_all"].mean(0) for s in MAP_SCHEMES}
    rows = []
    for i in range(len(vm)):
        rows.append([i, f"{lon[i]:.6f}", f"{lat[i]:.6f}",
                     f"{vm[i, 0]:.1f}", f"{vm[i, 1]:.1f}",
                     f"{np.degrees(ang[i]):.1f}"]
                    + [f"{accs[s][i]:.4f}" for s in MAP_SCHEMES])
    _w("seoul_map_realacc.csv",
       ["vehicle", "lon", "lat", "x_web_mercator", "y_web_mercator",
        "heading_deg", "acc_FACE", "acc_CachedDFL", "acc_V2V",
        "acc_LearningAware"], rows)


def export_deadline():
    rows = []
    for tag in ("kitti", "nuscenes"):
        path = f"results/metrics_v2x_real_{tag}_events.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        for sn in SCHEMES6:
            if f"{sn}__udeliv_all" not in z.files:
                continue
            ys = []
            for U, pm in zip(z[f"{sn}__udeliv_all"], z[f"{sn}__pmask_all"]):
                Up = U[:, pm]
                K = Up.shape[0]
                ys.append([np.array([Up[t:t + dl].any(0)
                                     for t in range(K - dl + 1)]).mean()
                           for dl in DEADLINES])
            ys = np.array(ys)
            for j, dl in enumerate(DEADLINES):
                rows.append([tag, sn, dl, f"{ys[:, j].mean():.4f}",
                             f"{ys[:, j].std():.4f}"])
    _w("seoul_deadline.csv",
       ["dataset", "scheme", "deadline_rounds", "delivery_success_mean",
        "delivery_success_std"], rows)


def export_calib(warmup=30, nbins=10):
    rows = []
    for tag in ("kitti", "nuscenes"):
        path = f"results/metrics_v2x_real_{tag}_events.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        c = z["Proposed__calib_all"]        # seed, round, pred, realized
        m = c[:, 1] >= warmup
        pred, real = c[m, 2], c[m, 3]
        q = np.quantile(pred, np.linspace(0, 1, nbins + 1))
        for i in range(nbins):
            mm = (pred >= q[i]) & ((pred < q[i + 1]) if i < nbins - 1
                                   else (pred <= q[i + 1]))
            sem = real[mm].std() / np.sqrt(mm.sum())
            rows.append([tag, i + 1, int(mm.sum()),
                         f"{pred[mm].mean():.6f}", f"{real[mm].mean():.6f}",
                         f"{sem:.6f}"])
    _w("seoul_calib.csv",
       ["dataset", "decile", "n_events", "predicted_gain_mean",
        "realized_gain_mean", "realized_gain_sem"], rows)


def export_ablation():
    from .face_figs import ABL_NPZ, ABL_ORDER
    d = np.load(ABL_NPZ)
    rows = []
    for k, lab in ABL_ORDER:
        if f"{k}__acc_all" not in d.files:
            continue
        acc = 100 * d[f"{k}__acc_all"][:, -1]
        poor = 100 * d[f"{k}__poor_all"][:, -1]
        mb = d[f"{k}__txmb_all"]
        rr = d[f"{k}__redund_all"]
        gb = mb.sum(1) / 1024
        red = (mb * rr).sum(1) / 1024
        use = gb - red
        lab_flat = lab.replace("\n", " ")
        rows.append([k, lab_flat,
                     f"{acc.mean():.2f}", f"{acc.std():.2f}",
                     f"{poor.mean():.2f}", f"{poor.std():.2f}",
                     f"{use.mean():.2f}", f"{use.std():.2f}",
                     f"{red.mean():.2f}", f"{red.std():.2f}"])
    _w("face_abl_2panel.csv",
       ["variant_key", "variant_label", "acc_all_mean_pct",
        "acc_all_std_pct", "acc_highdemand_mean_pct",
        "acc_highdemand_std_pct", "useful_gb_mean", "useful_gb_std",
        "redundant_gb_mean", "redundant_gb_std"], rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    export_map()
    export_deadline()
    export_calib()
    export_ablation()


if __name__ == "__main__":
    main()
