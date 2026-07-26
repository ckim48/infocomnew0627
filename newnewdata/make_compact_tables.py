"""Compact 6-column tables (user's manuscript design): Dataset | Method |
Accuracy | Loss | Rounds@tau | Comm@tau, booktabs rules, [1pt] group gaps,
footnote row. Generates the evening-peak and late-night off-peak tables
from the metrics npz; re-run with other npz paths when new windows land.

Run:  python3 newnewdata/make_compact_tables.py
Out:  newnewdata/tab_peak_compact.tex, newnewdata/tab_offpeak_compact.tex
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

ORDER = [("Caching-assisted", "Cached-DFL"), ("V2V-aware", "V2V"),
         ("Learning-aware", "Learning"), ("mmFedMC", "mmFedMC"),
         ("AutoFed", "AutoFed"), ("Proposed", "FACE")]
GROUP_END = {"Learning-aware", "AutoFed"}      # \\[1pt] after these rows
TAIL = 20


def _stats(npz):
    d = np.load(npz)
    schemes = [s for s, _ in ORDER if f"{s}__acc_all" in d.files]
    tau = 0.95 * max(d[f"{s}__acc"][-1] for s in schemes)
    out = {}
    for s in schemes:
        A = d[f"{s}__acc_all"]; L = d[f"{s}__vloss_all"]
        M = d[f"{s}__txmb_all"]
        acc = A[:, -TAIL:].mean(1)
        rr, gg = [], []
        for a, m in zip(A, M):
            reach = a >= tau
            if reach.any():
                r = int(np.argmax(reach)) + 1
                rr.append(r); gg.append(m[:r].sum() / 1024.0)
        out[s] = dict(
            acc=acc.mean(), sd=acc.std(),
            loss=L[:, -TAIL:].mean(1).mean(),
            rounds=np.mean(rr) if len(rr) == A.shape[0] else None,
            gb=np.mean(gg) if len(rr) == A.shape[0] else None,
            totalgb=float(M.sum(1).mean()) / 1024.0)
    return out


def make(npz_kitti, npz_nusc, caption, label, out_path):
    blocks = [("KITTI", _stats(npz_kitti)), ("nuScenes", _stats(npz_nusc))]
    L = []
    a = L.append
    a(r"\begin{table}[t]")
    a(r"    \centering")
    a(r"\caption{%s}" % caption)
    a(r"    \label{%s}" % label)
    a(r"    \footnotesize")
    a(r"    \renewcommand{\arraystretch}{1.0}")
    a(r"    \setlength{\tabcolsep}{3pt}")
    a(r"    \begin{tabular}{c|c|c|c|c|c}")
    a(r"        \toprule")
    a(r"        \multirow{2}{*}{\textsc{Dataset}}")
    a(r"        & \multirow{2}{*}{\textsc{Method}}")
    a(r"        & \multirow{2}{*}{\textsc{Accuracy}}")
    a(r"        & \multirow{2}{*}{\textsc{Loss}}")
    a(r"        & \textsc{Rounds.}")
    a(r"        & \textsc{Comm.} \\")
    a("")
    a(r"        &")
    a(r"        &")
    a(r"        &")
    a(r"        & @$\tau$ $\downarrow$")
    a(r"        & @$\tau$ (GB) $\downarrow$ \\")
    a(r"        \midrule")
    for bi, (ds, st) in enumerate(blocks):
        a("")
        a(f"        \\multirow{{{len(st)}}}{{*}}{{\\textsc{{{ds}}}}}")
        best_acc = max(v["acc"] for v in st.values())
        best_loss = min(v["loss"] for v in st.values())
        reach = [v for v in st.values() if v["rounds"] is not None]
        best_r = min((v["rounds"] for v in reach), default=None)
        best_g = min((v["gb"] for v in reach), default=None)
        for s, disp in ORDER:
            if s not in st:
                continue
            v = st[s]
            acc = f"{100*v['acc']:.1f} \\pm {100*v['sd']:.1f}"
            acc = f"$\\mathbf{{{acc}}}$" if abs(v["acc"] - best_acc) < 1e-9 \
                else f"${acc}$"
            loss = f"{v['loss']:.3f}"
            if abs(v["loss"] - best_loss) < 1e-9:
                loss = f"\\textbf{{{loss}}}"
            if v["rounds"] is None:
                rd, gb = r"\textsc{N/R}", f"$>{v['totalgb']:.1f}$"
            else:
                rd = f"{v['rounds']:.0f}"
                if best_r is not None and v["rounds"] == best_r:
                    rd = f"\\textbf{{{rd}}}"
                gb = f"{v['gb']:.1f}"
                if best_g is not None and v["gb"] == best_g:
                    gb = f"\\textbf{{{gb}}}"
            a(f"        & \\textsc{{{disp}}}")
            a(f"        & {acc}")
            a(f"        & {loss}")
            a(f"        & {rd}")
            tail = r" \\[1pt]" if s in GROUP_END else r" \\"
            a(f"        & {gb}{tail}")
            a("")
        if bi == 0:
            a(r"        \midrule")
    a(r"        \bottomrule")
    a(r"        \multicolumn{6}{l}{")
    a(r"            \scriptsize $\tau$: target accuracy; N/R: target"
      r" accuracy not reached.")
    a(r"        }")
    a(r"    \end{tabular}")
    a(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out_path)


make("results/metrics_v2x_real_kitti.npz",
     "results/metrics_v2x_real_nuscenes.npz",
     "Performance comparison on the Seoul V2X trace during the evening"
     " rush-hour peak.",
     "tab:seoul_results", "newnewdata/tab_peak_compact.tex")
make("newnewdata/metrics_v2x_real_kitti_night.npz",
     "newnewdata/metrics_v2x_real_nuscenes_night.npz",
     "Performance comparison on the Seoul V2X trace during late-night"
     " off-peak hours.",
     "tab:seoul_night", "newnewdata/tab_offpeak_compact.tex")
