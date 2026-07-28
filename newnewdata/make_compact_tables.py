"""Compact 6-column tables (manuscript design), final metric layout:

- PEAK table: Accuracy | Loss | Rounds@tau | Comm@tau with
  tau = the strongest BASELINE's final accuracy per dataset; schemes that
  never reach tau show N/R and --- (Comm@tau undefined), so the columns
  carry FACE's numbers only when no baseline sustains the target.
- OFF-PEAK (robustness) table: Accuracy | Loss | Useful-Delivery % |
  Delivery@d=20 (tau economics belong to the main table; the robustness
  window shows that ordering and delivery dominance persist).

Run:  python3 newnewdata/make_compact_tables.py
Out:  newnewdata/tab_peak_compact.tex, newnewdata/tab_offpeak_compact.tex
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

ORDER = [("Caching-assisted", "Cached-DFL"), ("V2V-aware", "V2V"),
         ("Learning-aware", "Learning"), ("mmFedMC", "MFedMC"),
         ("AutoFed", "AutoFed"), ("Proposed", "FACE")]
GROUP_END = {"Learning-aware", "AutoFed"}
TAIL = 20


def stats(npz, tau_mode, want_delivery, tau_fixed=None, exclude=()):
    d = np.load(npz)
    ss = [s for s, _ in ORDER
          if f"{s}__acc_all" in d.files and s not in exclude]
    if tau_mode == "fixed":
        tau = tau_fixed
    elif tau_mode == "bestbaseline":
        tau = max(d[f"{s}__acc"][-1] for s in ss if s != "Proposed")
    else:
        tau = 0.95 * max(d[f"{s}__acc"][-1] for s in ss)
    out = {}
    horizon = None
    for s in ss:
        A = d[f"{s}__acc_all"]; L = d[f"{s}__vloss_all"]
        M = d[f"{s}__txmb_all"]
        horizon = A.shape[1]
        if tau_mode == "fixed":
            # reaching judged on the seed-mean curve (pre-registered target)
            reach = A.mean(0) >= tau
            if reach.any():
                r = int(np.argmax(reach)) + 1
                rr = [r] * A.shape[0]
                gg = [m[:r].sum() / 1024.0 for m in M]
            else:
                rr, gg = [], []
        else:
            rr, gg = [], []
            for a, m in zip(A, M):
                reach = a >= tau
                if reach.any():
                    r = int(np.argmax(reach)) + 1
                    rr.append(r); gg.append(m[:r].sum() / 1024.0)
        v = dict(acc=A[:, -TAIL:].mean(1).mean(),
                 sd=A[:, -TAIL:].mean(1).std(),
                 loss=L[:, -TAIL:].mean(1).mean(),
                 rounds=np.mean(rr) if len(rr) == A.shape[0] else None,
                 gb=np.mean(gg) if len(rr) == A.shape[0] else None,
                 totalgb=float(M.sum(1).mean()) / 1024.0)
        if want_delivery:
            v["ud"] = 100 * d[f"{s}__usat_all"].mean()
            U = d[f"{s}__udeliv_all"]; pm = d[f"{s}__pmask_all"]
            dl = []
            for Ui, pmi in zip(U, pm):
                Up = Ui[:, pmi]; K = Up.shape[0]; dd = 20
                dl.append(np.array([Up[t:t + dd].any(0)
                                    for t in range(K - dd + 1)]).mean())
            v["d20"] = 100 * np.mean(dl)
        out[s] = v
    return out, tau, horizon


def emit(nk, nn, caption, label, mode, out_path, tau_mode="bestbaseline",
         taus=None, exclude=()):
    for p in (nk, nn):
        if not os.path.exists(p):
            print(f"  [{label}] SKIP: {p} not found yet")
            return
    hdr_tau = mode == "tau"
    L = []; a = L.append
    a(r"\begin{table}[t]"); a(r"    \centering")
    a(r"\caption{%s}" % caption); a(r"    \label{%s}" % label)
    a(r"    \footnotesize"); a(r"    \renewcommand{\arraystretch}{1.0}")
    a(r"    \setlength{\tabcolsep}{3pt}")
    a(r"    \begin{tabular}{c|c|c|c|c|c}"); a(r"        \toprule")
    a(r"        \multirow{2}{*}{\textsc{Dataset}}")
    a(r"        & \multirow{2}{*}{\textsc{Method}}")
    a(r"        & \multirow{2}{*}{\textsc{Accuracy}}")
    a(r"        & \multirow{2}{*}{\textsc{Loss}}")
    if hdr_tau:
        a(r"        & \textsc{Rounds.}"); a(r"        & \textsc{Comm.} \\")
        a(""); a(r"        &"); a(r"        &"); a(r"        &")
        a(r"        & @$\tau$ $\downarrow$")
        a(r"        & @$\tau$ (GB) $\downarrow$ \\")
    else:
        a(r"        & \shortstack{\textsc{Useful-}\\\textsc{Deliv. (\%)}}")
        a(r"        & \shortstack{\textsc{Delivery}\\@$d{=}20$ (\%)} \\")
        a(""); a(r"        &"); a(r"        &"); a(r"        &")
        a(r"        & $\uparrow$"); a(r"        & $\uparrow$ \\")
    a(r"        \midrule")
    horizon = 250
    for bi, (ds, npz) in enumerate([("KITTI", nk), ("nuScenes", nn)]):
        st, tau, horizon = stats(npz, tau_mode if hdr_tau else "best95",
                                 not hdr_tau,
                                 tau_fixed=(taus or {}).get(ds),
                                 exclude=exclude)
        group_end = {"AutoFed",
                     "Learning-aware" if "Learning-aware" in st
                     else "V2V-aware"}
        a("")
        a(f"        \\multirow{{{len(st)}}}{{*}}{{\\textsc{{{ds}}}}}")
        ba = max(v["acc"] for v in st.values())
        bl = min(v["loss"] for v in st.values())
        reach = [v for v in st.values() if v["rounds"] is not None]
        br = min((v["rounds"] for v in reach), default=None)
        bg = min((v["gb"] for v in reach), default=None)
        if not hdr_tau:
            bu = max(v["ud"] for v in st.values())
            bd = max(v["d20"] for v in st.values())
        for s, disp in ORDER:
            if s not in st:
                continue
            v = st[s]
            acc = f"{100*v['acc']:.1f} \\pm {100*v['sd']:.1f}"
            acc = f"$\\mathbf{{{acc}}}$" if abs(v["acc"] - ba) < 1e-9 \
                else f"${acc}$"
            loss = f"{v['loss']:.3f}"
            if abs(v["loss"] - bl) < 1e-9:
                loss = f"\\textbf{{{loss}}}"
            if hdr_tau:
                if v["rounds"] is None:
                    c4, c5 = f"$>{horizon}$", f"$>{v['totalgb']:.1f}$"
                else:
                    c4 = f"{v['rounds']:.0f}"; c5 = f"{v['gb']:.1f}"
                    if v["rounds"] == br:
                        c4 = f"\\textbf{{{c4}}}"
                    if v["gb"] == bg:
                        c5 = f"\\textbf{{{c5}}}"
            else:
                c4 = f"{v['ud']:.1f}"; c5 = f"{v['d20']:.1f}"
                if abs(v["ud"] - bu) < 1e-9:
                    c4 = f"\\textbf{{{c4}}}"
                if abs(v["d20"] - bd) < 1e-9:
                    c5 = f"\\textbf{{{c5}}}"
            a(f"        & \\textsc{{{disp}}}"); a(f"        & {acc}")
            a(f"        & {loss}"); a(f"        & {c4}")
            a(f"        & {c5}" + (r" \\[1pt]" if s in group_end else r" \\"))
            a("")
        if bi == 0:
            a(r"        \midrule")
        print(f"  [{label}/{ds}] tau={tau:.4f}")
    a(r"        \bottomrule")
    a(r"        \multicolumn{6}{l}{")
    if hdr_tau and tau_mode == "fixed":
        tt = "/".join(f"{100*(taus or {}).get(ds, 0):.0f}\\%"
                      for ds in ("KITTI", "nuScenes"))
        a(f"            \\scriptsize $\\tau$: pre-specified target accuracy"
          f" ({tt}); $>$: not reached within the ${horizon}$-round horizon"
          f" (lower bounds).")
    elif hdr_tau and tau_mode == "bestbaseline":
        a(r"            \scriptsize $\tau$: final accuracy of the strongest"
          r" baseline; $>$: not reached within the $%d$-round horizon"
          r" (lower bounds)." % horizon)
    elif hdr_tau:
        a(r"            \scriptsize $\tau$: target accuracy; $>$: not"
          r" reached within the $%d$-round horizon (lower bounds)."
          % horizon)
    else:
        a(r"            \scriptsize Useful-delivery ratio and windowed"
          r" $P$(useful delivery within $20$ rounds), high-demand vehicles.")
    a(r"        }")
    a(r"    \end{tabular}"); a(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out_path)


emit("results/metrics_v2x_real_kitti.npz",
     "results/metrics_v2x_real_nuscenes.npz",
     "Performance comparison on the Seoul V2X trace during the evening"
     " rush-hour peak.",
     "tab:seoul_results", "tau", "newnewdata/tab_peak_compact.tex")
emit("newnewdata/metrics_v2x_real_kitti_night.npz",
     "newnewdata/metrics_v2x_real_nuscenes_night.npz",
     "Performance comparison on the Seoul V2X trace during late-night"
     " off-peak hours.",
     "tab:seoul_night", "tau", "newnewdata/tab_offpeak_compact.tex",
     tau_mode="best95")

# Fixed, pre-specified tau shared by BOTH traffic windows (peak = Monday
# 90-min no-replay run, off-peak = the T=250 replay night run until the
# night45b re-collection lands), reaching judged on the seed-mean curve.
# Written to *_fixedtau.tex so the replay tables above stay intact until
# the swap is approved.
FIXED_TAUS = {"KITTI": 0.52, "nuScenes": 0.68}
emit("newnewdata/metrics_v2x_real_kitti_evening45.npz",
     "newnewdata/metrics_v2x_real_nuscenes_evening45.npz",
     "Performance comparison on the Seoul V2X trace during the evening"
     " rush-hour peak (90-min replay-free window).",
     "tab:seoul_results", "tau", "newnewdata/tab_peak_fixedtau.tex",
     tau_mode="fixed", taus=FIXED_TAUS, exclude=("Learning-aware",))
emit("newnewdata/metrics_v2x_real_kitti_night.npz",
     "newnewdata/metrics_v2x_real_nuscenes_night.npz",
     "Performance comparison on the Seoul V2X trace during late-night"
     " off-peak hours.",
     "tab:seoul_night", "tau", "newnewdata/tab_offpeak_fixedtau.tex",
     tau_mode="fixed", taus=FIXED_TAUS, exclude=("Learning-aware",))
