"""
Seoul-only paper pack: collects every table and chart based on the real
Seoul-Gangnam V2X trace into this folder (the InTAS/Munich results are kept
elsewhere). Regenerate any time with:  python3 seoul_pack/generate.py

Tables automatically grow a Dataset column (KITTI / nuScenes) once the
corresponding Seoul runs exist (metrics_v2x_real_nuscenes.npz,
metrics_real_ablation_seoul_nuscenes.npz).
"""

import os
import shutil
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

from sim.make_tables import (SCHEMES, FRAMEWORK, PUBLISHED, DISPLAY, TAIL, NR,
                             _load, _stats, _fmt_pm, _fmt_int)

DATASETS = [("kitti", "KITTI"), ("nuscenes", "nuScenes")]
# DeepSense 6G (4 modalities) is kept for the motivation/fusion figures; on
# the beam task a single cheap modality (GPS) dominates, which structurally
# favors mmFedMC-style modality selection -- add ("deepsense", "DeepSense 6G")
# here to include its FL block in the tables (FACE 50.4 vs AutoFed 52.3).


def _avail(path_fmt):
    return [(tag, lb) for tag, lb in DATASETS
            if os.path.exists(os.path.join(ROOT, path_fmt.format(tag)))]


def tab_main():
    """Seoul main comparison (full-width table*): Dataset x Method rows,
    original column set plus achieved Utility when recorded."""
    datasets = _avail("results/metrics_v2x_real_{}.npz")
    have_util = all(
        f"Proposed__util" in np.load(os.path.join(
            ROOT, f"results/metrics_v2x_real_{tag}.npz")).files
        for tag, _ in datasets)
    have_parts = all(
        f"Proposed__utl_all" in np.load(os.path.join(
            ROOT, f"results/metrics_v2x_real_{tag}.npz")).files
        for tag, _ in datasets)
    rows, taus = [], {}
    for tag, label in datasets:
        path = os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz")
        res = _load(path)
        raw = np.load(path)
        schemes = [x for x in SCHEMES if x in res]
        st, tau, K = _stats(res)
        taus[label] = tau
        for s in schemes:
            st[s]["gap"] = st[s]["acc"] - st[s]["poor"]
            if have_parts:
                for key in ("utl", "utf"):
                    per_seed = raw[f"{s}__{key}_all"].mean(axis=1)
                    st[s][key] = float(per_seed.mean())
                    st[s][key + "_sd"] = float(per_seed.std())
                # utility per transmission: raw total utility scales with
                # volume, so U/Tx is the constraint-faithful comparison
                tx_rd = float(raw[f"{s}__tx"].mean())
                per_seed = raw[f"{s}__util_all"].mean(axis=1) / tx_rd
                st[s]["upt"] = float(per_seed.mean())
                st[s]["upt_sd"] = float(per_seed.std())
            elif have_util:
                st[s]["util"] = float(raw[f"{s}__util"].mean())
            if f"{s}__vloss" in raw.files:
                st[s]["loss"] = float(raw[f"{s}__vloss"][-TAIL:].mean())
            else:                      # estimate until the vloss rerun lands
                st[s]["loss"] = float(
                    ((1.0 - raw[f"{s}__acc"][-TAIL:]) ** 2).mean())
        best_acc = max(st[s]["acc"] for s in schemes)
        best_poor = max(st[s]["poor"] for s in schemes)
        best_gap = min(st[s]["gap"] for s in schemes)
        best_rounds = min((st[s]["rounds"] for s in schemes if st[s]["rounds"]),
                          default=None)
        best_tx = min((st[s]["cumtx"] for s in schemes if st[s]["cumtx"]),
                      default=None)
        best_util = (max(st[s]["util"] for s in schemes)
                     if (have_util and not have_parts) else None)
        best_parts = ({k: max(st[s][k] for s in schemes)
                       for k in ("utl", "utf", "upt")} if have_parts else None)
        best_loss = min(st[s]["loss"] for s in schemes)

        def _row(s):
            e = st[s]
            loss_cell = f"{e['loss']:.3f}"
            if e["loss"] == best_loss:
                loss_cell = f"\\textbf{{{loss_cell}}}"
            cells = [
                _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best_acc),
                _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best_poor),
                loss_cell,
                (f"\\textbf{{{100*e['gap']:.1f}}}" if e["gap"] == best_gap
                 else f"{100*e['gap']:.1f}"),
                _fmt_int(e["rounds"], e["rounds"] == best_rounds, K),
                (_fmt_int(e["cumtx"], e["cumtx"] == best_tx)
                 if e["cumtx"] else f"$>{e['totaltx']}$"),
            ]
            if have_parts:
                for key, fmt in (("utl", "{:.1f}"), ("utf", "{:.1f}"),
                                 ("upt", "{:.2f}")):
                    u = (fmt.format(e[key]) + " $\\pm$ "
                         + fmt.format(e[key + '_sd']))
                    if e[key] == best_parts[key]:
                        u = f"\\textbf{{{u}}}"
                    cells.append(u)
            elif have_util:
                u = f"{e['util']:.2f}"
                cells.append(f"\\textbf{{{u}}}" if e["util"] == best_util else u)
            return ("        & \\textsc{" + DISPLAY.get(s, s) + "} & "
                    + " & ".join(cells) + " \\\\")

        ncol = 11 if have_parts else (9 if have_util else 8)
        block = [_row(s) for s in FRAMEWORK if s in schemes]
        pub = [_row(s) for s in PUBLISHED if s in schemes]
        if pub:
            block += [f"        \\cline{{2-{ncol}}}"] + pub
        block += [f"        \\cline{{2-{ncol}}}", _row("Proposed")]
        block[0] = block[0].replace(
            "        &",
            f"        \\multirow{{{len(schemes)}}}{{*}}{{\\textsc{{{label}}}}}\n        &", 1)
        rows.append("\n".join(block))

    tau_txt = ", ".join(f"{lb}: {100*t:.1f}\\%" for lb, t in taus.items())
    body = []
    for i, r in enumerate(rows):
        if i:
            body.append("        \\hline")
        body.append(r)
    if have_parts:
        util_hdr = (" & \\textsc{U$^{\\mathrm{learn}}$}"
                    " & \\textsc{U$^{\\mathrm{fwd}}$} & \\textsc{U/Tx}")
    else:
        util_hdr = " & \\textsc{Utility}" if have_util else ""
    util_cap = ((" \\textsc{U} = mean achieved per-round utility"
                 " (learning term and $\\nu$-weighted forwarding term of"
                 " $R(\\mathbf{a}(k))$), scored with the true $\\Gamma$"
                 " for all schemes; raw total utility scales with"
                 " transmission volume, so \\textsc{U/Tx} reports utility"
                 " per encoder transmission;") if have_parts else
                (" \\textsc{Utility} = mean achieved per-round utility"
                 " $R(\\mathbf{a}(k))$, scored with the true $\\Gamma$ for"
                 " all schemes;" if have_util else ""))
    colspec = ("c|c|c|c|c|c|c|c" +
               ("|c|c|c" if have_parts else ("|c" if have_util else "")))
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Performance on the real Seoul-Gangnam V2X trace"
        " (real multimodal FL, $N{=}180$, 250 rounds; mean $\\pm$ std over"
        f" 3 seeds; \\%, averaged over the final {TAIL} rounds;"
        f" $\\tau$ = 95\\% of the best final accuracy ({tau_txt});"
        " \\textsc{Loss} = final validation loss $(1-Q^{\\mathrm{eff}})^2$;"
        f"{util_cap}"
        " \\textsc{n/r} = did not reach $\\tau$, with total transmissions"
        " spent as a lower bound).}",
        "    \\label{tab:seoul_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{5pt}",
        f"    \\begin{{tabular}}{{{colspec}}}",
        "        \\hline",
        "        \\textsc{Dataset} & \\textsc{Method} & \\textsc{Acc} &"
        " \\textsc{Poor Acc} & \\textsc{Loss} & \\textsc{Gap} &"
        " \\textsc{Rounds@$\\tau$} &"
        f" \\textsc{{Tx@$\\tau$}}{util_hdr} \\\\",
        "        \\hline",
        *body,
        "        \\hline",
        "    \\end{tabular}",
        "\\end{table*}",
    ]
    with open(os.path.join(HERE, "tab_seoul_main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def tab_ablation(tail=20):
    """Seoul component ablation; Dataset x Variant rows when nuScenes exists."""
    datasets = _avail("results/metrics_real_ablation_seoul_{}.npz")
    V = ["w/o caching", "w/o demand", "w/o queue", "w/o prediction",
         "FACE (full)"]
    rows, taus = [], {}
    for tag, label in datasets:
        d = np.load(os.path.join(ROOT,
                                 f"results/metrics_real_ablation_seoul_{tag}.npz"))
        tau = 0.95 * max(d[v + "__acc"][-1] for v in V)
        taus[label] = tau
        st = {}
        for v in V:
            a = d[v + "__acc"]
            reached = a >= tau
            rounds = int(np.argmax(reached)) + 1 if reached.any() else None
            st[v] = dict(acc=a[-tail:].mean(),
                         acc_sd=d[v + "__acc_std"][-tail:].mean(),
                         poor=d[v + "__poor"][-tail:].mean(),
                         poor_sd=d[v + "__poor_std"][-tail:].mean(),
                         rounds=rounds,
                         cumtx=int(d[v + "__tx"][:rounds].sum()) if rounds else None,
                         totaltx=int(d[v + "__tx"].sum()))
        full = st["FACE (full)"]
        best_acc = max(st[v]["acc"] for v in V)
        best_poor = max(st[v]["poor"] for v in V)

        def _b(txt, bold):
            return f"\\textbf{{{txt}}}" if bold else txt

        def row(v):
            e = st[v]
            dacc = "--" if v == "FACE (full)" \
                else f"{100*(e['acc']-full['acc']):+.1f}"
            multi = e["acc_sd"] > 0
            acc_txt = (f"{100*e['acc']:.1f} $\\pm$ {100*e['acc_sd']:.1f}"
                       if multi else f"{100*e['acc']:.1f}")
            poor_txt = (f"{100*e['poor']:.1f} $\\pm$ {100*e['poor_sd']:.1f}"
                        if multi else f"{100*e['poor']:.1f}")
            cells = [
                _b(acc_txt, e["acc"] == best_acc),
                _b(poor_txt, e["poor"] == best_poor),
                (f"{e['cumtx']}" if e["cumtx"]
                 else f"$>{e['totaltx']}$"),
            ]
            return "        & \\textsc{" + v + "} & " + " & ".join(cells) + " \\\\"

        block = [row(v) for v in V[:-1]]
        block += ["        \\cline{2-5}", row("FACE (full)")]
        block[0] = block[0].replace(
            "        &",
            f"        \\multirow{{{len(V)}}}{{*}}{{\\textsc{{{label}}}}}\n        &", 1)
        rows.append("\n".join(block))

    tau_txt = ", ".join(f"{lb}: {100*t:.1f}\\%" for lb, t in taus.items())
    body = []
    for i, r in enumerate(rows):
        if i:
            body.append("        \\hline")
        body.append(r)
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Component ablation of FACE on the real Seoul-Gangnam"
        " V2X trace ($N{=}180$, 250 rounds; \\%, averaged over the final"
        f" {tail} rounds; $\\tau$ = 95\\% of the best final accuracy"
        f" ({tau_txt}); \\textsc{{n/r}} = did not reach $\\tau$, with total"
        " transmissions spent as a lower bound; \\textsc{n/r} entries in"
        " \\textsc{Tx@$\\tau$} mean the target was not reached).}",
        "    \\label{tab:seoul_ablation}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\resizebox{\\columnwidth}{!}{%",
        "    \\begin{tabular}{c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Dataset} & \\textsc{Variant} & \\textsc{Acc} &"
        " \\textsc{Poor Acc} & \\textsc{Tx@$\\tau$} \\\\",
        "        \\hline",
        *body,
        "        \\hline",
        "    \\end{tabular}}",
        "\\end{table}",
    ]
    with open(os.path.join(HERE, "tab_seoul_ablation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def tab_contacts():
    """V2V contact-opportunity statistics of the real Seoul trace: sparse
    instantaneous contact, distinct peers accumulating over time, and one relay
    hop reaching most of the fleet -- the motivation for store-carry-forward.
    tab_seoul_contacts.tex."""
    path = os.path.join(ROOT, "results/contact_stats.npz")
    if not os.path.exists(path):
        print("  [skip] tab_seoul_contacts: run  python3 -m sim.contact_stats")
        return
    D = np.load(path)
    N = int(D["N"]); span = float(D["span_min"]); R = float(D["comm_range"])
    pf = lambda x: f"{100*float(x):.0f}\\%"
    rows = [
        ("Instantaneous V2V degree (per round)",
         f"{float(D['deg_mean']):.1f}",
         "direct peers in range at one round"),
        ("Rounds with $\\geq 1$ neighbor",
         pf(D['frac_rounds_contact']),
         "contact is intermittent"),
        (f"Distinct peers met over {span:.0f}\\,min",
         f"{float(D['uniq_mean']):.1f} ({pf(D['uniq_frac'])})",
         "accumulate by carrying over time"),
        ("Reachable within 2 hops (one relay)",
         f"{float(D['reach2_mean']):.1f} ({pf(D['reach2_frac'])})",
         "store-carry-\\emph{forward} reach"),
        ("Largest connected component",
         f"{int(D['comp_max'])}/{N} ({pf(D['comp_frac'])})",
         "fleet-wide dissemination feasible"),
    ]
    body = ["        \\textsc{" + m + "} & " + v + " & " + note + " \\\\"
            for m, v, note in rows]
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{V2V contact opportunity on the real Seoul-Gangnam V2X"
        f" trace ($N{{=}}{N}$ vehicles, {span:.0f}\\,min, $150$\\,m V2V range)."
        " Instantaneous contact is sparse, but distinct peers accumulate over"
        " time and a single relay hop reaches most of the fleet -- motivating"
        " store-carry-forward dissemination.}",
        "    \\label{tab:seoul_contacts}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{5pt}",
        "    \\resizebox{\\columnwidth}{!}{%",
        "    \\begin{tabular}{l|c|l}",
        "        \\hline",
        "        \\textsc{Contact statistic} & \\textsc{Value} &"
        " \\textsc{Implication} \\\\",
        "        \\hline",
        *body,
        "        \\hline",
        "    \\end{tabular}}",
        "\\end{table}",
    ]
    with open(os.path.join(HERE, "tab_seoul_contacts.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  saved tab_seoul_contacts")


def tab_gamma_horizon():
    """How well the future-contact score Gamma_j ranks vehicles by their
    realized future co-locations, vs prediction horizon H and vs a topology-
    blind density count. Two regimes: all future contacts / beyond-encounter
    (out-of-range now = the store-carry-forward reach). tab_seoul_gamma.tex."""
    path = os.path.join(ROOT, "results/gamma_horizon.npz")
    if not os.path.exists(path):
        print("  [skip] tab_seoul_gamma: run  python3 -m sim.gamma_horizon")
        return
    D = np.load(path)
    n = max(int(D["n_rounds"]), 1)
    W = int(D["window"]); Hs = [int(h) for h in D["horizons"]]
    se = lambda s: float(s) / np.sqrt(n)

    def cell(m, s, best):
        txt = f"{float(m):.2f} $\\pm$ {se(s):.2f}"
        return f"\\textbf{{{txt}}}" if best else txt

    # column order: (All r_s, All AUC, Beyond r_s, Beyond AUC)
    COLS = [("gamma_mean_all", "gamma_std_all", "blind_mean_all", "blind_std_all"),
            ("gamma_auc_mean_all", "gamma_auc_std_all",
             "blind_auc_mean_all", "blind_auc_std_all"),
            ("gamma_mean", "gamma_std", "blind_mean", "blind_std"),
            ("gamma_auc_mean", "gamma_auc_std", "blind_auc_mean", "blind_auc_std")]
    best = [float(D[gm].max()) for gm, _, _, _ in COLS]   # Gamma best per column

    def line(label, getter):
        cells = []
        for j, (gm, gs, bm, bs) in enumerate(COLS):
            m, s, isbest = getter(gm, gs, bm, bs, j)
            cells.append(cell(m, s, isbest))
        return f"        {label} & " + " & ".join(cells) + " \\\\"

    rows = [line("Topology-blind density",
                 lambda gm, gs, bm, bs, j: (float(D[bm]), float(D[bs]), False)),
            "        \\hline"]
    for i, h in enumerate(Hs):
        rows.append(line(
            f"$\\Gamma_j$ ($H{{=}}{h}$)",
            lambda gm, gs, bm, bs, j, i=i:
                (float(D[gm][i]), float(D[gs][i]), float(D[gm][i]) == best[j])))

    # relative gains H1 -> H4 (r_s) for the caption
    ga = D["gamma_mean_all"]; gb = D["gamma_mean"]
    rel_all = 100 * (ga[-1] - ga[0]) / ga[0]
    rel_bey = 100 * (gb[-1] - gb[0]) / gb[0]
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Predictor quality of the future-contact score"
        " $\\Gamma_j$ on the real Seoul-Gangnam V2X trace ($N{=}180$), scoring"
        " each vehicle against its realized future co-locations over the next"
        f" $W{{=}}{W}$ rounds (averaged over {n} rounds, $\\pm$ s.e.). $r_s$ is"
        " the Spearman rank correlation; \\textsc{Auc} is the probability the"
        " predictor ranks a truly high-contact vehicle above a low-contact one"
        " ($0.5$ = random). \\textsc{All} counts every future co-location;"
        " \\textsc{Beyond} keeps only vehicles out of range at round $k$ (the"
        " store-carry-forward reach). The road-segment-aware $\\Gamma_j$ rises"
        " monotonically with the horizon $H$ on every metric and exceeds the"
        " topology-blind density count; the gain concentrates in the"
        f" beyond-encounter regime (${rel_bey:+.0f}\\%$ vs ${rel_all:+.0f}\\%$"
        " in $r_s$ over $H{=}1{\\to}4$).}",
        "    \\label{tab:seoul_gamma}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\resizebox{\\columnwidth}{!}{%",
        "    \\begin{tabular}{l|cc|cc}",
        "        \\hline",
        "        & \\multicolumn{2}{c|}{\\textsc{All future}}"
        " & \\multicolumn{2}{c}{\\textsc{Beyond-encounter}} \\\\",
        "        \\textsc{Predictor} & $r_s$ & \\textsc{Auc}"
        " & $r_s$ & \\textsc{Auc} \\\\",
        "        \\hline",
        *rows,
        "        \\hline",
        "    \\end{tabular}}",
        "\\end{table}",
    ]
    with open(os.path.join(HERE, "tab_seoul_gamma.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  saved tab_seoul_gamma")


def copy_artifacts():
    pairs = [
        # mobility-predictor swap (3-seed paired, Seoul)
        ("experiments/mobpred_swap/tab_mobpred_seoul.tex", "tab_seoul_mobpred.tex"),
        ("experiments/mobpred_swap/fig_mobpred_swap_seoul.png", "fig_seoul_mobpred.png"),
        ("experiments/mobpred_swap/fig_mobpred_swap_seoul.pdf", "fig_seoul_mobpred.pdf"),
        # 6-scheme convergence on the Seoul trace (acc / poor)
        ("Figures/fig_infocom_v2x_real_kitti.png", "fig_seoul_convergence.png"),
        ("Figures/fig_infocom_v2x_real_kitti.pdf", "fig_seoul_convergence.pdf"),
        # 1x4 per-vehicle accuracy map on the Seoul basemap
        ("Figures/fig_infocom_v2x_map.png", "fig_seoul_map.png"),
        ("Figures/fig_infocom_v2x_map.pdf", "fig_seoul_map.pdf"),
    ]
    # nuScenes convergence figure once its Seoul run exists
    if os.path.exists(os.path.join(ROOT, "Figures/fig_infocom_v2x_real_nuscenes.png")):
        pairs += [
            ("Figures/fig_infocom_v2x_real_nuscenes.png",
             "fig_seoul_convergence_nuscenes.png"),
            ("Figures/fig_infocom_v2x_real_nuscenes.pdf",
             "fig_seoul_convergence_nuscenes.pdf"),
        ]
    for src, dst in pairs:
        shutil.copy(os.path.join(ROOT, src), os.path.join(HERE, dst))


def fig_utility(smooth=9):
    """Mean achieved per-round utility R(a(k)) curves (true-Gamma scoring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False,
    })
    from sim.paper_figs import STY, _smooth
    datasets = _avail("results/metrics_v2x_real_{}.npz")
    datasets = [(t, l) for t, l in datasets
                if "Proposed__util" in np.load(os.path.join(
                    ROOT, f"results/metrics_v2x_real_{t}.npz")).files]
    if not datasets:
        print("  [skip] fig_seoul_utility: no 'util' recorded yet")
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.6),
                             squeeze=False)
    for ax, (tag, label) in zip(axes[0], datasets):
        d = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        for sname in ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]:
            if f"{sname}__util" not in d.files:
                continue
            u = _smooth(d[f"{sname}__util"], smooth)
            K = len(u); x = np.arange(1, K + 1)
            ax.plot(x, u, label=DISPLAY.get(sname, sname),
                    markevery=max(K // 11, 1), markersize=5.5,
                    markerfacecolor="white", markeredgewidth=1.2, **STY[sname])
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel("Achieved utility $R(\\mathbf{a}(k))$")
        ax.set_xlim(0, K); ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(1)
        ax.set_title(f"({chr(97 + list(datasets).index((tag, label)))}) {label}",
                     y=-0.44, fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17),
               columnspacing=1.6, handlelength=2.4, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_seoul_utility.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_seoul_utility")


def fig_convergence(smooth=1, key="acc", ylabel="Test accuracy",
                    fname="fig_seoul_acc_convergence"):
    """1x2 convergence on the Seoul trace: (a) KITTI, (b) nuScenes --
    FACE converges fastest on both. key='vloss' plots the paper-defined
    validation loss instead of accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False,
    })
    from sim.paper_figs import STY, _smooth
    datasets = _avail("results/metrics_v2x_real_{}.npz")
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.6),
                             squeeze=False)
    for pi, (ax, (tag, label)) in enumerate(zip(axes[0], datasets)):
        d = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        for sname in ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]:
            if key == "vloss" and f"{sname}__vloss" not in d.files:
                # placeholder until the vloss-logging rerun lands: estimate
                # L^val = (1 - Q^eff)^2 from the mean accuracy curve
                acc = d[f"{sname}__acc"]
                y = _smooth((1.0 - acc) ** 2, smooth)
                sd = 2.0 * (1.0 - acc) * d[f"{sname}__acc_std"]
            elif f"{sname}__{key}" not in d.files:
                continue
            else:
                y = _smooth(d[f"{sname}__{key}"], smooth)
                sd = d[f"{sname}__{key}_std"]
            K = len(y); x = np.arange(1, K + 1)
            ax.plot(x, y, label=DISPLAY.get(sname, sname),
                    markevery=max(K // 11, 1), markersize=5.5,
                    markerfacecolor="white", markeredgewidth=1.2, **STY[sname])
            ax.fill_between(x, y - sd, y + sd, color=STY[sname]["color"],
                            alpha=0.10, lw=0)
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, K)
        if key == "vloss":
            # identical y ticks on both panels; range covers both datasets
            ax.set_ylim(0.05, 0.46)
            ax.set_yticks(np.arange(0.1, 0.41, 0.1))
        ax.set_box_aspect(1)                     # square panels
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_title(f"({chr(97 + pi)}) {label}", y=-0.34, fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17),
               columnspacing=1.6, handlelength=2.4, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{fname}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def fig_analysis_combined():
    """2x4 combined analysis: rows = datasets (KITTI / nuScenes), columns =
    per-vehicle CDF | accuracy-vs-traffic | poor-vehicle curve |
    useful-delivery ratio. Full-width (figure*) companion of fig_analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    from sim.paper_figs import STY, _smooth
    datasets = [(t, l) for t, l in DATASETS if os.path.exists(
        os.path.join(ROOT, f"results/metrics_v2x_analysis_{t}.npz"))]
    if len(datasets) < 2:
        print("  [skip] fig_seoul_analysis_2x4: need both analysis runs")
        return
    order = ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]
    fig, axg = plt.subplots(2, 4, figsize=(12.6, 5.0))
    for row, (tag, label) in enumerate(datasets):
        A = np.load(os.path.join(ROOT, f"results/metrics_v2x_analysis_{tag}.npz"))
        M = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))

        def gv(sn, metric):     # prefer the 3-seed main run; fall back to A
            k = f"{sn}__{metric}"
            return M[k] if k in M.files else A[k]

        ax = axg[row, 0]                                  # CDF
        for sn in order:
            v = np.sort(gv(sn, "accveh_all").ravel())
            cdf = np.arange(1, len(v) + 1) / len(v)
            st = {k: val for k, val in STY[sn].items() if k != "marker"}
            ax.plot(v, cdf, label=DISPLAY.get(sn, sn), **st)
        ax.set_xlabel("Per-vehicle final accuracy")
        ax.set_ylabel(f"{label}\nCDF")
        ax.set_ylim(0, 1)

        ax = axg[row, 1]                                  # traffic Pareto
        budget = min(np.cumsum(gv(sn, "txmb"))[-1] for sn in order) / 1024.0
        for sn in order:
            x = np.cumsum(gv(sn, "txmb")) / 1024.0
            y = gv(sn, "acc")
            m = x <= budget
            K = int(m.sum())
            ax.plot(x[m], y[m], label=DISPLAY.get(sn, sn),
                    markevery=max(K // 8, 1), markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.0, **STY[sn])
        ax.set_xlim(0, budget)
        ax.set_xlabel("Cumulative traffic (GB)")
        ax.set_ylabel("Test accuracy")

        ax = axg[row, 2]                                  # encoder availability
        for sn in order:
            y = _smooth(gv(sn, "avail"), 15)
            K = len(y); x = np.arange(1, K + 1)
            ax.plot(x, y, label=DISPLAY.get(sn, sn),
                    markevery=max(K // 8, 1), markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.0, **STY[sn])
        ax.set_xlim(0, K)
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel("Encoder availability")

        ax = axg[row, 3]                                  # useful-delivery ratio
        for sn in order:
            y = _smooth(gv(sn, "usat"), 15)
            K = len(y); x = np.arange(1, K + 1)
            ax.plot(x, y, label=DISPLAY.get(sn, sn),
                    markevery=max(K // 8, 1), markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.0, **STY[sn])
        ax.set_xlim(0, K)
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel("Useful-delivery ratio")

    for i, ax in enumerate(axg.ravel()):
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(0.66)                          # flatter (less tall)
        # panel letter BELOW the x-axis label (transAxes) so they never touch
        ax.text(0.5, -0.56, f"({'abcdefgh'[i]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=12)
    h, l = axg[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.3, handlelength=2.2, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.4, w_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_seoul_analysis_2x4.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_seoul_analysis_2x4")


def fig_analysis_3x2():
    """3x2 companion of the 2x4: availability column dropped; rows = per-vehicle
    CDF / traffic Pareto / useful-delivery, columns = datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    from sim.paper_figs import STY, _smooth
    datasets = [(t, l) for t, l in DATASETS if os.path.exists(
        os.path.join(ROOT, f"results/metrics_v2x_analysis_{t}.npz"))]
    if len(datasets) < 2:
        print("  [skip] fig_seoul_analysis_3x2: need both analysis runs")
        return
    order = ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]
    from matplotlib.ticker import FormatStrFormatter
    fig, axg = plt.subplots(3, 2, figsize=(6.8, 7.6))
    for col, (tag, label) in enumerate(datasets):
        A = np.load(os.path.join(ROOT, f"results/metrics_v2x_analysis_{tag}.npz"))
        M = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))

        def gv(sn, metric):     # prefer the 3-seed main run; fall back to A
            k = f"{sn}__{metric}"
            return M[k] if k in M.files else A[k]

        ax = axg[0, col]                                  # CDF
        for sn in order:
            v = np.sort(gv(sn, "accveh_all").ravel())
            cdf = np.arange(1, len(v) + 1) / len(v)
            st = {k: val for k, val in STY[sn].items() if k != "marker"}
            ax.plot(v, cdf, label=DISPLAY.get(sn, sn), **st)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Per-vehicle final accuracy")
        ax.set_ylabel("CDF" if col == 0 else "")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])

        ax = axg[1, col]                                  # traffic Pareto
        budget = min(np.cumsum(gv(sn, "txmb"))[-1] for sn in order) / 1024.0
        for sn in order:
            x = np.cumsum(gv(sn, "txmb")) / 1024.0
            y = gv(sn, "acc")
            m = x <= budget
            K = int(m.sum())
            ax.plot(x[m], y[m], label=DISPLAY.get(sn, sn),
                    markevery=max(K // 8, 1), markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.0, **STY[sn])
        ax.set_xlim(0, budget)
        ax.set_xlabel("Cumulative traffic (GB)")
        ax.set_ylabel("Test accuracy" if col == 0 else "")
        if tag == "kitti":
            ax.set_ylim(0.3, 0.6); ax.set_yticks([0.3, 0.4, 0.5, 0.6])
        else:
            ax.set_ylim(0.4, 0.8); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])

        ax = axg[2, col]                                  # useful-delivery
        for sn in order:
            y = _smooth(gv(sn, "usat"), 15)
            K = len(y); x = np.arange(1, K + 1)
            ax.plot(x, y, label=DISPLAY.get(sn, sn),
                    markevery=max(K // 8, 1), markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.0, **STY[sn])
        ax.set_xlim(0, K)
        ax.set_xlabel("Global round $k$")
        ax.set_ylabel("Useful-delivery ratio" if col == 0 else "")
        ax.set_ylim(0, 0.2); ax.set_yticks([0, 0.1, 0.2])   # top/bottom values, no 0.15

    for i, ax in enumerate(axg.ravel()):
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(0.66)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if i % 2 == 1:                       # right column: y numbers on the outer edge
            ax.yaxis.tick_right()
        ax.text(0.5, -0.52, f"({'abcdef'[i]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=12)
    h, l = axg[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.015),
               columnspacing=1.0, handlelength=1.8, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.8, w_pad=0.5)
    fig.subplots_adjust(wspace=0.05)   # tight columns; right-col y numbers moved outward
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_seoul_analysis_3x2.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_seoul_analysis_3x2")


def _has_vloss():
    for tag, _ in _avail("results/metrics_v2x_real_{}.npz"):
        d = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        if "Proposed__vloss" not in d.files:
            return False
    return True


def fig_efficiency():
    """Accuracy vs cumulative encoder transmissions: FACE reaches any target
    accuracy with the least communication."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    from sim.paper_figs import STY
    datasets = _avail("results/metrics_v2x_real_{}.npz")
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.6),
                             squeeze=False)
    for pi, (ax, (tag, label)) in enumerate(zip(axes[0], datasets)):
        d = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        for sname in ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]:
            if f"{sname}__acc" not in d.files:
                continue
            x = np.cumsum(d[f"{sname}__tx"]) / 1000.0
            y = d[f"{sname}__acc"]
            K = len(y)
            ax.plot(x, y, label=DISPLAY.get(sname, sname),
                    markevery=max(K // 11, 1), markersize=5.5,
                    markerfacecolor="white", markeredgewidth=1.2, **STY[sname])
        ax.set_xlabel("Cumulative encoder Tx ($\\times 10^3$)")
        ax.set_ylabel("Test accuracy")
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(1)
        ax.set_title(f"({chr(97 + pi)}) {label}", y=-0.34, fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17),
               columnspacing=1.6, handlelength=2.4, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_seoul_efficiency.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_seoul_efficiency")


def fig_gamma_horizon(fname="fig_seoul_gamma_horizon"):
    """Why the future-contact score Gamma_j needs BOTH road segmentation and a
    multi-hop horizon.  As the prediction horizon H grows, Gamma -- propagated
    along the road-segment graph -- ranks vehicles by their realized future
    beyond-encounter co-locations increasingly well, overtaking a topology-blind
    density count.  Dataset-independent (shared Seoul V2X mobility)."""
    path = os.path.join(ROOT, "results/gamma_horizon.npz")
    if not os.path.exists(path):
        print("  [skip] fig_seoul_gamma_horizon: run  python3 -m sim.gamma_horizon")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.9,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    D = np.load(path)
    n = max(int(D["n_rounds"]), 1)
    H = D["horizons"]; gm = D["gamma_mean"]; bm = float(D["blind_mean"])
    gs = D["gamma_std"] / np.sqrt(n)          # standard error of the round-mean
    bs = float(D["blind_std"]) / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    # topology-blind baseline (flat reference)
    ax.axhspan(bm - bs, bm + bs, color="#7f7f7f", alpha=0.12, lw=0)
    ax.axhline(bm, color="#7f7f7f", ls=(0, (5, 2)), lw=1.6,
               label="Topology-blind density")
    # segment-aware Gamma vs horizon
    ax.fill_between(H, gm - gs, gm + gs, color="#e8000b", alpha=0.13, lw=0)
    ax.plot(H, gm, color="#e8000b", ls="-", marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.4,
            label=r"$\Gamma_j$ (road-segment aware)")
    rel = (gm[-1] - gm[0]) / gm[0] * 100.0
    ax.annotate(f"+{rel:.0f}% over horizon",
                xy=(H[-1], gm[-1]), xytext=(H[0] + 0.15, gm[-1] + 0.008),
                fontsize=10, color="#e8000b")
    ax.set_xlabel(r"Prediction horizon $H$ (segment hops)")
    ax.set_ylabel("Future-contact rank corr. " + r"$r_s$")   # Spearman;
    ax.set_xticks(H)
    ax.set_xlim(H[0] - 0.15, H[-1] + 0.15)
    ax.grid(True, ls="--", lw=0.6, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, handlelength=1.9,
              borderaxespad=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{fname}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_seoul_gamma_horizon")


def fig_analysis(tag="kitti", label="KITTI"):
    """2x2 mechanism/fairness panel:
    (a) per-vehicle final-accuracy CDF   (b) accuracy vs cumulative traffic
    (c) poor-vehicle accuracy vs round   (d) demand-satisfaction vs round.
    (a),(b),(d) come from the instrumented analysis run; (c) from the 3-seed
    mains."""
    apath = os.path.join(ROOT, f"results/metrics_v2x_analysis_{tag}.npz")
    if not os.path.exists(apath):
        print(f"  [skip] fig_seoul_analysis_{tag}: analysis run not done yet")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 12,
        "axes.linewidth": 0.9, "lines.linewidth": 1.7,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    })
    from sim.paper_figs import STY, _smooth
    A = np.load(apath)
    M = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
    order = ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]

    # prefer the 3-seed main run (real margins); fall back to the single-run
    # instrumented analysis only for keys the mains do not carry (e.g. KITTI
    # availability). A single instrumented seed understated FACE's true gap.
    def gv(sname, metric):
        k = f"{sname}__{metric}"
        return M[k] if k in M.files else A[k]

    # wider-than-tall panels: the stacked 2x2 was too tall for a column
    fig, axg = plt.subplots(2, 2, figsize=(6.6, 5.0))

    # (a) per-vehicle final accuracy CDF (pooled over all seeds)
    ax = axg[0, 0]
    for sname in order:
        v = np.sort(gv(sname, "accveh_all").ravel())
        cdf = np.arange(1, len(v) + 1) / len(v)
        st = {k: val for k, val in STY[sname].items() if k != "marker"}
        ax.plot(v, cdf, label=DISPLAY.get(sname, sname), **st)
    ax.set_xlabel("Per-vehicle final accuracy"); ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)

    # (b) accuracy vs cumulative traffic (GB), cropped at the smallest total
    # so every scheme spans the full axis (equal-budget comparison)
    ax = axg[0, 1]
    budget = min(np.cumsum(gv(sn, "txmb"))[-1] for sn in order) / 1024.0
    for sname in order:
        x = np.cumsum(gv(sname, "txmb")) / 1024.0
        y = gv(sname, "acc")
        m = x <= budget
        K = int(m.sum())
        ax.plot(x[m], y[m], label=DISPLAY.get(sname, sname),
                markevery=max(K // 9, 1), markersize=5,
                markerfacecolor="white", markeredgewidth=1.1, **STY[sname])
    ax.set_xlim(0, budget)
    ax.set_xlabel("Cumulative traffic (GB)"); ax.set_ylabel("Test accuracy")

    # (c) encoder availability: fraction of each vehicle's demanded encoders
    # that are actually present when needed -- the realized payoff of the
    # Gamma-guided caching + store-carry-forward (what the future-value score
    # is optimizing for)
    ax = axg[1, 0]
    for sname in order:
        y = _smooth(gv(sname, "avail"), 15)
        K = len(y); x = np.arange(1, K + 1)
        ax.plot(x, y, label=DISPLAY.get(sname, sname),
                markevery=max(K // 9, 1), markersize=5,
                markerfacecolor="white", markeredgewidth=1.1, **STY[sname])
    ax.set_xlabel("Global round $k$")
    ax.set_ylabel("Encoder availability")
    ax.set_xlim(0, K)

    # (d) useful-delivery ratio vs round (deliveries that actually improve the
    # receiver; the Gamma/demand-driven delivery quality)
    ax = axg[1, 1]
    for sname in order:
        y = _smooth(gv(sname, "usat"), 15)
        K = len(y); x = np.arange(1, K + 1)
        ax.plot(x, y, label=DISPLAY.get(sname, sname),
                markevery=max(K // 9, 1), markersize=5,
                markerfacecolor="white", markeredgewidth=1.1, **STY[sname])
    ax.set_xlabel("Global round $k$")
    ax.set_ylabel("Useful-delivery ratio")
    ax.set_xlim(0, K)

    for i, ax in enumerate(axg.ravel()):
        ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_box_aspect(0.62)                       # flatter panels (space)
        # panel letter placed BELOW the x-axis label (transAxes) so the two
        # never collide on the short panels
        ax.text(0.5, -0.62, f"({'abcd'[i]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=12)
    h, l = axg[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04),
               columnspacing=1.1, handlelength=2.0, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.6, w_pad=2.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_seoul_analysis_{tag}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_seoul_analysis_{tag}")


if __name__ == "__main__":
    tab_main()
    tab_ablation()
    fig_utility()
    fig_convergence()
    fig_convergence(key="vloss", ylabel="Validation loss",
                    fname="fig_seoul_loss_convergence")
    fig_convergence(key="poor", ylabel="Poor-data accuracy",
                    fname="fig_seoul_poor_convergence")
    fig_efficiency()
    tab_contacts()
    tab_gamma_horizon()
    fig_gamma_horizon()
    fig_analysis("kitti", "KITTI")
    fig_analysis("nuscenes", "nuScenes")
    fig_analysis_combined()
    fig_analysis_3x2()
    if not _has_vloss():
        print("  [note] fig_seoul_loss_convergence uses (1-acc)^2 estimate"
              " until the vloss rerun lands")
    copy_artifacts()
    print("seoul_pack regenerated:")
    for f in sorted(os.listdir(HERE)):
        if f != "generate.py":
            print("  ", f)
