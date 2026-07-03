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
            if have_util:
                st[s]["util"] = float(raw[f"{s}__util"].mean())
        best_acc = max(st[s]["acc"] for s in schemes)
        best_poor = max(st[s]["poor"] for s in schemes)
        best_gap = min(st[s]["gap"] for s in schemes)
        best_rounds = min((st[s]["rounds"] for s in schemes if st[s]["rounds"]),
                          default=None)
        best_tx = min((st[s]["cumtx"] for s in schemes if st[s]["cumtx"]),
                      default=None)
        best_util = max(st[s]["util"] for s in schemes) if have_util else None

        def _row(s):
            e = st[s]
            cells = [
                _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best_acc),
                _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best_poor),
                (f"\\textbf{{{100*e['gap']:.1f}}}" if e["gap"] == best_gap
                 else f"{100*e['gap']:.1f}"),
                _fmt_int(e["rounds"], e["rounds"] == best_rounds, K),
                (_fmt_int(e["cumtx"], e["cumtx"] == best_tx)
                 if e["cumtx"] else f"$>{e['totaltx']}$"),
            ]
            if have_util:
                u = f"{e['util']:.2f}"
                cells.append(f"\\textbf{{{u}}}" if e["util"] == best_util else u)
            return ("        & \\textsc{" + DISPLAY.get(s, s) + "} & "
                    + " & ".join(cells) + " \\\\")

        ncol = 8 if have_util else 7
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
    util_hdr = " & \\textsc{Utility}" if have_util else ""
    util_cap = (" \\textsc{Utility} = mean achieved per-round utility"
                " $R(\\mathbf{a}(k))$, scored with the true $\\Gamma$ for"
                " all schemes;" if have_util else "")
    colspec = "c|c|c|c|c|c|c" + ("|c" if have_util else "")
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Performance on the real Seoul-Gangnam V2X trace"
        " (real multimodal FL, $N{=}180$, 250 rounds; mean $\\pm$ std over"
        f" 3 seeds; \\%, averaged over the final {TAIL} rounds;"
        f" $\\tau$ = 95\\% of the best final accuracy ({tau_txt});"
        f"{util_cap}"
        " \\textsc{n/r} = did not reach $\\tau$, with total transmissions"
        " spent as a lower bound).}",
        "    \\label{tab:seoul_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{5pt}",
        f"    \\begin{{tabular}}{{{colspec}}}",
        "        \\hline",
        "        \\textsc{Dataset} & \\textsc{Method} & \\textsc{Acc} &"
        " \\textsc{Poor Acc} & \\textsc{Gap} & \\textsc{Rounds@$\\tau$} &"
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
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.7 * len(datasets), 3.1),
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
        ax.set_title(f"({chr(97 + list(datasets).index((tag, label)))}) {label}",
                     y=-0.34, fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08),
               columnspacing=1.2, handlelength=2.2, fontsize=10)
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
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.7 * len(datasets), 3.1),
                             squeeze=False)
    for pi, (ax, (tag, label)) in enumerate(zip(axes[0], datasets)):
        d = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        for sname in ["Proposed"] + [x for x in SCHEMES if x != "Proposed"]:
            if f"{sname}__{key}" not in d.files:
                continue
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
        ax.set_xlim(0, K); ax.grid(True, ls="--", lw=0.6, alpha=0.5)
        ax.set_title(f"({chr(97 + pi)}) {label}", y=-0.34, fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08),
               columnspacing=1.2, handlelength=2.2, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{fname}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _has_vloss():
    for tag, _ in _avail("results/metrics_v2x_real_{}.npz"):
        d = np.load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        if "Proposed__vloss" not in d.files:
            return False
    return True


if __name__ == "__main__":
    tab_main()
    tab_ablation()
    fig_utility()
    fig_convergence()
    if _has_vloss():
        fig_convergence(key="vloss", ylabel="Validation loss",
                        fname="fig_seoul_loss_convergence")
    else:
        print("  [skip] fig_seoul_loss_convergence: no 'vloss' yet")
    copy_artifacts()
    print("seoul_pack regenerated:")
    for f in sorted(os.listdir(HERE)):
        if f != "generate.py":
            print("  ", f)
