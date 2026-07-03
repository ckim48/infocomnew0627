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
    """Seoul main comparison; Dataset x Method rows when nuScenes exists."""
    datasets = _avail("results/metrics_v2x_real_{}.npz")
    rows, taus = [], {}
    for tag, label in datasets:
        res = _load(os.path.join(ROOT, f"results/metrics_v2x_real_{tag}.npz"))
        schemes = [x for x in SCHEMES if x in res]
        st, tau, K = _stats(res)
        taus[label] = tau
        for s in schemes:
            st[s]["gap"] = st[s]["acc"] - st[s]["poor"]
        best_acc = max(st[s]["acc"] for s in schemes)
        best_poor = max(st[s]["poor"] for s in schemes)
        best_gap = min(st[s]["gap"] for s in schemes)
        best_rounds = min((st[s]["rounds"] for s in schemes if st[s]["rounds"]),
                          default=None)
        best_tx = min((st[s]["cumtx"] for s in schemes if st[s]["cumtx"]),
                      default=None)

        def _row(s):
            e = st[s]
            if e["cumtx"]:
                txcell = f"{e['cumtx']}\\,({e['rounds']})"
                if e["cumtx"] == best_tx:
                    txcell = f"\\textbf{{{txcell}}}"
            else:
                txcell = f"$>{e['totaltx']}$"
            cells = [
                _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best_acc),
                _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best_poor),
                txcell,
            ]
            return ("        & \\textsc{" + DISPLAY.get(s, s) + "} & "
                    + " & ".join(cells) + " \\\\")

        block = [_row(s) for s in FRAMEWORK if s in schemes]
        pub = [_row(s) for s in PUBLISHED if s in schemes]
        if pub:
            block += ["        \\cline{2-5}"] + pub
        block += ["        \\cline{2-5}", _row("Proposed")]
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
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Performance on the real Seoul-Gangnam V2X trace"
        " (real multimodal FL, $N{=}180$, 250 rounds; mean $\\pm$ std over"
        f" 3 seeds; \\%, averaged over the final {TAIL} rounds;"
        f" $\\tau$ = 95\\% of the best final accuracy ({tau_txt});"
        " \\textsc{Tx@$\\tau$ (Rd)} = transmissions (rounds) to reach"
        " $\\tau$; $>$ marks schemes that never reach $\\tau$, showing"
        " total transmissions spent).}",
        "    \\label{tab:seoul_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\resizebox{\\columnwidth}{!}{%",
        "    \\begin{tabular}{c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Dataset} & \\textsc{Method} & \\textsc{Acc} &"
        " \\textsc{Poor Acc} & \\textsc{Tx@$\\tau$ (Rd)} \\\\",
        "        \\hline",
        *body,
        "        \\hline",
        "    \\end{tabular}}",
        "\\end{table}",
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
            cells = [
                _b(acc_txt, e["acc"] == best_acc),
                dacc,
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
        " $\\Delta$\\textsc{Acc} & \\textsc{Tx@$\\tau$} \\\\",
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


if __name__ == "__main__":
    tab_main()
    tab_ablation()
    copy_artifacts()
    print("seoul_pack regenerated:")
    for f in sorted(os.listdir(HERE)):
        if f != "generate.py":
            print("  ", f)
