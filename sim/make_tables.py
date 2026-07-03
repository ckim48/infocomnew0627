"""
Generate the INFOCOM-style results tables (LaTeX) from the saved real-FL
metrics. Design follows the two-panel PACS-style template: METHOD column,
vertical-bar separated metric columns, baselines grouped above a rule, the
proposed scheme (FACE) in the final row with bold entries.

Accuracy cells are the test accuracy averaged over the final 20 rounds
(mean +- std over 3 seeds). "Rounds@tau" / "Tx@tau" are the first round at
which the seed-mean accuracy reaches tau = 95% of the best final accuracy and
the cumulative encoder transmissions up to that round; ">K" / "--" mark
schemes that never reach tau.
"""

import os
import numpy as np

SCHEMES = ["Caching-assisted", "V2V-aware", "Learning-aware",
           "mmFedMC", "AutoFed", "Proposed"]
# row groups: framework-variant baselines | published benchmarks | proposed
FRAMEWORK = ["Caching-assisted", "V2V-aware", "Learning-aware"]
PUBLISHED = ["mmFedMC", "AutoFed"]
DISPLAY = {"Proposed": "FACE", "Caching-assisted": "Caching",
           "V2V-aware": "V2V", "Learning-aware": "Learning"}
TAIL = 20  # rounds averaged for the accuracy cells


def _load(path):
    """Load metrics keyed by scheme; schemes absent from the file (e.g. runs
    predating the published benchmarks) are dropped from the table."""
    d = np.load(path)
    res = {s: {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(s + "__")}
           for s in SCHEMES}
    return {s: v for s, v in res.items() if v}


def _stats(res):
    """Per-scheme table entries for one dataset."""
    schemes = [x for x in SCHEMES if x in res]
    tau = 0.95 * max(res[s]["acc"][-1] for s in schemes)
    K = len(res["Proposed"]["acc"])
    out = {}
    for s in schemes:
        acc = res[s]["acc"][-TAIL:].mean()
        acc_sd = res[s]["acc_std"][-TAIL:].mean()
        poor = res[s]["poor"][-TAIL:].mean()
        poor_sd = res[s]["poor_std"][-TAIL:].mean()
        reached = res[s]["acc"] >= tau
        rounds = int(np.argmax(reached)) + 1 if reached.any() else None
        tx = res[s]["tx"]
        cumtx = int(tx[:rounds].sum()) if rounds else None
        out[s] = dict(acc=acc, acc_sd=acc_sd, poor=poor, poor_sd=poor_sd,
                      rounds=rounds, cumtx=cumtx, totaltx=int(tx.sum()))
    return out, tau, K


def _fmt_pm(v, sd, bold):
    cell = f"{100*v:.1f} $\\pm$ {100*sd:.1f}"
    return f"\\textbf{{{cell}}}" if bold else cell


NR = "\\textsc{n/r}"      # target accuracy not reached within the run


def _fmt_int(v, bold, K=None):
    if v is None:
        return NR
    return f"\\textbf{{{v}}}" if bold else f"{v}"


def _combined_table(datasets):
    """Single table* over all datasets (Dataset | Method | metric columns),
    with per-dataset best entries in bold."""
    rows, taus = [], {}
    for tag, label in datasets:
        res = _load(f"results/metrics_real_{tag}.npz")
        schemes = [x for x in SCHEMES if x in res]
        st, tau, K = _stats(res)
        taus[label] = tau
        for s in schemes:
            st[s]["gap"] = st[s]["acc"] - st[s]["poor"]
        best_acc = max(st[s]["acc"] for s in schemes)
        best_poor = max(st[s]["poor"] for s in schemes)
        best_gap = min(st[s]["gap"] for s in schemes)
        best_rounds = min(st[s]["rounds"] for s in schemes if st[s]["rounds"])
        best_tx = min(st[s]["cumtx"] for s in schemes if st[s]["cumtx"])
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
            return (f"        & \\textsc{{{DISPLAY.get(s, s)}}} & "
                    + " & ".join(cells) + " \\\\")

        ncol = 7
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
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Performance comparison on KITTI and nuScenes over the"
        " InTAS (Munich) mobility trace"
        f" (\\%, mean $\\pm$ std over 3 seeds, averaged over the final {TAIL}"
        " rounds). \\textsc{Gap} is the mean-to-poor accuracy gap;"
        " \\textsc{Rounds@$\\tau$} / \\textsc{Tx@$\\tau$} are the rounds and"
        " cumulative encoder transmissions to reach $\\tau$ (95\\% of the best"
        f" final accuracy; {tau_txt});"
        " \\textsc{n/r} = did not reach $\\tau$, with the total"
        " transmissions spent shown as a lower bound.}",
        "    \\label{tab:real_dataset_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{5pt}",
        "    \\begin{tabular}{c|c|c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Dataset} & \\textsc{Method} & \\textsc{Acc} &"
        " \\textsc{Poor Acc} & \\textsc{Gap} & \\textsc{Rounds@$\\tau$} &"
        " \\textsc{Tx@$\\tau$} \\\\",
        "        \\hline",
        rows[0],
        "        \\hline",
        rows[1],
        "        \\hline",
        "    \\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def _mobility_table():
    """Cross-city generality: the same real multimodal FL (KITTI) pipeline on
    the InTAS (Munich) mobility versus the real Seoul-Gangnam V2X trace.
    InTAS cells: mean +- std over 3 seeds; Seoul: single 250-round run."""
    intas = _load("results/metrics_real_kitti.npz")
    seoul = _load("results/metrics_v2x_real_kitti.npz")

    def cells(res, s, with_sd):
        acc = res[s]["acc"][-TAIL:].mean(); poor = res[s]["poor"][-TAIL:].mean()
        if with_sd:
            return [(acc, res[s]["acc_std"][-TAIL:].mean()),
                    (poor, res[s]["poor_std"][-TAIL:].mean())]
        return [(acc, None), (poor, None)]

    schemes = [x for x in SCHEMES if x in intas and x in seoul]
    seoul_multi = bool(seoul["Proposed"]["acc_std"][-TAIL:].mean() > 0)
    data = {s: cells(intas, s, True) + cells(seoul, s, seoul_multi)
            for s in schemes}
    best = [max(data[s][c][0] for s in schemes) for c in range(4)]

    def row(s):
        out = []
        for c, (v, sd) in enumerate(data[s]):
            cell = f"{100*v:.1f}" if sd is None else f"{100*v:.1f} $\\pm$ {100*sd:.1f}"
            out.append(f"\\textbf{{{cell}}}" if data[s][c][0] == best[c] else cell)
        return f"        \\textsc{{{DISPLAY.get(s, s)}}} & " + " & ".join(out) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Generality across mobility environments: real multimodal"
        " FL (KITTI) over the InTAS (Munich) trace and the real"
        " Seoul-Gangnam V2X trace ($N{=}180$, 250 rounds); mean $\\pm$ std"
        " over 3 seeds. Accuracies in \\%, averaged over the final"
        f" {TAIL} rounds.}}",
        "    \\label{tab:mobility}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c|c|c}",
        "        \\hline",
        "        \\multirow{2}{*}{\\textsc{Method}} &"
        " \\multicolumn{2}{c|}{\\textsc{InTAS (Munich)}} &"
        " \\multicolumn{2}{c}{\\textsc{Seoul V2X (Gangnam)}} \\\\",
        "        & \\textsc{Acc} & \\textsc{Poor Acc}"
        " & \\textsc{Acc} & \\textsc{Poor Acc} \\\\",
        "        \\hline",
    ]
    for s in FRAMEWORK:
        if s in schemes:
            lines.append(row(s))
    pub = [row(s) for s in PUBLISHED if s in schemes]
    if pub:
        lines += ["        \\hline"] + pub
    lines += ["        \\hline", row("Proposed"), "        \\hline",
              "    \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _seoul_table():
    """Same metric columns as the combined table, for the real Seoul-Gangnam
    V2X trace run (KITTI, N=180, single 250-round run -> no seed std)."""
    res = _load("results/metrics_v2x_real_kitti.npz")
    schemes = [x for x in SCHEMES if x in res]
    tau = 0.95 * max(res[s]["acc"][-1] for s in schemes)
    K = len(res["Proposed"]["acc"])
    multi = bool(res["Proposed"]["acc_std"][-TAIL:].mean() > 0)
    st = {}
    for s in schemes:
        acc = res[s]["acc"][-TAIL:].mean(); poor = res[s]["poor"][-TAIL:].mean()
        reached = res[s]["acc"] >= tau
        rounds = int(np.argmax(reached)) + 1 if reached.any() else None
        st[s] = dict(acc=acc, poor=poor, gap=acc - poor, rounds=rounds,
                     acc_sd=res[s]["acc_std"][-TAIL:].mean(),
                     poor_sd=res[s]["poor_std"][-TAIL:].mean(),
                     cumtx=int(res[s]["tx"][:rounds].sum()) if rounds else None,
                     totaltx=int(res[s]["tx"].sum()))
    best = dict(acc=max(st[s]["acc"] for s in schemes),
                poor=max(st[s]["poor"] for s in schemes),
                gap=min(st[s]["gap"] for s in schemes),
                rounds=min(st[s]["rounds"] for s in schemes if st[s]["rounds"]),
                cumtx=min(st[s]["cumtx"] for s in schemes if st[s]["cumtx"]))

    def b(txt, is_best):
        return f"\\textbf{{{txt}}}" if is_best else txt

    def row(s):
        e = st[s]
        cells = [
            _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best["acc"]) if multi
            else b(f"{100*e['acc']:.1f}", e["acc"] == best["acc"]),
            _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best["poor"]) if multi
            else b(f"{100*e['poor']:.1f}", e["poor"] == best["poor"]),
            b(f"{100*e['gap']:.1f}", e["gap"] == best["gap"]),
            _fmt_int(e["rounds"], e["rounds"] == best["rounds"], K),
            (_fmt_int(e["cumtx"], e["cumtx"] == best["cumtx"])
             if e["cumtx"] else f"$>{e['totaltx']}$"),
        ]
        return f"        \\textsc{{{DISPLAY.get(s, s)}}} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Performance on the real Seoul-Gangnam V2X trace"
        " (real multimodal FL on KITTI, $N{=}180$, 250 rounds;"
        + (" mean $\\pm$ std over 3 seeds;" if multi else " single run;")
        + f" \\%, averaged over the final {TAIL} rounds;"
        f" $\\tau={100*tau:.1f}\\%$;"
        " \\textsc{n/r} = did not reach $\\tau$).}",
        "    \\label{tab:seoul_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Method} & \\textsc{Acc} & \\textsc{Poor Acc} &"
        " \\textsc{Gap} & \\textsc{Rounds@$\\tau$} & \\textsc{Tx@$\\tau$} \\\\",
        "        \\hline",
    ]
    for s in FRAMEWORK:
        if s in schemes:
            lines.append(row(s))
    pub = [row(s) for s in PUBLISHED if s in schemes]
    if pub:
        lines += ["        \\hline"] + pub
    lines += ["        \\hline", row("Proposed"), "        \\hline",
              "    \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


ABL_VARIANTS = ["w/o caching", "w/o demand", "w/o queue", "w/o prediction",
                "FACE (full)"]


def _abl_stats(path):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    res = {v: {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(v + "__")}
           for v in ABL_VARIANTS}
    return {v: dict(acc=res[v]["acc"][-TAIL:].mean(),
                    acc_sd=res[v]["acc_std"][-TAIL:].mean(),
                    poor=res[v]["poor"][-TAIL:].mean(),
                    poor_sd=res[v]["poor_std"][-TAIL:].mean())
            for v in ABL_VARIANTS if res[v]}


def _ablation_table():
    """Component ablation on the real FL backend, on the dense InTAS trace
    and (when available) the sparse Seoul V2X trace where store-carry-forward
    is expected to matter (Tables/tab_ablation.tex)."""
    intas = _abl_stats("results/metrics_real_ablation_kitti.npz")
    if intas is None:
        return None
    seoul = _abl_stats("results/metrics_real_ablation_seoul.npz")

    def block(st):
        full = st["FACE (full)"]
        best = max(st[v]["acc"] for v in st)

        def cells(v):
            if v not in st:
                return ["--", "--"]
            e = st[v]
            dacc = "--" if v == "FACE (full)" \
                else f"{100*(e['acc']-full['acc']):+.1f}"
            if e["acc_sd"] > 0:
                acc_cell = _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best)
            else:                       # single-seed column: no +- shown
                acc_cell = f"{100*e['acc']:.1f}"
                if e["acc"] == best:
                    acc_cell = f"\\textbf{{{acc_cell}}}"
            return [acc_cell, dacc]
        return cells

    c_in = block(intas)
    c_se = block(seoul) if seoul else None

    def row(v):
        cells = c_in(v) + (c_se(v) if c_se else [])
        return f"        \\textsc{{{v}}} & " + " & ".join(cells) + " \\\\"

    if seoul:
        colspec, header = "c|c|c|c|c", (
            "        \\multirow{2}{*}{\\textsc{Variant}} &"
            " \\multicolumn{2}{c|}{\\textsc{InTAS (Munich)}} &"
            " \\multicolumn{2}{c}{\\textsc{Seoul V2X}} \\\\\n"
            "        & \\textsc{Acc} & $\\Delta$ & \\textsc{Acc} & $\\Delta$ \\\\")
        cap_extra = (" InTAS: mean $\\pm$ std over 3 seeds; Seoul V2X"
                     " (sparse contacts): single 250-round run.")
    else:
        colspec, header = "c|c|c", (
            "        \\textsc{Variant} & \\textsc{Acc} & $\\Delta$\\textsc{Acc} \\\\")
        cap_extra = " Mean $\\pm$ std over 3 seeds."

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Component ablation of FACE (real multimodal FL on"
        f" KITTI; \\%, averaged over the final {TAIL} rounds).{cap_extra}}}",
        "    \\label{tab:ablation}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        f"    \\begin{{tabular}}{{{colspec}}}",
        "        \\hline",
        header,
        "        \\hline",
    ]
    for v in ABL_VARIANTS[:-1]:
        lines.append(row(v))
    lines += ["        \\hline", row("FACE (full)"), "        \\hline",
              "    \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main(outdir="Tables"):
    os.makedirs(outdir, exist_ok=True)
    outputs = [
        ("tab_real_combined.tex",
         _combined_table([("kitti", "KITTI"), ("nuscenes", "nuScenes")])),
        ("tab_mobility.tex", _mobility_table()),
        ("tab_seoul.tex", _seoul_table()),
    ]
    abl = _ablation_table()
    if abl:
        outputs.append(("tab_ablation.tex", abl))
    else:
        print("  [skip] tab_ablation: run `python3 -m sim.real_ablation` first")
    for fname, tex in outputs:
        path = os.path.join(outdir, fname)
        with open(path, "w") as f:
            f.write(tex + "\n")
        print(f"--- {path} ---")
        print(tex)
        print()


if __name__ == "__main__":
    main()
