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

SCHEMES = ["Caching-assisted", "V2V-aware", "Learning-aware", "Proposed"]
DISPLAY = {"Proposed": "FACE"}
TAIL = 20  # rounds averaged for the accuracy cells


def _load(path):
    d = np.load(path)
    return {s: {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(s + "__")}
            for s in SCHEMES}


def _stats(res):
    """Per-scheme table entries for one dataset."""
    tau = 0.95 * max(res[s]["acc"][-1] for s in SCHEMES)
    K = len(res["Proposed"]["acc"])
    out = {}
    for s in SCHEMES:
        acc = res[s]["acc"][-TAIL:].mean()
        acc_sd = res[s]["acc_std"][-TAIL:].mean()
        poor = res[s]["poor"][-TAIL:].mean()
        poor_sd = res[s]["poor_std"][-TAIL:].mean()
        reached = res[s]["acc"] >= tau
        rounds = int(np.argmax(reached)) + 1 if reached.any() else None
        tx = res[s]["tx"]
        cumtx = int(tx[:rounds].sum()) if rounds else None
        out[s] = dict(acc=acc, acc_sd=acc_sd, poor=poor, poor_sd=poor_sd,
                      rounds=rounds, cumtx=cumtx)
    return out, tau, K


def _fmt_pm(v, sd, bold):
    cell = f"{100*v:.1f} $\\pm$ {100*sd:.1f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def _fmt_int(v, bold, K=None):
    if v is None:
        return f"$>{K}$" if K else "--"
    return f"\\textbf{{{v}}}" if bold else f"{v}"


def _combined_table(datasets):
    """Single table* over all datasets (Dataset | Method | metric columns),
    with per-dataset best entries in bold."""
    rows, taus = [], {}
    for tag, label in datasets:
        res = _load(f"results/metrics_real_{tag}.npz")
        st, tau, K = _stats(res)
        taus[label] = tau
        for s in SCHEMES:
            st[s]["txrd"] = float(res[s]["tx"].mean())
            st[s]["gap"] = st[s]["acc"] - st[s]["poor"]
        best_acc = max(st[s]["acc"] for s in SCHEMES)
        best_poor = max(st[s]["poor"] for s in SCHEMES)
        best_gap = min(st[s]["gap"] for s in SCHEMES)
        best_rounds = min(st[s]["rounds"] for s in SCHEMES if st[s]["rounds"])
        best_tx = min(st[s]["cumtx"] for s in SCHEMES if st[s]["cumtx"])
        best_txrd = min(st[s]["txrd"] for s in SCHEMES)
        block = []
        for s in SCHEMES:
            e = st[s]
            cells = [
                _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best_acc),
                _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best_poor),
                (f"\\textbf{{{100*e['gap']:.1f}}}" if e["gap"] == best_gap
                 else f"{100*e['gap']:.1f}"),
                _fmt_int(e["rounds"], e["rounds"] == best_rounds, K),
                _fmt_int(e["cumtx"], e["cumtx"] == best_tx),
                (f"\\textbf{{{e['txrd']:.1f}}}" if e["txrd"] == best_txrd
                 else f"{e['txrd']:.1f}"),
            ]
            block.append(f"        & \\textsc{{{DISPLAY.get(s, s)}}} & "
                         + " & ".join(cells) + " \\\\")
        block[0] = block[0].replace(
            "        &", f"        \\multirow{{4}}{{*}}{{\\textsc{{{label}}}}}\n        &", 1)
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
        f" final accuracy; {tau_txt}); \\textsc{{Tx/Rd}} is the mean"
        " transmissions per round.}",
        "    \\label{tab:real_dataset_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{5pt}",
        "    \\begin{tabular}{c|c|c|c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Dataset} & \\textsc{Method} & \\textsc{Acc} &"
        " \\textsc{Poor Acc} & \\textsc{Gap} & \\textsc{Rounds@$\\tau$} &"
        " \\textsc{Tx@$\\tau$} & \\textsc{Tx/Rd} \\\\",
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
    seoul = _load("results/metrics_v2x_real.npz")

    def cells(res, s, with_sd):
        acc = res[s]["acc"][-TAIL:].mean(); poor = res[s]["poor"][-TAIL:].mean()
        if with_sd:
            return [(acc, res[s]["acc_std"][-TAIL:].mean()),
                    (poor, res[s]["poor_std"][-TAIL:].mean())]
        return [(acc, None), (poor, None)]

    data = {s: cells(intas, s, True) + cells(seoul, s, False) for s in SCHEMES}
    best = [max(data[s][c][0] for s in SCHEMES) for c in range(4)]

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
        " FL (KITTI) over the InTAS (Munich) trace (mean $\\pm$ std over 3"
        " seeds) and over the real Seoul-Gangnam V2X trace ($N{=}180$,"
        " 250 rounds). Accuracies in \\%, averaged over the final"
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
    for s in SCHEMES[:-1]:
        lines.append(row(s))
    lines += ["        \\hline", row("Proposed"), "        \\hline",
              "    \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _seoul_table():
    """Same metric columns as the combined table, for the real Seoul-Gangnam
    V2X trace run (KITTI, N=180, single 250-round run -> no seed std)."""
    res = _load("results/metrics_v2x_real.npz")
    tau = 0.95 * max(res[s]["acc"][-1] for s in SCHEMES)
    K = len(res["Proposed"]["acc"])
    st = {}
    for s in SCHEMES:
        acc = res[s]["acc"][-TAIL:].mean(); poor = res[s]["poor"][-TAIL:].mean()
        reached = res[s]["acc"] >= tau
        rounds = int(np.argmax(reached)) + 1 if reached.any() else None
        st[s] = dict(acc=acc, poor=poor, gap=acc - poor, rounds=rounds,
                     cumtx=int(res[s]["tx"][:rounds].sum()) if rounds else None,
                     txrd=float(res[s]["tx"].mean()))
    best = dict(acc=max(st[s]["acc"] for s in SCHEMES),
                poor=max(st[s]["poor"] for s in SCHEMES),
                gap=min(st[s]["gap"] for s in SCHEMES),
                rounds=min(st[s]["rounds"] for s in SCHEMES if st[s]["rounds"]),
                cumtx=min(st[s]["cumtx"] for s in SCHEMES if st[s]["cumtx"]),
                txrd=min(st[s]["txrd"] for s in SCHEMES))

    def b(txt, is_best):
        return f"\\textbf{{{txt}}}" if is_best else txt

    def row(s):
        e = st[s]
        cells = [
            b(f"{100*e['acc']:.1f}", e["acc"] == best["acc"]),
            b(f"{100*e['poor']:.1f}", e["poor"] == best["poor"]),
            b(f"{100*e['gap']:.1f}", e["gap"] == best["gap"]),
            _fmt_int(e["rounds"], e["rounds"] == best["rounds"], K),
            _fmt_int(e["cumtx"], e["cumtx"] == best["cumtx"]),
            b(f"{e['txrd']:.1f}", e["txrd"] == best["txrd"]),
        ]
        return f"        \\textsc{{{DISPLAY.get(s, s)}}} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Performance on the real Seoul-Gangnam V2X trace"
        " (real multimodal FL on KITTI, $N{=}180$, single 250-round run;"
        f" \\%, averaged over the final {TAIL} rounds;"
        f" $\\tau={100*tau:.1f}\\%$).}}",
        "    \\label{tab:seoul_results}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Method} & \\textsc{Acc} & \\textsc{Poor Acc} &"
        " \\textsc{Gap} & \\textsc{Rounds@$\\tau$} & \\textsc{Tx@$\\tau$} &"
        " \\textsc{Tx/Rd} \\\\",
        "        \\hline",
    ]
    for s in SCHEMES[:-1]:
        lines.append(row(s))
    lines += ["        \\hline", row("Proposed"), "        \\hline",
              "    \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


ABL_VARIANTS = ["w/o caching", "w/o demand", "w/o queue", "FACE (full)"]


def _ablation_table(dataset="kitti", label="KITTI"):
    """Component ablation on the real FL backend (Tables/tab_ablation.tex)."""
    path = f"results/metrics_real_ablation_{dataset}.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    res = {v: {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(v + "__")}
           for v in ABL_VARIANTS}
    st = {}
    for v in ABL_VARIANTS:
        st[v] = dict(acc=res[v]["acc"][-TAIL:].mean(),
                     acc_sd=res[v]["acc_std"][-TAIL:].mean(),
                     poor=res[v]["poor"][-TAIL:].mean(),
                     poor_sd=res[v]["poor_std"][-TAIL:].mean(),
                     txrd=float(res[v]["tx"].mean()))
    full = st["FACE (full)"]
    best_acc = max(st[v]["acc"] for v in ABL_VARIANTS)
    best_poor = max(st[v]["poor"] for v in ABL_VARIANTS)

    def row(v):
        e = st[v]
        dacc = "--" if v == "FACE (full)" else f"{100*(e['acc']-full['acc']):+.1f}"
        cells = [
            _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best_acc),
            dacc,
            _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best_poor),
            f"{e['txrd']:.1f}",
        ]
        return f"        \\textsc{{{v}}} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{Component ablation of FACE (real multimodal FL,"
        f" {label} over InTAS; \\%, mean $\\pm$ std over 3 seeds, averaged"
        f" over the final {TAIL} rounds).}}",
        "    \\label{tab:ablation}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c|c|c}",
        "        \\hline",
        "        \\textsc{Variant} & \\textsc{Acc} & $\\Delta$\\textsc{Acc}"
        " & \\textsc{Poor Acc} & \\textsc{Tx/Rd} \\\\",
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
