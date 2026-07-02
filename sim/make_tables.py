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


def _table(res, dataset_label, K_label):
    st, tau, K = _stats(res)
    best_acc = max(st[s]["acc"] for s in SCHEMES)
    best_poor = max(st[s]["poor"] for s in SCHEMES)
    best_rounds = min(st[s]["rounds"] for s in SCHEMES if st[s]["rounds"])
    best_tx = min(st[s]["cumtx"] for s in SCHEMES if st[s]["cumtx"])

    def row(s):
        e = st[s]
        cells = [
            _fmt_pm(e["acc"], e["acc_sd"], e["acc"] == best_acc),
            _fmt_pm(e["poor"], e["poor_sd"], e["poor"] == best_poor),
            _fmt_int(e["rounds"], e["rounds"] == best_rounds, K),
            _fmt_int(e["cumtx"], e["cumtx"] == best_tx),
        ]
        return f"        \\textsc{{{DISPLAY.get(s, s)}}} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{Real multimodal FL on {dataset_label}: test accuracy and"
        f" poor-data vehicle accuracy (\\%, mean $\\pm$ std over 3 seeds, averaged"
        f" over the final {TAIL} rounds), rounds and cumulative encoder"
        f" transmissions to reach $\\tau={100*tau:.1f}\\%$.}}",
        f"    \\label{{tab:real_{K_label}}}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c|c|c}",
        "        \\hline",
        "        \\multirow{2}{*}{\\textsc{Method}} &"
        " \\multicolumn{4}{c}{\\textsc{Performance on " + dataset_label + "}} \\\\",
        "        & \\textsc{Acc} & \\textsc{Poor Acc} &"
        " \\textsc{Rounds@$\\tau$} & \\textsc{Tx@$\\tau$} \\\\",
        "        \\hline",
    ]
    for s in SCHEMES[:-1]:
        lines.append(row(s))
    lines += ["        \\hline", row("Proposed"), "        \\hline",
              "    \\end{tabular}", "\\end{table}"]
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


def main(outdir="Tables"):
    os.makedirs(outdir, exist_ok=True)
    outputs = []
    for tag, label in [("kitti", "KITTI"), ("nuscenes", "nuScenes")]:
        res = _load(f"results/metrics_real_{tag}.npz")
        outputs.append((f"tab_real_{tag}.tex", _table(res, label, tag)))
    outputs.append(("tab_mobility.tex", _mobility_table()))
    for fname, tex in outputs:
        path = os.path.join(outdir, fname)
        with open(path, "w") as f:
            f.write(tex + "\n")
        print(f"--- {path} ---")
        print(tex)
        print()


if __name__ == "__main__":
    main()
