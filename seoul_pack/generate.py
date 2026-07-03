"""
Seoul-only paper pack: collects every table and chart based on the real
Seoul-Gangnam V2X trace into this folder (the InTAS/Munich results are kept
elsewhere). Regenerate any time with:  python3 seoul_pack/generate.py
"""

import os
import shutil
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def tab_main():
    """6-scheme main comparison on Seoul (reuses sim.make_tables layout)."""
    from sim.make_tables import _seoul_table
    tex = _seoul_table()
    with open(os.path.join(HERE, "tab_seoul_main.tex"), "w") as f:
        f.write(tex + "\n")


def tab_ablation(tail=20):
    """Seoul-only component ablation (single 250-round paired run)."""
    d = np.load(os.path.join(ROOT, "results/metrics_real_ablation_seoul.npz"))
    V = ["w/o caching", "w/o demand", "w/o queue", "w/o prediction",
         "FACE (full)"]
    st = {v: dict(acc=d[v + "__acc"][-tail:].mean(),
                  poor=d[v + "__poor"][-tail:].mean()) for v in V}
    full = st["FACE (full)"]
    best_acc = max(st[v]["acc"] for v in V)
    best_poor = max(st[v]["poor"] for v in V)

    def row(v):
        e = st[v]
        dacc = "--" if v == "FACE (full)" else f"{100*(e['acc']-full['acc']):+.1f}"
        a = f"{100*e['acc']:.1f}"
        p = f"{100*e['poor']:.1f}"
        if e["acc"] == best_acc:
            a = f"\\textbf{{{a}}}"
        if e["poor"] == best_poor:
            p = f"\\textbf{{{p}}}"
        return f"        \\textsc{{{v}}} & {a} & {dacc} & {p} \\\\"

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Component ablation of FACE on the real Seoul-Gangnam"
        " V2X trace (real multimodal FL on KITTI, $N{=}180$, 250 rounds;"
        f" \\%, averaged over the final {tail} rounds).}}",
        "    \\label{tab:seoul_ablation}",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{4.5pt}",
        "    \\begin{tabular}{c|c|c|c}",
        "        \\hline",
        "        \\textsc{Variant} & \\textsc{Acc} & $\\Delta$\\textsc{Acc}"
        " & \\textsc{Poor Acc} \\\\",
        "        \\hline",
        *[row(v) for v in V[:-1]],
        "        \\hline",
        row("FACE (full)"),
        "        \\hline",
        "    \\end{tabular}",
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
        ("Figures/fig_infocom_v2x_real.png", "fig_seoul_convergence.png"),
        ("Figures/fig_infocom_v2x_real.pdf", "fig_seoul_convergence.pdf"),
        # 1x4 per-vehicle accuracy map on the Seoul basemap
        ("Figures/fig_infocom_v2x_map.png", "fig_seoul_map.png"),
        ("Figures/fig_infocom_v2x_map.pdf", "fig_seoul_map.pdf"),
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
