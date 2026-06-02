"""
normalization-ablation_8-9-10.py
──────────────────────────────────
Three-way bar chart: no norm / grad-only norm / full norm.
iFOPNG on Split-CIFAR10 HN. Sub-RQ2 summary figure.

Output: plots/normalization-ablation_8-9-10.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import STYLE, load_exp, skip_exp10_contamination

RESULTS = "results/"
OUT     = "plots/"


def _load_ifopng(fname, skip_fn=None):
    data = load_exp(fname, min_seeds=2, skip_fn=skip_fn)
    return data.get("ifopng", None)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    # Load each condition — iFOPNG only
    d8  = _load_ifopng(RESULTS + "408.csv")                          # full norm
    d9  = _load_ifopng(RESULTS + "409.csv")                          # no norm
    d10 = _load_ifopng(RESULTS + "410.csv",                          # grad-only
                       skip_fn=skip_exp10_contamination)

    conditions = [
        ("No norm\n(Exp 9)",     d9,  "#C94040"),
        ("Grad-only\n(Exp 10)",  d10, "#E07B2A"),
        ("Full norm\n(Exp 8)",   d8,  "#1B6CA8"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(
        "Sub-RQ2: Normalization Ablation — iFOPNG on Split-CIFAR10 HN\n"
        "No normalization vs gradient-only vs full normalization",
        fontsize=10, fontweight="bold",
    )

    for ax_i, (ylabel, metric, ylim, title) in enumerate([
        ("Avg. Accuracy", "acc", (0.35, 1.0), "Average Accuracy"),
        ("BWT",           "bwt", (-0.25, 0.05), "Backward Transfer"),
    ]):
        ax = axes[ax_i]
        for xi, (label, d, col) in enumerate(conditions):
            if d is None:
                ax.bar(xi, 0, color="#eee", width=0.55, zorder=3,
                       hatch="//")
                ax.text(xi, 0.02, "no data", ha="center", fontsize=7,
                        color="#aaa")
                continue
            vals = d[f"{metric}s"]
            if not vals:
                continue
            mean = np.mean(vals); std = np.std(vals)
            ax.bar(xi, mean, yerr=std, color=col, width=0.55, capsize=4,
                   zorder=3, alpha=0.88,
                   error_kw={"linewidth": 1.2, "ecolor": "#333", "capthick": 1.2})
            jit = np.linspace(-0.1, 0.1, len(vals))
            for j, v in zip(jit, vals):
                ax.scatter(xi + j, v, color="white", s=16, zorder=4,
                           edgecolors="#333", linewidths=0.7)
            ax.text(xi, mean + std + 0.01, f"{mean:.3f}",
                    ha="center", fontsize=8, fontweight="bold", color=col)

        ax.axhline(0, color="#333", lw=0.7, ls="--", alpha=0.5)
        if metric == "acc":
            ax.axhline(0.50, color="#888", lw=0.8, ls=":", alpha=0.7)
            ax.text(2.45, 0.51, "random", ha="right", fontsize=7, color="#888")
        ax.set_xticks(range(3))
        ax.set_xticklabels([c[0] for c in conditions])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylim(ylim)

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "normalization-ablation_8-9-10.png")
    plt.close()
    print("Saved normalization-ablation_8-9-10.png")

if __name__ == "__main__":
    main()