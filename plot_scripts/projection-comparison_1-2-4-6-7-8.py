"""
projection-comparison_1-2-4-6-7-8.py
──────────────────────────────────────
iFOPNG vs FOPNG cross-benchmark summary. Sub-RQ3 main figure.
Single panel: grouped bars per benchmark.
Output: plots/projection-comparison_1-2-4-6-7-8.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import STYLE, load_exp, skip_exp6, skip_exp7, COLORS

RESULTS = "results/"
OUT     = "plots/"


def _load_pair(fname, skip_fn=None):
    data = load_exp(fname, min_seeds=2, skip_fn=skip_fn)
    out  = {}
    for m in ("ifopng", "fopng"):
        if m in data:
            out[m] = data[m]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(STYLE)

    benchmarks = [
        ("Perm-MNIST\n(Exp 1)",  _load_pair(RESULTS + "401.csv"), "standalone"),
        ("MNIST MH\n(Exp 2)",    _load_pair(RESULTS + "402.csv"), "standalone"),
        ("CIFAR10 MH\n(Exp 4)",  _load_pair(RESULTS + "404.csv"), "standalone"),
        ("CIFAR100 MH\n(Exp 6)", _load_pair(RESULTS + "406.csv", skip_fn=skip_exp6), "standalone"),
        ("MNIST HN\n(Exp 7)",    _load_pair(RESULTS + "407.csv", skip_fn=skip_exp7), "HN"),
        ("CIFAR10 HN\n(Exp 8)",  _load_pair(RESULTS + "408.csv"), "HN"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    fig.suptitle(
        "Sub-RQ3: iFOPNG vs FOPNG — Does parameter inertia improve retention?\n"
        "Standalone benchmarks (left of divider) vs HN benchmarks (right)",
        fontsize=10, fontweight="bold",
    )

    COL_IF = COLORS["ifopng"]
    COL_FP = COLORS["fopng"]
    WIDTH  = 0.32
    x      = np.arange(len(benchmarks))

    for xi, (label, data, btype) in enumerate(benchmarks):
        for offset, key, col in [(-WIDTH / 2, "ifopng", COL_IF),
                                  (+WIDTH / 2, "fopng",  COL_FP)]:
            if key not in data:
                continue
            mean = data[key]["acc_mean"]
            std  = data[key]["acc_std"]
            ax.bar(xi + offset, mean, width=WIDTH, color=col, yerr=std,
                   capsize=3, zorder=3, alpha=0.88,
                   error_kw={"linewidth": 0.8, "ecolor": "#333", "capthick": 0.8})
            pts = data[key]["accs"]
            jit = np.linspace(-0.05, 0.05, len(pts))
            for j, v in zip(jit, pts):
                ax.scatter(xi + offset + j, v, color="white", s=10, zorder=4,
                           edgecolors="#333", linewidths=0.5)

    # Divider between standalone and HN benchmarks
    ax.axvline(3.5, color="#999", lw=1.0, ls="--", alpha=0.6)
    ax.text(3.6, 1.0, "HN ->", fontsize=7.5, color="#666", va="top",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in benchmarks], fontsize=8.5)
    ax.set_ylabel("Avg. Accuracy", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("iFOPNG vs FOPNG per benchmark", fontsize=10,
                 fontweight="bold", pad=6)
    ax.legend(handles=[
        mpatches.Patch(color=COL_IF, label=r"iFOPNG  ($F_c = \hat{F}_\mathrm{new} + \hat{F}_\mathrm{old}$)"),
        mpatches.Patch(color=COL_FP, label=r"FOPNG   ($\hat{F}_\mathrm{new}$ only)"),
    ], fontsize=8.5, loc="lower left")

    plt.tight_layout(pad=1.5)
    plt.savefig(OUT + "projection-comparison_1-2-4-6-7-8.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved projection-comparison_1-2-4-6-7-8.png")


if __name__ == "__main__":
    main()